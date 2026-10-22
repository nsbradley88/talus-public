//! Piece 1 (`qwen4exp-p1`): the device residency table and the prompt path's loan of cache slots (serve / generate lend + refill).

#![allow(clippy::missing_safety_doc)]

use super::*;

// =====================================================================================================================
// The residency table on the device (d_res) and the prompt path's loan of cache slots (serve `lend` / `refill`)
// =====================================================================================================================

/// `strata::program::split_lend_slots`: how many tail slots of a cache hold `need` bytes (sized slots: walk the
/// offsets from the end; uniform: whole `max_blob` slots).
pub fn split_lend_slots(
    slots: i64,
    bytes: i64,
    offsets: Option<&[u64]>,
    max_blob: u64,
    need: u64,
) -> i64 {
    if let Some(off) = offsets {
        let mut k = 0i64;
        while k < slots && ((bytes - off[(slots - k) as usize] as i64) as u64) < need {
            k += 1;
        }
        return k;
    }
    if max_blob == 0 {
        0
    } else {
        ((need + max_blob - 1) / max_blob) as i64
    }
}

/// generate.cpp `request_chunk`: `tokens` rounded up to the 256-token grid, at most `max_chunk`.
pub fn request_chunk(tokens: i64, max_chunk: i64) -> i64 {
    if tokens <= 0 || max_chunk <= 0 {
        return 0;
    }
    let rounded = if tokens > i64::MAX - 255 {
        tokens
    } else {
        (tokens + 255) / 256 * 256
    };
    max_chunk.min(rounded)
}

/// generate.cpp `equal_chunk` (STRATA_PREFILL_EQUAL=1: equal chunks; otherwise `request_chunk`). `stream_all_min` is
/// `Prefill::stream_all_min_tokens()`.
pub fn serve_equal_chunk(
    tokens: i64,
    max_chunk: i64,
    equal_chunks: bool,
    stream_all_min: i64,
) -> i64 {
    let plain = request_chunk(tokens, max_chunk);
    if !equal_chunks || tokens <= 0 || max_chunk <= 0 {
        return plain;
    }
    let n = tokens / max_chunk + (tokens % max_chunk != 0) as i64;
    let last = tokens - (n - 1) * max_chunk;
    if n <= 1 || last < stream_all_min {
        return plain;
    }
    let per = tokens / n + (tokens % n != 0) as i64;
    let rounded = if per > i64::MAX - 255 {
        per
    } else {
        (per + 255) / 256 * 256
    };
    max_chunk.min(rounded)
}

/// `tk_qwen4exp_p1_lend_ctx_t`: the C++ side of the serve loan (one prompt path borrowing the tail of the one cache).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkQwen4ExpP1LendCtx {
    pub user: *mut c_void,
    /// the C++ residency table (n_layers * n_expert): Rust edits it in place
    pub host_res: *mut i32,
    pub n_res: usize,
    pub n_layers: i64,
    pub n_expert: i64,
    pub d_res: *mut i32,
    pub cache_base: *mut u8,
    pub cache_slots: i64,
    pub cache_bytes: i64,
    pub cache_blob: i64,
    pub slot_offsets: *const u64,
    pub max_blob: u64,
    pub blob_bytes: *const u64,
    /// the part: its first lendable slot for the planned chunk (-1: no loan) and where its buffers are laid out now
    pub first: i32,
    pub first_now: i32,
    pub active: i32,
    pub head_alias: i32,
    pub prefill_chunk: i64,
    pub split_small: i64,
    pub split_small_max: i64,
    pub equal_chunks: i32,
    pub refill_blocking: i32,
    pub stream_all_min: i64,
    pub bytes_needed: Option<unsafe extern "C" fn(user: *mut c_void, chunk: i64) -> u64>,
    pub sp_chunk: Option<unsafe extern "C" fn(user: *mut c_void) -> i64>,
    pub relayout: Option<
        unsafe extern "C" fn(
            user: *mut c_void,
            chunk: i64,
            base: *mut u8,
            bytes: u64,
            err: *mut c_char,
            cap: usize,
        ) -> c_int,
    >,
    pub blob: Option<unsafe extern "C" fn(user: *mut c_void, layer: i32, expert: i32) -> *const u8>,
    pub note_async_read: Option<unsafe extern "C" fn(user: *mut c_void, blob: *const u8)>,
}

/// Uploads the residency table (`res_put`): a blocking copy, then the legacy stream's sync (the table is read by
/// kernels on non-blocking streams).
pub unsafe fn res_put(d_res: *mut i32, host_res: *const i32, n: usize) -> Result<(), String> {
    if d_res.is_null() {
        return Ok(());
    }
    if hip_sys::hipMemcpy(
        d_res.cast(),
        host_res.cast(),
        n * 4,
        hip_sys::hipMemcpyHostToDevice,
    ) != hip_sys::hipSuccess
    {
        return Err("res_put: hipMemcpy failed".into());
    }
    if hip_sys::hipStreamSynchronize(std::ptr::null_mut()) != hip_sys::hipSuccess {
        return Err("res_put: hipStreamSynchronize failed".into());
    }
    Ok(())
}

/// The serve loan's Rust state (generate.cpp `PfPart` for the single-GPU part, `lend` / `refill`).
pub struct ServeLend {
    ctx: TkQwen4ExpP1LendCtx,
    lent: Vec<(i32, i32)>,
    lent_chunk: i64,
    first_now: i32,
    staging: *mut u8,
    staging_bytes: usize,
}

unsafe impl Send for ServeLend {}

impl Drop for ServeLend {
    fn drop(&mut self) {
        if !self.staging.is_null() {
            unsafe {
                let _ = hip_sys::hipHostFree(self.staging.cast());
            }
        }
    }
}

impl ServeLend {
    pub fn new(ctx: TkQwen4ExpP1LendCtx) -> Result<Self, String> {
        let mut s = Self {
            ctx,
            lent: Vec::new(),
            lent_chunk: 0,
            first_now: ctx.first_now,
            staging: std::ptr::null_mut(),
            staging_bytes: 0,
        };
        if ctx.active != 0 && ctx.refill_blocking != 0 {
            let mut p: *mut c_void = std::ptr::null_mut();
            let n = ctx.cache_blob.max(ctx.max_blob as i64) as usize;
            if unsafe { hip_sys::hipHostMalloc(&mut p, n, hip_sys::hipHostMallocDefault) }
                != hip_sys::hipSuccess
            {
                return Err("qwen4exp-p1: the refill staging buffer failed".into());
            }
            s.staging = p.cast();
            s.staging_bytes = n;
        }
        Ok(s)
    }

    pub fn lent_count(&self) -> usize {
        self.lent.len()
    }

    fn offsets(&self) -> Option<&[u64]> {
        if self.ctx.slot_offsets.is_null() {
            None
        } else {
            Some(unsafe {
                std::slice::from_raw_parts(self.ctx.slot_offsets, self.ctx.cache_slots as usize + 1)
            })
        }
    }

    fn host_res(&mut self) -> &mut [i32] {
        unsafe { std::slice::from_raw_parts_mut(self.ctx.host_res, self.ctx.n_res) }
    }

    /// `part_slots`: the tail slots the prompt path's buffers for a `c`-token chunk take.
    fn part_slots(&self, c: i64) -> i64 {
        let need = match self.ctx.bytes_needed {
            Some(f) => unsafe { f(self.ctx.user, c) },
            None => 0,
        };
        split_lend_slots(
            self.ctx.cache_slots,
            self.ctx.cache_bytes,
            self.offsets(),
            self.ctx.max_blob,
            need,
        )
    }

    /// `part_bytes`: the bytes from slot `first` to the end of the cache.
    fn part_bytes(&self, first: i32) -> u64 {
        match self.offsets() {
            Some(off) => (self.ctx.cache_bytes - off[first as usize] as i64) as u64,
            None => (self.ctx.cache_slots - first as i64) as u64 * self.ctx.max_blob,
        }
    }

    fn device_slot(&self, slot: i32) -> *mut u8 {
        if slot < 0 || slot as i64 >= self.ctx.cache_slots {
            return std::ptr::null_mut();
        }
        let o = match self.offsets() {
            Some(off) => off[slot as usize],
            None => slot as u64 * self.ctx.cache_blob as u64,
        };
        unsafe { self.ctx.cache_base.add(o as usize) }
    }

    fn res_upload(&mut self) -> Result<(), String> {
        unsafe { res_put(self.ctx.d_res, self.ctx.host_res, self.ctx.n_res) }
    }

    /// `refill_issue` + `refill_wait`: every lent slot gets its expert back, the table its slot.
    fn refill_one(&mut self) -> Result<(), String> {
        if self.lent.is_empty() {
            return Ok(());
        }
        let ne = self.ctx.n_expert;
        let blocking = self.ctx.refill_blocking != 0;
        let lent = std::mem::take(&mut self.lent);
        for &(i, slot) in &lent {
            let l = i as i64 / ne;
            let b = match self.ctx.blob {
                Some(f) => unsafe { f(self.ctx.user, l as i32, (i as i64 % ne) as i32) },
                None => std::ptr::null(),
            };
            let nb = unsafe { *self.ctx.blob_bytes.add(l as usize) } as i64;
            let n = (if nb > 0 && nb <= self.ctx.cache_blob {
                nb
            } else {
                self.ctx.cache_blob
            }) as usize;
            let dst = self.device_slot(slot);
            if b.is_null() || dst.is_null() {
                self.lent = lent;
                return Err(
                    "ExpertCache::fill_slot: slot outside the arena or the host blob is null"
                        .into(),
                );
            }
            unsafe {
                if blocking {
                    if n > self.staging_bytes {
                        self.lent = lent;
                        return Err(
                            "ExpertCache::fill_slot_blocking: HIP staging buffer is too small"
                                .into(),
                        );
                    }
                    std::ptr::copy_nonoverlapping(b, self.staging, n);
                    if hip_sys::hipMemcpy(
                        dst.cast(),
                        self.staging.cast(),
                        n,
                        hip_sys::hipMemcpyHostToDevice,
                    ) != hip_sys::hipSuccess
                    {
                        self.lent = lent;
                        return Err("ExpertCache::fill_slot_blocking: hipMemcpy failed".into());
                    }
                } else {
                    if hip_sys::hipMemcpyAsync(
                        dst.cast(),
                        b.cast(),
                        n,
                        hip_sys::hipMemcpyHostToDevice,
                        std::ptr::null_mut(),
                    ) != hip_sys::hipSuccess
                    {
                        self.lent = lent;
                        return Err("ExpertCache::fill_slot_queued: hipMemcpyAsync failed".into());
                    }
                    if let Some(f) = self.ctx.note_async_read {
                        f(self.ctx.user, b);
                    }
                }
            }
            self.host_res()[i as usize] = slot;
        }
        if unsafe { hip_sys::hipStreamSynchronize(std::ptr::null_mut()) } != hip_sys::hipSuccess {
            return Err("ExpertCache::sync_queued: hipStreamSynchronize failed".into());
        }
        self.lent_chunk = 0;
        Ok(())
    }

    /// generate's one-shot loan (`prefill_batched`): every slot from `first` on is taken out of the table.
    pub fn take_from(&mut self, first: i32) -> Result<usize, String> {
        let n = (self.ctx.n_layers * self.ctx.n_expert) as usize;
        let mut taken = Vec::new();
        for (i, r) in self.host_res().iter_mut().enumerate().take(n) {
            if *r >= first {
                taken.push((i as i32, *r));
                *r = NOT_RESIDENT;
            }
        }
        let k = taken.len();
        self.lent.extend(taken);
        self.res_upload()?;
        Ok(k)
    }

    /// generate.cpp serve `refill`.
    pub fn refill(&mut self) -> Result<(), String> {
        if self.lent.is_empty() {
            return Ok(());
        }
        self.refill_one()?;
        self.res_upload()
    }

    /// generate.cpp serve `lend`: the slots a `tokens`-token segment's chunk needs are taken out of the table (their
    /// experts become misses), the prompt path is laid out on them.
    pub fn lend(&mut self, tokens: i64) -> Result<(), String> {
        if self.ctx.active == 0 {
            return Ok(());
        }
        let c = self.ctx;
        let want_full = serve_equal_chunk(
            tokens,
            c.prefill_chunk,
            c.equal_chunks != 0,
            c.stream_all_min,
        );
        let want = if c.head_alias != 0 {
            c.prefill_chunk
        } else if c.split_small > 0 && tokens <= c.split_small_max {
            want_full.min(c.split_small)
        } else {
            want_full
        };
        if want <= 0 {
            return Err("prefill: cannot lend buffers for an empty request segment".into());
        }
        let mut any = false;
        if c.first >= 0 {
            let mut skip = false;
            if !self.lent.is_empty() {
                if want <= self.lent_chunk {
                    skip = true;
                } else {
                    self.refill_one()?;
                }
            }
            if !skip {
                let mut want_p = want;
                let mut first = self.first_now;
                if c.head_alias == 0 {
                    let mut need_p = self.part_slots(want_p);
                    while (need_p <= 0 || need_p + 128 > c.cache_slots) && want_p > 256 {
                        want_p = 256i64.max(want_p / 2 / 256 * 256);
                        need_p = self.part_slots(want_p);
                    }
                    if need_p <= 0 || need_p + 128 > c.cache_slots {
                        return Err("prefill: no chunk of this request fits the slots that can be lent (#2090)".into());
                    }
                    first = (c.cache_slots - need_p) as i32;
                }
                let cur = match c.sp_chunk {
                    Some(f) => unsafe { f(c.user) },
                    None => -1,
                };
                if want_p != cur || first != self.first_now {
                    let mut ebuf = [0 as c_char; 512];
                    let ok = match c.relayout {
                        Some(f) => {
                            let rc = unsafe {
                                f(
                                    c.user,
                                    want_p,
                                    self.device_slot(first),
                                    self.part_bytes(first),
                                    ebuf.as_mut_ptr(),
                                    ebuf.len(),
                                )
                            };
                            rc == 0
                        }
                        None => false,
                    };
                    if !ok {
                        let msg = unsafe { CStr::from_ptr(ebuf.as_ptr()) }
                            .to_string_lossy()
                            .into_owned();
                        return Err(if msg.is_empty() {
                            "prefill: relayout failed".into()
                        } else {
                            msg
                        });
                    }
                    self.first_now = first;
                }
                let n = (c.n_layers * c.n_expert) as usize;
                let mut taken = Vec::new();
                {
                    let hr = self.host_res();
                    for (i, r) in hr.iter_mut().enumerate().take(n) {
                        if *r >= first {
                            taken.push((i as i32, *r));
                            *r = NOT_RESIDENT;
                        }
                    }
                }
                any = !taken.is_empty();
                self.lent.extend(taken);
                self.lent_chunk = want_p;
            }
        }
        if any {
            self.res_upload()?;
        }
        Ok(())
    }
}
