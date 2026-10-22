//! qwen4exp-r1: the PLE n-gram table reader in Rust - `strata::kernels::PleTable` (src/kernels/ngram.cpp),
//! `strata::ngram::PleReader` (src/ngram/ple_reader.cpp) and the POSIX `strata::platform::DirectFile`
//! (src/platform/direct_file.cpp) it reads through.
//!
//! `PleTable` finds `per_layer_token_embd.weight` in its GGUF shard (format, scale, the offset checked against the
//! file), then serves rows either from the mapping (`--ple-io mmap` / `ram`, optionally mlocked) or - the default,
//! `--ple-io direct` - through `PleReader`: page-aligned O_DIRECT reads issued by an I/O worker thread into a slab of
//! two-page slots, a bounded 8-way row cache, tickets per token / per prompt chunk, the SSD keep-alive and the
//! fault-injection arm. `DirectFile` is a pool of threads each doing one blocking pread (STRATA_IO_THREADS, default
//! 16; the engine sets 64). The rows are dequantized on the CPU exactly as the C++ table does (IQ4_NL, Q5_0, FP8 with
//! its scale, Q5_1, Q8_0, Q4_0, BF16), so the floats handed to the PLE block are the same bits.
//!
//! The C++ `PleTable` adopts a handle to this object (`PleTable::adopt_rust`, `qwen4exp-r1` lines in ngram.cpp) and
//! forwards every method. Like the C++ table it is not internally synchronized: one thread at a time asks for rows.

#![allow(clippy::missing_safety_doc)]

use core::ffi::{c_int, c_void};
use std::cell::UnsafeCell;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::sys::*;

pub const PLE_HEAD_DIM: usize = 160;
pub const PLE_N_HEADS: usize = 16;
pub const NG_N_EMBD: usize = PLE_HEAD_DIM * PLE_N_HEADS;
pub const PLE_ROW_BYTES: u32 = 90;
pub const PLE_ROW_BYTES_MAX: usize = 320;
const PAGE: u64 = 4096;

/// `strata::platform::now_us`: a steady clock in microseconds.
pub fn now_us() -> f64 {
    static T0: OnceLock<Instant> = OnceLock::new();
    T0.get_or_init(Instant::now).elapsed().as_secs_f64() * 1e6 + 1.0
}

// =====================================================================================================================
// Row formats (the format table of ngram.cpp)
// =====================================================================================================================

/// Exact fp16 bits -> f32 (`f32_from_f16` / `fp16_to_fp32`: every input maps exactly, subnormals included).
pub fn f32_from_f16(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let ex = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let out = if ex == 0 {
        if man == 0 {
            sign
        } else {
            let mut m = man;
            let mut s = 0u32;
            while m & 0x400 == 0 {
                m <<= 1;
                s += 1;
            }
            sign | ((113 - s) << 23) | ((m & 0x3ff) << 13)
        }
    } else if ex == 31 {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((ex - 15 + 127) << 23) | (man << 13)
    };
    f32::from_bits(out)
}

#[inline]
fn rd16(p: &[u8]) -> u16 {
    p[0] as u16 | ((p[1] as u16) << 8)
}

const IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

fn iq4nl_row(row: &[u8], _s: f32, out: &mut [f32]) {
    for b in 0..PLE_HEAD_DIM / 32 {
        let blk = &row[b * 18..];
        let d = f32_from_f16(rd16(blk));
        let qs = &blk[2..];
        for j in 0..16 {
            out[b * 32 + j] = d * IQ4NL[(qs[j] & 0x0f) as usize] as f32;
            out[b * 32 + j + 16] = d * IQ4NL[(qs[j] >> 4) as usize] as f32;
        }
    }
}

fn q5_0_row(row: &[u8], _s: f32, out: &mut [f32]) {
    for b in 0..PLE_HEAD_DIM / 32 {
        let blk = &row[b * 22..];
        let d = f32_from_f16(rd16(blk));
        let h = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
        let qs = &blk[6..];
        let o = &mut out[b * 32..];
        for j in 0..16 {
            let x0 = (qs[j] & 0x0f) as i32 | ((((h >> j) & 1) << 4) as i32);
            let x1 = (qs[j] >> 4) as i32 | ((((h >> (j + 16)) & 1) << 4) as i32);
            o[j] = (x0 - 16) as f32 * d;
            o[j + 16] = (x1 - 16) as f32 * d;
        }
    }
}

fn q5_1_row(row: &[u8], _s: f32, out: &mut [f32]) {
    for b in 0..PLE_HEAD_DIM / 32 {
        let blk = &row[b * 24..];
        let d = f32_from_f16(rd16(blk));
        let m = f32_from_f16(rd16(&blk[2..]));
        let h = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
        let qs = &blk[8..];
        let o = &mut out[b * 32..];
        for j in 0..16 {
            let x0 = (qs[j] & 0x0f) as i32 | ((((h >> j) & 1) << 4) as i32);
            let x1 = (qs[j] >> 4) as i32 | ((((h >> (j + 16)) & 1) << 4) as i32);
            o[j] = x0 as f32 * d + m;
            o[j + 16] = x1 as f32 * d + m;
        }
    }
}

fn q8_0_row(row: &[u8], _s: f32, out: &mut [f32]) {
    for b in 0..PLE_HEAD_DIM / 32 {
        let blk = &row[b * 34..];
        let d = f32_from_f16(rd16(blk));
        for j in 0..32 {
            out[b * 32 + j] = (blk[2 + j] as i8) as f32 * d;
        }
    }
}

fn q4_0_row(row: &[u8], _s: f32, out: &mut [f32]) {
    for b in 0..PLE_HEAD_DIM / 32 {
        let blk = &row[b * 18..];
        let d = f32_from_f16(rd16(blk));
        let qs = &blk[2..];
        for j in 0..16 {
            out[b * 32 + j] = ((qs[j] & 0x0f) as i32 - 8) as f32 * d;
            out[b * 32 + j + 16] = ((qs[j] >> 4) as i32 - 8) as f32 * d;
        }
    }
}

fn pow2(k: i32) -> f32 {
    f32::from_bits(((127 + k) as u32) << 23)
}

fn fp8_table() -> &'static [f32; 256] {
    static T: OnceLock<[f32; 256]> = OnceLock::new();
    T.get_or_init(|| {
        let mut v = [0f32; 256];
        for (x, o) in v.iter_mut().enumerate() {
            let (s, e, m) = (x >> 7, ((x >> 3) & 15) as i32, (x & 7) as i32);
            let f = if e == 15 && m == 7 {
                0.0
            } else if e == 0 {
                (m as f32 / 8.0) * pow2(-6)
            } else {
                (1.0 + m as f32 / 8.0) * pow2(e - 7)
            };
            *o = if s != 0 { -f } else { f };
        }
        v
    })
}

fn fp8_row(row: &[u8], scale: f32, out: &mut [f32]) {
    let t = fp8_table();
    for j in 0..PLE_HEAD_DIM {
        out[j] = t[row[j] as usize] * scale;
    }
}

fn bf16_row(row: &[u8], _s: f32, out: &mut [f32]) {
    for j in 0..PLE_HEAD_DIM {
        let bits = row[2 * j] as u32 | ((row[2 * j + 1] as u32) << 8);
        out[j] = f32::from_bits(bits << 16);
    }
}

/// `strata::kernels::PleFormatInfo`.
pub struct PleFormat {
    pub index: u32,
    pub name: &'static str,
    pub gguf_type: &'static str,
    pub row_bytes: u32,
    pub needs_scale: bool,
    pub dequant: fn(&[u8], f32, &mut [f32]),
}

pub static PLE_FORMATS: [PleFormat; 7] = [
    PleFormat {
        index: 0,
        name: "IQ4_NL",
        gguf_type: "IQ4_NL",
        row_bytes: 90,
        needs_scale: false,
        dequant: iq4nl_row,
    },
    PleFormat {
        index: 1,
        name: "Q5_0",
        gguf_type: "Q5_0",
        row_bytes: 110,
        needs_scale: false,
        dequant: q5_0_row,
    },
    PleFormat {
        index: 2,
        name: "F8_E4M3",
        gguf_type: "I8",
        row_bytes: 160,
        needs_scale: true,
        dequant: fp8_row,
    },
    PleFormat {
        index: 3,
        name: "Q5_1",
        gguf_type: "Q5_1",
        row_bytes: 120,
        needs_scale: false,
        dequant: q5_1_row,
    },
    PleFormat {
        index: 4,
        name: "Q8_0",
        gguf_type: "Q8_0",
        row_bytes: 170,
        needs_scale: false,
        dequant: q8_0_row,
    },
    PleFormat {
        index: 5,
        name: "Q4_0",
        gguf_type: "Q4_0",
        row_bytes: 90,
        needs_scale: false,
        dequant: q4_0_row,
    },
    PleFormat {
        index: 6,
        name: "BF16",
        gguf_type: "BF16",
        row_bytes: 320,
        needs_scale: false,
        dequant: bf16_row,
    },
];

fn format_list() -> String {
    let mut out = String::new();
    for (i, f) in PLE_FORMATS.iter().enumerate() {
        if i > 0 {
            out.push_str(if i + 1 == PLE_FORMATS.len() {
                " or "
            } else {
                ", "
            });
        }
        out.push_str(if f.gguf_type == "I8" {
            "FP8 (I8)"
        } else {
            f.name
        });
    }
    out
}

/// The GGUF type name `TensorInfo::type_name()` gives (the names the format table keys on).
fn gguf_type_name(t: talus_core_api::QuantType) -> String {
    t.name().to_string()
}

// =====================================================================================================================
// DirectFile (POSIX): O_DIRECT reads on a pool of threads, one blocking pread each
// =====================================================================================================================

#[derive(Clone, Copy, Default, Debug)]
pub struct Completion {
    pub tag: u64,
    pub bytes: u32,
    pub ok: bool,
}

pub const WAKE_TAG: u64 = u64::MAX;

#[derive(Clone, Copy)]
struct Pending {
    offset: u64,
    buffer: *mut u8,
    length: u32,
    tag: u64,
}
unsafe impl Send for Pending {}

#[derive(Default)]
struct DfState {
    queue: VecDeque<Pending>,
    done: VecDeque<Completion>,
    stop: bool,
}

struct DfShared {
    fd: c_int,
    mu: Mutex<DfState>,
    cv_work: Condvar,
    cv_done: Condvar,
}

fn lk<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// `strata::platform::DirectFile` (POSIX).
pub struct DirectFile {
    shared: Option<Arc<DfShared>>,
    size: u64,
    pool: Vec<JoinHandle<()>>,
}

fn io_threads(dflt: i64) -> usize {
    let n = env("STRATA_IO_THREADS").map(|v| c_atoi(&v)).unwrap_or(dflt);
    n.clamp(1, 64) as usize
}

impl DirectFile {
    fn new() -> Self {
        Self {
            shared: None,
            size: 0,
            pool: Vec::new(),
        }
    }

    fn open(&mut self, path: &str, pin_aux: Option<unsafe extern "C" fn()>) -> Result<(), String> {
        self.close();
        let fd = open_path(path, O_RDONLY | O_DIRECT);
        if fd < 0 {
            return Err(format!("DirectFile: cannot open {path}"));
        }
        let Some(size) = file_size(path) else {
            unsafe { close(fd) };
            return Err(format!("DirectFile: cannot size {path}"));
        };
        self.size = size;
        let sh = Arc::new(DfShared {
            fd,
            mu: Mutex::new(DfState::default()),
            cv_work: Condvar::new(),
            cv_done: Condvar::new(),
        });
        for _ in 0..io_threads(16) {
            let s = sh.clone();
            let pin = pin_aux.map(|f| f as usize);
            self.pool.push(std::thread::spawn(move || {
                if let Some(f) = pin {
                    let f: unsafe extern "C" fn() = unsafe { std::mem::transmute(f) };
                    unsafe { f() };
                }
                let mut g = lk(&s.mu);
                loop {
                    g = s
                        .cv_work
                        .wait_while(g, |st| !st.stop && st.queue.is_empty())
                        .unwrap_or_else(|e| e.into_inner());
                    if g.stop && g.queue.is_empty() {
                        return;
                    }
                    let p = g.queue.pop_front().unwrap();
                    drop(g);
                    let got =
                        unsafe { pread(s.fd, p.buffer.cast(), p.length as usize, p.offset as i64) };
                    g = lk(&s.mu);
                    g.done.push_back(Completion {
                        tag: p.tag,
                        bytes: if got < 0 { 0 } else { got as u32 },
                        ok: got >= 0,
                    });
                    s.cv_done.notify_all();
                }
            }));
        }
        self.shared = Some(sh);
        Ok(())
    }

    fn close(&mut self) {
        if let Some(sh) = &self.shared {
            lk(&sh.mu).stop = true;
            sh.cv_work.notify_all();
            for t in self.pool.drain(..) {
                let _ = t.join();
            }
            unsafe { close(sh.fd) };
        }
        self.shared = None;
        self.size = 0;
    }

    fn is_open(&self) -> bool {
        self.shared.is_some()
    }

    fn submit(
        &self,
        offset: u64,
        buffer: *mut u8,
        length: u32,
        tag: u64,
        err: &mut String,
    ) -> bool {
        if offset % PAGE != 0
            || length as u64 % PAGE != 0
            || buffer as u64 % PAGE != 0
            || length == 0
        {
            *err = "DirectFile: unaligned request".into();
            return false;
        }
        let Some(sh) = &self.shared else {
            *err = "DirectFile: not open".into();
            return false;
        };
        lk(&sh.mu).queue.push_back(Pending {
            offset,
            buffer,
            length,
            tag,
        });
        sh.cv_work.notify_one();
        true
    }

    fn wake(&self) {
        if let Some(sh) = &self.shared {
            lk(&sh.mu).done.push_back(Completion {
                tag: WAKE_TAG,
                bytes: 0,
                ok: true,
            });
            sh.cv_done.notify_all();
        }
    }

    fn wait(&self, out: &mut Vec<Completion>, max: usize, timeout_ms: i32) {
        out.clear();
        let Some(sh) = &self.shared else { return };
        let mut g = lk(&sh.mu);
        if g.done.is_empty() && timeout_ms != 0 {
            if timeout_ms < 0 {
                g = sh
                    .cv_done
                    .wait_while(g, |s| s.done.is_empty())
                    .unwrap_or_else(|e| e.into_inner());
            } else {
                g = sh
                    .cv_done
                    .wait_timeout_while(g, Duration::from_millis(timeout_ms as u64), |s| {
                        s.done.is_empty()
                    })
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
            }
        }
        while out.len() < max {
            let Some(c) = g.done.pop_front() else { break };
            out.push(c);
        }
    }
}

impl Drop for DirectFile {
    fn drop(&mut self) {
        self.close();
    }
}

// =====================================================================================================================
// PleReader
// =====================================================================================================================

/// `strata::ngram::ReaderStats`.
#[derive(Clone, Debug, Default)]
pub struct ReaderStats {
    pub requests: u64,
    pub cache_hits: u64,
    pub dedup_rows: u64,
    pub reads: u64,
    pub bytes: u64,
    pub wait_us: f64,
    pub submit_us: f64,
    pub read_us_sum: f64,
    pub late_injected: u64,
    pub keepalive_reads: u64,
    pub keepalive_us_max: f64,
    pub read_us: Vec<f32>,
}

impl ReaderStats {
    pub fn percentile(&self, q: f64) -> f64 {
        if self.read_us.is_empty() {
            return 0.0;
        }
        let mut v = self.read_us.clone();
        let k = (v.len() - 1).min((q * (v.len() - 1) as f64 + 0.5) as usize);
        v.select_nth_unstable_by(k, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        v[k] as f64
    }
}

const LATENCY_RING: usize = 65536;
const WAYS: u64 = 8;
const EMPTY: u32 = u32::MAX;

#[derive(Default)]
struct RowCache {
    sets: u64,
    rb: u32,
    keys: Vec<u32>,
    data: Vec<u8>,
    next: Vec<u8>,
    used: u64,
}

impl RowCache {
    fn init(&mut self, rows: u64, rb: u32) {
        self.sets = rows / WAYS;
        self.rb = rb;
        self.keys = vec![EMPTY; (self.sets * WAYS) as usize];
        self.data = vec![0; (self.sets * WAYS * rb as u64) as usize];
        self.next = vec![0; self.sets as usize];
        self.used = 0;
    }
    fn mix(r: u32) -> u64 {
        let x = (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        x ^ (x >> 29)
    }
    fn find(&self, row: u32) -> Option<usize> {
        if self.sets == 0 {
            return None;
        }
        let s = Self::mix(row) % self.sets;
        (0..WAYS)
            .find(|w| self.keys[(s * WAYS + w) as usize] == row)
            .map(|w| ((s * WAYS + w) * self.rb as u64) as usize)
    }
    fn insert(&mut self, row: u32, bytes: *const u8) {
        if self.sets == 0 || self.find(row).is_some() {
            return;
        }
        let s = Self::mix(row) % self.sets;
        let w = self.next[s as usize] as u64;
        self.next[s as usize] = ((w + 1) % WAYS) as u8;
        let k = (s * WAYS + w) as usize;
        if self.keys[k] == EMPTY {
            self.used += 1;
        }
        self.keys[k] = row;
        let at = k * self.rb as usize;
        unsafe {
            std::ptr::copy_nonoverlapping(bytes, self.data.as_mut_ptr().add(at), self.rb as usize)
        };
    }
}

#[derive(Clone, Copy)]
struct Use {
    row: u32,
    in_page: u32,
    dst: *mut u8,
}

#[derive(Default)]
struct Job {
    offset: u64,
    length: u32,
    ticket: u32,
    uses: Vec<Use>,
    issued_us: f64,
    keepalive: bool,
}

struct ReaderInner {
    table_offset: u64,
    n_rows: u64,
    row_bytes: u32,
    max_inflight: u32,
    slab: *mut u8,
    free_slots: Vec<u32>,
    inflight: Vec<Job>,
    delayed: Vec<Completion>,
    queue: VecDeque<Job>,
    tickets: HashMap<u32, u32>,
    next_ticket: u32,
    cache: RowCache,
    stats: ReaderStats,
    ring_pos: usize,
    delay_us: f64,
    error: String,
    keep_us: f64,
    keep_window_us: f64,
    last_issue_us: f64,
    last_read_us: f64,
    rng: u64,
    threaded: bool,
    stop: bool,
}

unsafe impl Send for ReaderInner {}

impl Default for ReaderInner {
    fn default() -> Self {
        Self {
            table_offset: 0,
            n_rows: 0,
            row_bytes: PLE_ROW_BYTES,
            max_inflight: 0,
            slab: std::ptr::null_mut(),
            free_slots: Vec::new(),
            inflight: Vec::new(),
            delayed: Vec::new(),
            queue: VecDeque::new(),
            tickets: HashMap::new(),
            next_ticket: 1,
            cache: RowCache::default(),
            stats: ReaderStats::default(),
            ring_pos: 0,
            delay_us: 0.0,
            error: String::new(),
            keep_us: 0.0,
            keep_window_us: 0.0,
            last_issue_us: 0.0,
            last_read_us: 0.0,
            rng: 0x9E37_79B9_7F4A_7C15,
            threaded: false,
            stop: false,
        }
    }
}

impl ReaderInner {
    fn slot_buf(&self, s: u32) -> *mut u8 {
        unsafe { self.slab.add(s as usize * 2 * PAGE as usize) }
    }
    fn busy(&self) -> bool {
        (self.free_slots.len() as u32) < self.max_inflight || !self.delayed.is_empty()
    }
    fn dec_ticket(&mut self, t: u32) {
        if let Some(p) = self.tickets.get_mut(&t) {
            if *p > 0 {
                *p -= 1;
            }
        }
    }
    fn cancel_queued(&mut self) {
        let q: Vec<u32> = self.queue.iter().map(|j| j.ticket).collect();
        for t in q {
            self.dec_ticket(t);
        }
        self.queue.clear();
    }
    fn record_latency(&mut self, us: f64) {
        self.stats.read_us_sum += us;
        if self.stats.read_us.len() < LATENCY_RING {
            self.stats.read_us.push(us as f32);
        } else {
            let i = self.ring_pos % LATENCY_RING;
            self.stats.read_us[i] = us as f32;
            self.ring_pos += 1;
        }
    }
    fn pump(&mut self, file: &DirectFile) -> bool {
        while !self.queue.is_empty() && !self.free_slots.is_empty() {
            let s = self.free_slots.pop().unwrap();
            let job = self.queue.pop_front().unwrap();
            self.inflight[s as usize] = job;
            let now = now_us();
            self.inflight[s as usize].issued_us = now;
            let (off, len, keep) = (
                self.inflight[s as usize].offset,
                self.inflight[s as usize].length,
                self.inflight[s as usize].keepalive,
            );
            let mut kerr = String::new();
            let mut err = String::new();
            let ok = file.submit(
                off,
                self.slot_buf(s),
                len,
                s as u64,
                if keep { &mut kerr } else { &mut err },
            );
            if !keep && !err.is_empty() {
                self.error = err;
            }
            if !ok {
                if keep {
                    self.keep_us = 0.0;
                    self.inflight[s as usize].keepalive = false;
                    self.free_slots.push(s);
                    continue;
                }
                self.inflight[s as usize].uses.clear();
                let t = self.inflight[s as usize].ticket;
                self.dec_ticket(t);
                self.free_slots.push(s);
                self.cancel_queued();
                return false;
            }
            self.last_read_us = now;
            if !keep {
                self.stats.submit_us += now_us() - now;
                self.stats.reads += 1;
            }
        }
        true
    }
    fn keepalive_due(&self, now: f64) -> f64 {
        if self.keep_us <= 0.0
            || !self.error.is_empty()
            || self.last_issue_us <= 0.0
            || now - self.last_issue_us > self.keep_window_us
        {
            return -1.0;
        }
        self.last_read_us + self.keep_us
    }
    fn queue_keepalive(&mut self) {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        let first = self.table_offset / PAGE;
        let end = (self.table_offset + self.n_rows * self.row_bytes as u64) / PAGE;
        let off = (first
            + if end > first {
                self.rng % (end - first)
            } else {
                0
            })
            * PAGE;
        self.queue.push_back(Job {
            offset: off,
            length: PAGE as u32,
            keepalive: true,
            ..Job::default()
        });
    }
    fn finish(&mut self, c: &Completion, file: &DirectFile) -> bool {
        let s = c.tag as u32;
        if s as usize >= self.inflight.len() {
            self.error = "PleReader: invalid table-read completion".into();
            self.cancel_queued();
            return false;
        }
        if self.inflight[s as usize].keepalive {
            if c.ok {
                self.stats.keepalive_reads += 1;
                self.stats.keepalive_us_max = self
                    .stats
                    .keepalive_us_max
                    .max(now_us() - self.inflight[s as usize].issued_us);
            } else {
                self.keep_us = 0.0;
            }
            self.inflight[s as usize].keepalive = false;
            self.free_slots.push(s);
            return if self.error.is_empty() {
                self.pump(file)
            } else {
                true
            };
        }
        let ticket = self.inflight[s as usize].ticket;
        if !c.ok {
            self.error = "PleReader: a table read failed".into();
            self.cancel_queued();
            self.inflight[s as usize].uses.clear();
            self.dec_ticket(ticket);
            self.free_slots.push(s);
            return false;
        }
        let lat = now_us() - self.inflight[s as usize].issued_us;
        self.record_latency(lat);
        self.stats.bytes += c.bytes as u64;
        let buf = self.slot_buf(s);
        let rb = self.row_bytes;
        if self.inflight[s as usize]
            .uses
            .iter()
            .any(|u| u.in_page + rb > c.bytes)
        {
            self.error = "PleReader: short read inside the table".into();
            self.cancel_queued();
            self.inflight[s as usize].uses.clear();
            self.dec_ticket(ticket);
            self.free_slots.push(s);
            return false;
        }
        let uses = std::mem::take(&mut self.inflight[s as usize].uses);
        for u in &uses {
            unsafe {
                std::ptr::copy_nonoverlapping(buf.add(u.in_page as usize), u.dst, rb as usize);
                self.cache.insert(u.row, buf.add(u.in_page as usize));
            }
        }
        self.dec_ticket(ticket);
        self.free_slots.push(s);
        if self.error.is_empty() {
            self.pump(file)
        } else {
            true
        }
    }
    fn process(&mut self, got: &[Completion], file: &DirectFile) -> bool {
        let mut ok = true;
        for c in got {
            if c.tag == WAKE_TAG {
                continue;
            }
            if c.tag as usize >= self.inflight.len() {
                self.error = "PleReader: invalid table-read completion".into();
                self.cancel_queued();
                ok = false;
                continue;
            }
            if self.delay_us > 0.0
                && now_us() - self.inflight[c.tag as usize].issued_us < self.delay_us
            {
                self.delayed.push(*c);
                self.stats.late_injected += 1;
                continue;
            }
            if !self.finish(c, file) {
                ok = false;
            }
        }
        ok
    }
    fn release_delayed(&mut self, file: &DirectFile) -> bool {
        let now = now_us();
        let mut i = 0;
        while i < self.delayed.len() {
            if now - self.inflight[self.delayed[i].tag as usize].issued_us >= self.delay_us {
                let c = self.delayed.remove(i);
                if !self.finish(&c, file) {
                    return false;
                }
            } else {
                i += 1;
            }
        }
        true
    }
    fn drain(&mut self, mut timeout_ms: i32, file: &DirectFile) -> bool {
        if !self.delayed.is_empty() {
            if !self.release_delayed(file) {
                return false;
            }
            timeout_ms = 0;
        }
        let mut got = Vec::new();
        file.wait(&mut got, 64, timeout_ms);
        self.process(&got, file)
    }
}

struct ReaderShared {
    file: DirectFile,
    mu: Mutex<ReaderInner>,
    cv_work: Condvar,
    cv_done: Condvar,
}

/// `strata::ngram::PleReader`.
pub struct PleReader {
    sh: Option<Arc<ReaderShared>>,
    worker: Option<JoinHandle<()>>,
}

fn worker_loop(sh: Arc<ReaderShared>) {
    let mut g = lk(&sh.mu);
    loop {
        while !(g.stop || !g.queue.is_empty() || g.busy()) {
            let now = now_us();
            let due = g.keepalive_due(now);
            if due < 0.0 {
                g = sh.cv_work.wait(g).unwrap_or_else(|e| e.into_inner());
            } else if now >= due {
                g.queue_keepalive();
            } else {
                let us = 1000f64.max(due - now) as u64;
                g = sh
                    .cv_work
                    .wait_timeout(g, Duration::from_micros(us))
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
            }
        }
        if g.stop && !g.busy() {
            break;
        }
        if g.error.is_empty() && !g.pump(&sh.file) && g.error.is_empty() {
            g.error = "PleReader: submit failed".into();
        }
        if !g.delayed.is_empty() && !g.release_delayed(&sh.file) && g.error.is_empty() {
            g.error = "PleReader: read failed".into();
        }
        if !g.error.is_empty() {
            sh.cv_done.notify_all();
            if !g.busy() {
                let _g = sh
                    .cv_work
                    .wait_while(g, |s| !s.stop)
                    .unwrap_or_else(|e| e.into_inner());
                return;
            }
        }
        if !g.busy() {
            sh.cv_done.notify_all();
            continue;
        }
        let timeout = if g.delayed.is_empty() { -1 } else { 0 };
        drop(g);
        let mut got = Vec::new();
        sh.file.wait(&mut got, 64, timeout);
        g = lk(&sh.mu);
        if !g.process(&got, &sh.file) && g.error.is_empty() {
            g.error = "PleReader: read failed".into();
        }
        sh.cv_done.notify_all();
    }
}

impl PleReader {
    pub fn new() -> Self {
        Self {
            sh: None,
            worker: None,
        }
    }

    pub fn is_open(&self) -> bool {
        self.sh.as_ref().is_some_and(|s| s.file.is_open())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn open(
        &mut self,
        path: &str,
        table_offset: u64,
        n_rows: u64,
        max_inflight: u32,
        cache_rows: u64,
        io_thread: bool,
        row_bytes: u32,
        pin_aux: Option<unsafe extern "C" fn()>,
    ) -> Result<(), String> {
        self.close();
        if max_inflight == 0 || max_inflight > 1024 {
            return Err("PleReader: max_inflight must be 1..1024".into());
        }
        if row_bytes == 0 || row_bytes as u64 > PAGE {
            return Err("PleReader: row_bytes must be 1..4096".into());
        }
        let mut file = DirectFile::new();
        file.open(path, pin_aux)?;
        if table_offset + n_rows * row_bytes as u64 > file.size {
            return Err(format!(
                "PleReader: the table extends past the end of {path}"
            ));
        }
        let mut slab: *mut c_void = std::ptr::null_mut();
        if unsafe {
            posix_memalign(
                &mut slab,
                PAGE as usize,
                max_inflight as usize * 2 * PAGE as usize,
            )
        } != 0
        {
            return Err("PleReader: cannot allocate read buffers".into());
        }
        let mut inner = ReaderInner {
            table_offset,
            n_rows,
            row_bytes,
            max_inflight,
            slab: slab.cast(),
            inflight: (0..max_inflight).map(|_| Job::default()).collect(),
            free_slots: (0..max_inflight).rev().collect(),
            threaded: io_thread,
            ..ReaderInner::default()
        };
        inner.cache.init(cache_rows, row_bytes);
        inner.rng = 0x9E37_79B9_7F4A_7C15 ^ (now_us() as u64);
        if inner.rng == 0 {
            inner.rng = 1;
        }
        let sh = Arc::new(ReaderShared {
            file,
            mu: Mutex::new(inner),
            cv_work: Condvar::new(),
            cv_done: Condvar::new(),
        });
        if io_thread {
            let s = sh.clone();
            let pin = pin_aux.map(|f| f as usize);
            self.worker = Some(std::thread::spawn(move || {
                if let Some(f) = pin {
                    let f: unsafe extern "C" fn() = unsafe { std::mem::transmute(f) };
                    unsafe { f() };
                }
                worker_loop(s);
            }));
        }
        self.sh = Some(sh);
        Ok(())
    }

    pub fn close(&mut self) {
        let Some(sh) = self.sh.take() else { return };
        if let Some(w) = self.worker.take() {
            {
                let mut g = lk(&sh.mu);
                g.stop = true;
                g.queue.clear();
            }
            sh.cv_work.notify_all();
            sh.file.wake();
            let _ = w.join();
        }
        // outstanding reads finish before their buffers are released
        {
            let mut g = lk(&sh.mu);
            let mut got = Vec::new();
            while (g.free_slots.len() as u32) < g.max_inflight {
                sh.file.wait(&mut got, 64, -1);
                if got.is_empty() {
                    break;
                }
                for c in &got {
                    if c.tag != WAKE_TAG {
                        g.free_slots.push(c.tag as u32);
                    }
                }
            }
            unsafe { free(g.slab.cast()) };
            g.slab = std::ptr::null_mut();
        }
        if let Ok(mut s) = Arc::try_unwrap(sh) {
            s.file.close();
        }
    }

    pub fn set_keepalive(&self, period_ms: f64, window_s: f64) {
        let Some(sh) = &self.sh else { return };
        {
            let mut g = lk(&sh.mu);
            g.keep_us = if g.threaded && period_ms > 0.0 {
                period_ms * 1000.0
            } else {
                0.0
            };
            g.keep_window_us = if window_s > 0.0 { window_s * 1e6 } else { 0.0 };
        }
        sh.cv_work.notify_all();
    }

    pub fn set_injected_delay_us(&self, us: f64) {
        if let Some(sh) = &self.sh {
            lk(&sh.mu).delay_us = if us < 0.0 { 0.0 } else { us };
        }
    }

    /// `issue`: row i's raw bytes land at `out_raw + row_bytes * i` (zero for an out-of-range row).
    pub unsafe fn issue(&self, rows: &[u32], out_raw: *mut u8) -> u32 {
        let Some(sh) = &self.sh else { return 0 };
        let mut g = lk(&sh.mu);
        let id = g.next_ticket;
        g.next_ticket = g.next_ticket.wrapping_add(1);
        if g.next_ticket == 0 {
            g.next_ticket = 1;
        }
        let now = now_us();
        let rearm =
            g.keep_us > 0.0 && (g.last_issue_us <= 0.0 || now - g.last_issue_us > g.keep_window_us);
        g.last_issue_us = now;
        g.tickets.insert(id, 0);
        let mut by_page: HashMap<u64, usize> = HashMap::new();
        let mut jobs: Vec<Job> = Vec::new();
        let rb = g.row_bytes;
        for (i, &row) in rows.iter().enumerate() {
            let dst = out_raw.add(i * rb as usize);
            g.stats.requests += 1;
            if row as u64 >= g.n_rows {
                std::ptr::write_bytes(dst, 0, rb as usize);
                continue;
            }
            if let Some(at) = g.cache.find(row) {
                std::ptr::copy_nonoverlapping(g.cache.data.as_ptr().add(at), dst, rb as usize);
                g.stats.cache_hits += 1;
                continue;
            }
            let at = g.table_offset + row as u64 * rb as u64;
            let first = at / PAGE * PAGE;
            let length = ((at + rb as u64 - 1) / PAGE * PAGE - first + PAGE) as u32;
            if let Some(&f) = by_page.get(&first) {
                let j = &mut jobs[f];
                j.length = j.length.max(length);
                j.uses.push(Use {
                    row,
                    in_page: (at - first) as u32,
                    dst,
                });
                g.stats.dedup_rows += 1;
                continue;
            }
            by_page.insert(first, jobs.len());
            jobs.push(Job {
                offset: first,
                length,
                ticket: id,
                uses: vec![Use {
                    row,
                    in_page: (at - first) as u32,
                    dst,
                }],
                ..Job::default()
            });
        }
        jobs.sort_by_key(|j| j.offset);
        let n_jobs = jobs.len() as u32;
        g.tickets.insert(id, n_jobs);
        let has_jobs = n_jobs > 0;
        g.queue.extend(jobs);
        if g.threaded {
            drop(g);
            if has_jobs {
                sh.cv_work.notify_one();
                sh.file.wake();
            } else if rearm {
                sh.cv_work.notify_one();
            }
        } else if !g.pump(&sh.file) && g.error.is_empty() {
            g.error = "PleReader: submit failed".into();
        }
        id
    }

    /// `collect`: blocks until every row of the ticket is in its buffer.
    pub fn collect(&self, t: u32) -> Result<(), String> {
        let Some(sh) = &self.sh else {
            return Err("PleReader: unknown ticket".into());
        };
        let start = now_us();
        let mut g = lk(&sh.mu);
        if !g.tickets.contains_key(&t) {
            return Err("PleReader: unknown ticket".into());
        }
        if g.threaded {
            g = sh
                .cv_done
                .wait_while(g, |s| s.tickets.get(&t).is_some_and(|&p| p != 0))
                .unwrap_or_else(|e| e.into_inner());
            if !g.tickets.contains_key(&t) {
                return Err("PleReader: unknown ticket".into());
            }
            if !g.error.is_empty() {
                return Err(g.error.clone());
            }
            g.stats.wait_us += now_us() - start;
            g.tickets.remove(&t);
            return Ok(());
        }
        while g.tickets.get(&t).is_some_and(|&p| p > 0) {
            let drained = g.drain(-1, &sh.file);
            if !drained && g.error.is_empty() {
                g.error = "PleReader: read failed".into();
            }
            if !g.tickets.contains_key(&t) {
                return Err("PleReader: unknown ticket".into());
            }
        }
        if !g.error.is_empty() {
            return Err(g.error.clone());
        }
        g.stats.wait_us += now_us() - start;
        g.tickets.remove(&t);
        Ok(())
    }

    pub fn snapshot(&self) -> ReaderStats {
        self.sh
            .as_ref()
            .map(|s| lk(&s.mu).stats.clone())
            .unwrap_or_default()
    }
    pub fn cache_capacity(&self) -> u64 {
        self.sh
            .as_ref()
            .map(|s| lk(&s.mu).cache.sets * WAYS)
            .unwrap_or(0)
    }
    pub fn cache_size(&self) -> u64 {
        self.sh.as_ref().map(|s| lk(&s.mu).cache.used).unwrap_or(0)
    }
}

impl Default for PleReader {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for PleReader {
    fn drop(&mut self) {
        self.close();
    }
}

// =====================================================================================================================
// PleTable
// =====================================================================================================================

/// `PleIoOptions` (the p1 plan's `tk_qwen4exp_p1_ple_io_t` fields).
#[derive(Clone, Copy, Debug)]
pub struct PleIoOptions {
    pub mmap: bool,
    pub lock: bool,
    pub max_inflight: u32,
    pub cache_rows: u64,
    pub io_thread: bool,
    pub keepalive_ms: f64,
    pub keepalive_window_s: f64,
}

const MAX_PREFETCH: usize = 16;

struct TableInner {
    file: Option<talus_gguf::GgufFile>,
    data: *const u8,
    n_rows: u64,
    fmt: &'static PleFormat,
    bytes_read: u64,
    mmap_mode: bool,
    reader: PleReader,
    ticket: u32,
    pending: bool,
    locked: bool,
    lock_range: (usize, usize),
    rows: [u32; PLE_N_HEADS],
    raw: Vec<u8>,
    n_prefetch: usize,
    prefetch_tickets: [u32; MAX_PREFETCH],
    prefetch_keys: [[u32; PLE_N_HEADS]; MAX_PREFETCH],
    prefetch_raw: Vec<u8>,
    scale: f32,
    rb: u32,
}

/// `strata::kernels::PleTable`.
pub struct PleTable {
    t: UnsafeCell<TableInner>,
    pin_aux: Option<unsafe extern "C" fn()>,
    prefetch_enabled: Option<unsafe extern "C" fn() -> c_int>,
}

unsafe impl Send for PleTable {}
unsafe impl Sync for PleTable {}

impl PleTable {
    pub fn new(
        pin_aux: Option<unsafe extern "C" fn()>,
        prefetch_enabled: Option<unsafe extern "C" fn() -> c_int>,
    ) -> Self {
        Self {
            t: UnsafeCell::new(TableInner {
                file: None,
                data: std::ptr::null(),
                n_rows: 0,
                fmt: &PLE_FORMATS[0],
                bytes_read: 0,
                mmap_mode: true,
                reader: PleReader::new(),
                ticket: 0,
                pending: false,
                locked: false,
                lock_range: (0, 0),
                rows: [0; PLE_N_HEADS],
                raw: vec![0; PLE_N_HEADS * PLE_ROW_BYTES_MAX],
                n_prefetch: 0,
                prefetch_tickets: [0; MAX_PREFETCH],
                prefetch_keys: [[0; PLE_N_HEADS]; MAX_PREFETCH],
                prefetch_raw: vec![0; MAX_PREFETCH * PLE_N_HEADS * PLE_ROW_BYTES_MAX],
                scale: 1.0,
                rb: PLE_ROW_BYTES,
            }),
            pin_aux,
            prefetch_enabled,
        }
    }

    #[allow(clippy::mut_from_ref)]
    fn m(&self) -> &mut TableInner {
        unsafe { &mut *self.t.get() }
    }

    fn ple_prefetch(&self) -> bool {
        self.prefetch_enabled
            .map(|f| unsafe { f() } != 0)
            .unwrap_or(true)
    }

    pub fn open(&self, path: &str, io: &PleIoOptions) -> Result<(), String> {
        self.close();
        let m = self.m();
        let file = talus_gguf::GgufFile::open(path).map_err(|e| e.to_string())?;
        let Some(t) = file.find_tensor("per_layer_token_embd.weight").cloned() else {
            return Err(format!("per_layer_token_embd.weight is not in {path}"));
        };
        if t.shape().len() != 2 || t.shape()[0] != PLE_HEAD_DIM as u64 {
            return Err("per_layer_token_embd.weight has an unexpected shape".into());
        }
        let tn = gguf_type_name(t.qtype);
        let Some(fmt) = PLE_FORMATS.iter().find(|f| f.gguf_type == tn) else {
            return Err(format!(
                "per_layer_token_embd.weight is {tn}, not {}",
                format_list()
            ));
        };
        let mut scale = 1.0f32;
        if fmt.needs_scale {
            let f = file.get_str("strata.ple.format");
            let s = file.get_f64("strata.ple.scale");
            match (f, s) {
                (Some("f8_e4m3"), Some(s)) if s > 0.0 => scale = s as f32,
                _ => {
                    return Err("per_layer_token_embd.weight is I8 without strata.ple.format = f8_e4m3 and a positive \
                                strata.ple.scale"
                        .into())
                }
            }
        }
        let rb = fmt.row_bytes;
        let n_rows = t.shape()[1];
        if n_rows > u64::MAX / rb as u64 || file.data_start() > file.file_size() {
            return Err(format!(
                "PLE table size overflow or invalid data offset in {path}"
            ));
        }
        let need = n_rows * rb as u64;
        let have = file.file_size() - file.data_start();
        let alone = file.tensors().len() == 1;
        if t.offset > have || need > have - t.offset || (alone && (t.offset != 0 || need != have)) {
            let mut err = format!(
                "PLE table size mismatch: {n_rows} rows x {rb} B = {need} at offset {}, but the file holds {have} from \
                 data_start {}",
                t.offset,
                file.data_start()
            );
            if alone && have > need {
                err.push_str("; the file is longer than its table - delete it and its .done mark and run setup again");
            }
            return Err(err);
        }
        m.fmt = fmt;
        m.rb = rb;
        m.scale = scale;
        m.n_rows = n_rows;
        if !io.mmap {
            let table_offset = file.data_start() + t.offset;
            drop(file);
            m.reader.open(
                path,
                table_offset,
                n_rows,
                io.max_inflight,
                io.cache_rows,
                io.io_thread,
                rb,
                self.pin_aux,
            )?;
            m.reader
                .set_keepalive(io.keepalive_ms, io.keepalive_window_s);
            m.mmap_mode = false;
            return Ok(());
        }
        let data = unsafe {
            file.as_bytes()
                .as_ptr()
                .add((file.data_start() + t.offset) as usize)
        };
        m.data = data;
        m.file = Some(file);
        if io.lock {
            let a0 = data as usize & !(PAGE as usize - 1);
            let a1 = data as usize + need as usize;
            unsafe {
                let _ = madvise(a0 as *mut c_void, a1 - a0, MADV_WILLNEED);
            }
            const PIECE: usize = 64 << 20;
            let pieces = (a1 - a0).div_ceil(PIECE);
            let threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
                .clamp(1, 16);
            let next = std::sync::atomic::AtomicUsize::new(0);
            std::thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= pieces {
                            break;
                        }
                        let mut p = a0 + i * PIECE;
                        let e = a1.min(p + PIECE);
                        while p < e {
                            unsafe { std::ptr::read_volatile(p as *const u8) };
                            p += PAGE as usize;
                        }
                    });
                }
            });
            if unsafe { mlock(a0 as *const c_void, a1 - a0) } == 0 {
                m.locked = true;
                m.lock_range = (a0, a1 - a0);
            } else {
                eprintln!(
                    "strata: PLE table mlock failed ({}; raise `ulimit -l`): its pages stay faulted in but may be reclaimed",
                    strerror_str(errno())
                );
            }
        }
        m.mmap_mode = true;
        Ok(())
    }

    pub fn wait_prefetches(&self) {
        let m = self.m();
        if m.n_prefetch == 0 {
            return;
        }
        for i in 0..m.n_prefetch {
            let _ = m.reader.collect(m.prefetch_tickets[i]);
        }
        m.n_prefetch = 0;
    }

    pub fn prefetch_rows(&self, rows16: &[u32; PLE_N_HEADS]) {
        let m = self.m();
        if !m.mmap_mode && m.reader.is_open() {
            if m.prefetch_keys[..m.n_prefetch].iter().any(|k| k == rows16) {
                return;
            }
            if m.n_prefetch >= MAX_PREFETCH {
                self.wait_prefetches();
            }
            let m = self.m();
            let slot = m.n_prefetch;
            m.n_prefetch += 1;
            m.prefetch_keys[slot] = *rows16;
            let raw = unsafe {
                m.prefetch_raw
                    .as_mut_ptr()
                    .add(slot * PLE_N_HEADS * PLE_ROW_BYTES_MAX)
            };
            m.prefetch_tickets[slot] = unsafe { m.reader.issue(&m.prefetch_keys[slot], raw) };
        }
    }

    pub fn close(&self) {
        self.wait_prefetches();
        let m = self.m();
        m.reader.close();
        m.pending = false;
        if m.locked {
            unsafe {
                let _ = munlock(m.lock_range.0 as *const c_void, m.lock_range.1);
            }
        }
        m.locked = false;
        m.mmap_mode = true;
        m.file = None;
        m.data = std::ptr::null();
        m.n_rows = 0;
        m.rb = PLE_ROW_BYTES;
        m.fmt = &PLE_FORMATS[0];
        m.scale = 1.0;
        m.bytes_read = 0;
    }

    pub fn is_open(&self) -> bool {
        let m = self.m();
        !m.data.is_null() || m.reader.is_open()
    }
    pub fn locked(&self) -> bool {
        self.m().locked
    }
    pub fn format_index(&self) -> u32 {
        self.m().fmt.index
    }
    pub fn mmap_mode(&self) -> bool {
        self.m().mmap_mode
    }
    pub fn rows(&self) -> u64 {
        self.m().n_rows
    }
    pub fn bytes_read(&self) -> u64 {
        self.m().bytes_read
    }

    fn decode(&self, row: &[u8], out160: &mut [f32]) {
        let m = self.m();
        (m.fmt.dequant)(row, m.scale, out160);
    }

    pub fn read_row(&self, row: u32, out160: &mut [f32]) {
        let m = self.m();
        let rb = m.rb as usize;
        if !m.mmap_mode && m.reader.is_open() {
            let mut raw = [0u8; PLE_ROW_BYTES_MAX];
            let t = unsafe { m.reader.issue(&[row], raw.as_mut_ptr()) };
            if m.reader.collect(t).is_err() {
                out160[..PLE_HEAD_DIM].fill(0.0);
                return;
            }
            self.decode(&raw[..rb], out160);
            self.m().bytes_read += rb as u64;
            return;
        }
        if m.data.is_null() || row as u64 >= m.n_rows {
            out160[..PLE_HEAD_DIM].fill(0.0);
            return;
        }
        let s = unsafe { std::slice::from_raw_parts(m.data.add(row as usize * rb), rb) };
        self.decode(s, out160);
        self.m().bytes_read += rb as u64;
    }

    pub fn issue(&self, rows16: &[u32; PLE_N_HEADS]) -> bool {
        self.wait_prefetches();
        let m = self.m();
        m.rows = *rows16;
        if !m.mmap_mode {
            if m.pending {
                return false;
            }
            m.ticket = unsafe { m.reader.issue(&m.rows, m.raw.as_mut_ptr()) };
            m.pending = true;
            return true;
        }
        m.pending = true;
        true
    }

    pub fn collect(&self, out2560: &mut [f32]) -> Result<(), String> {
        let m = self.m();
        if !m.pending {
            return Err("PleTable::collect without issue".into());
        }
        m.pending = false;
        let rb = m.rb as usize;
        if !m.mmap_mode {
            m.reader.collect(m.ticket)?;
            for h in 0..PLE_N_HEADS {
                let row = &self.m().raw[h * rb..(h + 1) * rb];
                self.decode(row, &mut out2560[h * PLE_HEAD_DIM..(h + 1) * PLE_HEAD_DIM]);
            }
            self.m().bytes_read += (PLE_N_HEADS * rb) as u64;
            return Ok(());
        }
        let rows = m.rows;
        for h in 0..PLE_N_HEADS {
            self.read_row(
                rows[h],
                &mut out2560[h * PLE_HEAD_DIM..(h + 1) * PLE_HEAD_DIM],
            );
        }
        Ok(())
    }

    pub fn gather_batch(
        &self,
        rows: &[u32],
        n_tokens: usize,
        out: &mut [f32],
    ) -> Result<(), String> {
        let m = self.m();
        if m.pending {
            return Err("PleTable::gather_batch while a token is in flight".into());
        }
        let n = n_tokens * PLE_N_HEADS;
        let rb = m.rb as usize;
        if !m.mmap_mode {
            if m.n_prefetch >= n_tokens && n_tokens <= MAX_PREFETCH {
                let exact = (0..n_tokens).all(|t| {
                    m.prefetch_keys[t][..] == rows[t * PLE_N_HEADS..(t + 1) * PLE_N_HEADS]
                });
                if exact {
                    for i in 0..m.n_prefetch {
                        if let Err(e) = m.reader.collect(m.prefetch_tickets[i]) {
                            m.n_prefetch = 0;
                            return Err(e);
                        }
                    }
                    m.n_prefetch = 0;
                    for t in 0..n_tokens {
                        for h in 0..PLE_N_HEADS {
                            let at = t * PLE_N_HEADS * PLE_ROW_BYTES_MAX + h * rb;
                            let row = &self.m().prefetch_raw[at..at + rb];
                            let o = (t * PLE_N_HEADS + h) * PLE_HEAD_DIM;
                            self.decode(row, &mut out[o..o + PLE_HEAD_DIM]);
                        }
                    }
                    self.m().bytes_read += (n * rb) as u64;
                    return Ok(());
                }
            }
            self.wait_prefetches();
            let m = self.m();
            let mut raw = vec![0u8; n * rb];
            let ticket = unsafe { m.reader.issue(&rows[..n], raw.as_mut_ptr()) };
            m.reader.collect(ticket)?;
            for i in 0..n {
                self.decode(
                    &raw[i * rb..(i + 1) * rb],
                    &mut out[i * PLE_HEAD_DIM..(i + 1) * PLE_HEAD_DIM],
                );
            }
            self.m().bytes_read += (n * rb) as u64;
            return Ok(());
        }
        self.wait_prefetches();
        const WIN: usize = 4096;
        let data = m.data;
        let n_rows = m.n_rows;
        let hint_on = self.ple_prefetch();
        let hint = |a: usize, b: usize| {
            if data.is_null() || !hint_on {
                return;
            }
            for &r in &rows[a..b] {
                if r as u64 >= n_rows {
                    continue;
                }
                let r0 = data as usize + r as usize * rb;
                let p0 = r0 & !4095usize;
                unsafe {
                    let _ = madvise(p0 as *mut c_void, r0 + rb - p0, MADV_WILLNEED);
                }
            }
        };
        hint(0, n.min(WIN));
        let mut w = 0;
        while w < n {
            if w + WIN < n {
                hint(w + WIN, n.min(w + 2 * WIN));
            }
            let e = n.min(w + WIN);
            for i in w..e {
                self.read_row(rows[i], &mut out[i * PLE_HEAD_DIM..(i + 1) * PLE_HEAD_DIM]);
            }
            w += WIN;
        }
        Ok(())
    }

    pub fn set_injected_delay_us(&self, us: f64) {
        self.m().reader.set_injected_delay_us(us);
    }

    pub fn io_report(&self) -> String {
        let m = self.m();
        if m.mmap_mode || !m.reader.is_open() {
            return String::new();
        }
        let s = m.reader.snapshot();
        let mut out = format!(
            "ple io: {} rows, {:.1}% row-cache hits, {} SSD reads ({:.1} MB), read p50 {:.0} us p99 {:.0} us, blocked \
             {:.3} ms total (submit {:.3} ms), cache {}/{} rows",
            s.requests,
            if s.requests > 0 { 100.0 * s.cache_hits as f64 / s.requests as f64 } else { 0.0 },
            s.reads,
            s.bytes as f64 / 1e6,
            s.percentile(0.5),
            s.percentile(0.99),
            s.wait_us / 1000.0,
            s.submit_us / 1000.0,
            m.reader.cache_size(),
            m.reader.cache_capacity()
        );
        if s.keepalive_reads > 0 {
            out.push_str(&format!(
                ", SSD kept awake by {} reads (slowest {:.1} ms)",
                s.keepalive_reads,
                s.keepalive_us_max / 1000.0
            ));
        }
        out
    }

    pub fn gather(&self, rows16: &[u32; PLE_N_HEADS], out2560: &mut [f32]) {
        if !self.m().mmap_mode {
            let ok = self.issue(rows16);
            let r = if ok {
                self.collect(out2560)
            } else {
                Err(String::new())
            };
            if let Err(e) = r {
                eprintln!(
                    "PleTable::gather: {}",
                    if e.is_empty() {
                        "a token is already in flight"
                    } else {
                        &e
                    }
                );
                out2560[..NG_N_EMBD].fill(0.0);
            }
            return;
        }
        for h in 0..PLE_N_HEADS {
            self.read_row(
                rows16[h],
                &mut out2560[h * PLE_HEAD_DIM..(h + 1) * PLE_HEAD_DIM],
            );
        }
    }
}

impl Drop for PleTable {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_exact() {
        assert_eq!(f32_from_f16(0x3c00), 1.0);
        assert_eq!(f32_from_f16(0x0001), 2f32.powi(-24));
        assert_eq!(f32_from_f16(0xc000), -2.0);
        assert_eq!(f32_from_f16(0x7c00), f32::INFINITY);
    }

    #[test]
    fn fp8_and_formats() {
        let t = fp8_table();
        assert_eq!(t[0x38], 1.0); // e=7, m=0
        assert_eq!(t[0x01], 2f32.powi(-9));
        assert_eq!(t[0x7f], 0.0);
        assert_eq!(t[0xb8], -1.0);
        assert_eq!(
            format_list(),
            "IQ4_NL, Q5_0, FP8 (I8), Q5_1, Q8_0, Q4_0 or BF16"
        );
        // an IQ4_NL row: d = 1.0, codes 0..15 in the low nibbles, 15..0 in the high ones
        let mut row = vec![0u8; 90];
        for b in 0..5 {
            row[b * 18] = 0x00;
            row[b * 18 + 1] = 0x3c;
            for j in 0..16 {
                row[b * 18 + 2 + j] = (j as u8) | ((15 - j as u8) << 4);
            }
        }
        let mut out = [0f32; 160];
        iq4nl_row(&row, 1.0, &mut out);
        assert_eq!(out[0], -127.0);
        assert_eq!(out[16], 113.0);
        assert_eq!(out[15 + 32], 113.0);
    }

    #[test]
    fn row_cache() {
        let mut c = RowCache::default();
        c.init(64, 90);
        let r = [7u8; 90];
        c.insert(12345, r.as_ptr());
        assert!(c.find(12345).is_some());
        assert_eq!(c.used, 1);
        assert!(c.find(1).is_none());
    }
}
