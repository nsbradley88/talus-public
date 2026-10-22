//! Session files (`SAVE` / `RESTORE`), format v1: a Rust port of `src/core/conversation_file.cpp`.
//!
//! Byte-for-byte the C++ format (little-endian, fixed-width integers): a file written here restores in the C++
//! engine and the reverse. Header v1: `magic[8] version:u32 header_size:u32 model:u64 config:u64 payload:u64
//! reserved:u64[2] header_hash:u64`, then the payload, then `payload_hash:u64 "STRSEND\x01"`. Linux only
//! (the engine's only Rust target): O_DIRECT when the filesystem takes it, buffered I/O otherwise.

use std::collections::HashMap;
use std::ffi::CStr;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt};
use std::os::unix::io::AsRawFd;

use super::{Checkpoint, ConvBuffer, ConvKv, ImageKey, SavedConversation};

const MAGIC: [u8; 8] = *b"STRSESS\x01";
const END: [u8; 8] = *b"STRSEND\x01";
const VERSION: u32 = 1;
pub const HEADER: u64 = 64;
pub const TRAILER: u64 = 16;
const ALIGN: usize = 4096;
pub const BLOCK: usize = 16 << 20;

// ---------------------------------------------------------------------------------------------------------------
// errors
// ---------------------------------------------------------------------------------------------------------------

/// Why a session file operation failed (`SessionError`): the server maps it to an HTTP status.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SessionError {
    #[default]
    None,
    Invalid,
    Storage,
    Memory,
    Io,
}

impl SessionError {
    pub fn name(self) -> &'static str {
        match self {
            SessionError::None => "none",
            SessionError::Invalid => "invalid",
            SessionError::Storage => "storage",
            SessionError::Memory => "memory",
            SessionError::Io => "io",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SessionStatus {
    pub error: SessionError,
    pub published: bool,
    pub dir_flush_unsupported: bool,
}

const ENOENT: i32 = 2;
const ENOMEM: i32 = 12;
const EEXIST: i32 = 17;
const EINVAL: i32 = 22;
const EFBIG: i32 = 27;
const ENOSPC: i32 = 28;
const ELOOP: i32 = 40;
const EDQUOT: i32 = 122;

fn errno_kind(e: i32) -> SessionError {
    match e {
        ENOSPC | EDQUOT | EFBIG => SessionError::Storage,
        ENOMEM => SessionError::Memory,
        _ => SessionError::Io,
    }
}

extern "C" {
    fn strerror(errnum: i32) -> *const std::os::raw::c_char;
    fn fcntl(fd: i32, cmd: i32, ...) -> i32;
    fn statvfs(path: *const std::os::raw::c_char, buf: *mut u64) -> i32;
}

/// `std::strerror(e)`: the exact C library text the C++ engine prints.
pub fn strerror_text(e: i32) -> String {
    unsafe {
        let p = strerror(e);
        if p.is_null() {
            return format!("error {e}");
        }
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

fn os_err(e: &std::io::Error) -> (SessionError, String) {
    match e.raw_os_error() {
        Some(n) => (errno_kind(n), strerror_text(n)),
        None => (SessionError::Io, e.to_string()),
    }
}

// ---------------------------------------------------------------------------------------------------------------
// hashing (xxHash64-style rounds, NOT standard XXH64)
// ---------------------------------------------------------------------------------------------------------------

const P1: u64 = 0x9E37_79B1_85EB_CA87;
const P2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const P3: u64 = 0x1656_67B1_9E37_79F9;
const P4: u64 = 0x85EB_CA77_C2B2_AE63;
const P5: u64 = 0x27D4_EB2F_1656_67C5;

#[inline]
fn round1(acc: u64, w: u64) -> u64 {
    acc.wrapping_add(w.wrapping_mul(P2))
        .rotate_left(31)
        .wrapping_mul(P1)
}

#[inline]
fn load64(p: &[u8]) -> u64 {
    u64::from_le_bytes([p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]])
}

/// `SessionHasher`: streaming and one-shot agree.
#[derive(Clone)]
pub struct SessionHasher {
    lane: [u64; 4],
    pending: [u8; 32],
    npending: usize,
    total: u64,
}

impl SessionHasher {
    pub fn new(seed: u64) -> Self {
        Self {
            lane: [
                seed.wrapping_add(P1).wrapping_add(P2),
                seed.wrapping_add(P2),
                seed,
                seed.wrapping_sub(P1),
            ],
            pending: [0; 32],
            npending: 0,
            total: 0,
        }
    }

    fn block(lane: &mut [u64; 4], p: &[u8]) {
        for (i, l) in lane.iter_mut().enumerate() {
            *l = round1(*l, load64(&p[8 * i..]));
        }
    }

    pub fn update(&mut self, mut p: &[u8]) {
        self.total = self.total.wrapping_add(p.len() as u64);
        if self.npending > 0 {
            let take = p.len().min(32 - self.npending);
            self.pending[self.npending..self.npending + take].copy_from_slice(&p[..take]);
            self.npending += take;
            p = &p[take..];
            if self.npending < 32 {
                return;
            }
            let pend = self.pending;
            Self::block(&mut self.lane, &pend);
            self.npending = 0;
        }
        while p.len() >= 32 {
            Self::block(&mut self.lane, &p[..32]);
            p = &p[32..];
        }
        if !p.is_empty() {
            self.pending[..p.len()].copy_from_slice(p);
        }
        self.npending = p.len();
    }

    pub fn digest(&self) -> u64 {
        let l = &self.lane;
        let mut h = l[0]
            .rotate_left(1)
            .wrapping_add(l[1].rotate_left(7))
            .wrapping_add(l[2].rotate_left(12))
            .wrapping_add(l[3].rotate_left(18));
        for &x in l {
            h ^= round1(0, x);
            h = h.wrapping_mul(P1).wrapping_add(P4);
        }
        h = h.wrapping_add(self.total);
        let mut i = 0usize;
        while i + 8 <= self.npending {
            h ^= round1(0, load64(&self.pending[i..]));
            h = h.rotate_left(27).wrapping_mul(P1).wrapping_add(P4);
            i += 8;
        }
        while i < self.npending {
            h ^= (self.pending[i] as u64).wrapping_mul(P5);
            h = h.rotate_left(11).wrapping_mul(P1);
            i += 1;
        }
        h ^= h >> 33;
        h = h.wrapping_mul(P2);
        h ^= h >> 29;
        h = h.wrapping_mul(P3);
        h ^= h >> 32;
        h
    }
}

pub fn session_hash64(data: &[u8], seed: u64) -> u64 {
    let mut h = SessionHasher::new(seed);
    h.update(data);
    h.digest()
}

/// Typed, length-delimited identity fields (`SessionIdentityBuilder`).
pub struct IdentityBuilder {
    hash: SessionHasher,
}

impl IdentityBuilder {
    pub fn new(seed: u64) -> Self {
        Self {
            hash: SessionHasher::new(seed),
        }
    }
    fn field(&mut self, tag: &str, ty: u8, data: &[u8]) {
        self.hash.update(&(tag.len() as u64).to_le_bytes());
        self.hash.update(tag.as_bytes());
        self.hash.update(&[ty]);
        self.hash.update(&(data.len() as u64).to_le_bytes());
        if !data.is_empty() {
            self.hash.update(data);
        }
    }
    pub fn str(&mut self, tag: &str, v: &str) {
        self.field(tag, b's', v.as_bytes());
    }
    pub fn i64(&mut self, tag: &str, v: i64) {
        self.field(tag, b'i', &v.to_le_bytes());
    }
    pub fn u64(&mut self, tag: &str, v: u64) {
        self.field(tag, b'u', &v.to_le_bytes());
    }
    pub fn f64(&mut self, tag: &str, v: f64) {
        self.field(tag, b'f', &v.to_bits().to_le_bytes());
    }
    pub fn bytes(&mut self, tag: &str, v: &[u8]) {
        self.field(tag, b'b', v);
    }
    pub fn digest(&self) -> u64 {
        self.hash.digest()
    }
}

/// The resolved rope configuration (`SessionRope`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SessionRope {
    pub ty: i64,
    pub freq_base: f64,
    pub factor: f64,
    pub freq_scale: f64,
    pub orig_ctx: f64,
    pub ext_factor: f64,
    pub attn_factor: f64,
    pub beta_fast: f64,
    pub beta_slow: f64,
}

/// Every engine setting that changes what the saved state means (`SessionConfig`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SessionConfig {
    pub engine_version: String,
    pub backend: String,
    pub kv: String,
    pub max_context: i64,
    pub kv_resident: i64,
    pub mtp_window: i64,
    pub kv_rot: bool,
    pub rope: SessionRope,
    pub cvec: u64,
    pub switches: Vec<(String, i64)>,
}

pub fn session_config_fingerprint(c: &SessionConfig) -> u64 {
    let mut b = IdentityBuilder::new(0x434f_4e46_4947_3031); // "CONFIG01"
    b.str("engine", &c.engine_version);
    b.str("backend", &c.backend);
    b.str("kv", &c.kv);
    b.i64("max_context", c.max_context);
    b.i64("kv_resident", c.kv_resident);
    b.i64("mtp_window", c.mtp_window);
    b.i64("kv_rot", c.kv_rot as i64);
    b.i64("rope.type", c.rope.ty);
    b.f64("rope.freq_base", c.rope.freq_base);
    b.f64("rope.factor", c.rope.factor);
    b.f64("rope.freq_scale", c.rope.freq_scale);
    b.f64("rope.orig_ctx", c.rope.orig_ctx);
    b.f64("rope.ext_factor", c.rope.ext_factor);
    b.f64("rope.attn_factor", c.rope.attn_factor);
    b.f64("rope.beta_fast", c.rope.beta_fast);
    b.f64("rope.beta_slow", c.rope.beta_slow);
    b.u64("cvec", c.cvec);
    b.u64("switches", c.switches.len() as u64);
    for (name, value) in &c.switches {
        b.i64(&format!("switch.{name}"), *value);
    }
    b.digest()
}

/// One model input and its role (`SessionModelFile`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionModelFile {
    pub role: String,
    pub path: String,
    pub optional: bool,
}

/// Sampled fingerprint of the model inputs: role, size, first and last MiB of each distinct file.
pub fn session_model_fingerprint(
    files: &[SessionModelFile],
    mut beat: impl FnMut(),
) -> Result<u64, String> {
    let mut h = IdentityBuilder::new(0x5354_5241_5441_4d44);
    let mut buf = vec![0u8; 1 << 20];
    let mut seen: HashMap<String, (bool, u64)> = HashMap::new();
    h.u64("files", files.len() as u64);
    for file in files {
        if !seen.contains_key(&file.path) {
            let mut d = (false, 0u64);
            match File::open(&file.path) {
                Err(e) => {
                    let missing = e.raw_os_error() == Some(ENOENT);
                    if !(file.optional && missing) {
                        return Err(format!(
                            "session fingerprint: {} {}: {}",
                            file.role,
                            file.path,
                            os_err(&e).1
                        ));
                    }
                }
                Ok(f) => {
                    let md = f.metadata().map_err(|e| {
                        format!(
                            "session fingerprint: {} {}: {}",
                            file.role,
                            file.path,
                            os_err(&e).1
                        )
                    })?;
                    if !md.is_file() {
                        return Err(format!(
                            "session fingerprint: {} {}: not a regular file",
                            file.role, file.path
                        ));
                    }
                    let size = md.len();
                    let mut one = IdentityBuilder::new(0x4649_4c45_5341_4d50); // "FILESAMP"
                    one.u64("size", size);
                    let head = size.min(buf.len() as u64) as usize;
                    read_exact_at(&f, 0, &mut buf[..head])
                        .map_err(|e| format!("session fingerprint: reading {}: {e}", file.path))?;
                    one.bytes("head", &buf[..head]);
                    if size > buf.len() as u64 {
                        let tail = (size - head as u64).min(buf.len() as u64) as usize;
                        read_exact_at(&f, size - tail as u64, &mut buf[..tail]).map_err(|e| {
                            format!("session fingerprint: reading {}: {e}", file.path)
                        })?;
                        one.bytes("tail", &buf[..tail]);
                    }
                    d = (true, one.digest());
                }
            }
            seen.insert(file.path.clone(), d);
            beat();
        }
        let d = seen[&file.path];
        if !d.0 {
            if !file.optional {
                return Err(format!(
                    "session fingerprint: {} {} is missing",
                    file.role, file.path
                ));
            }
            h.str("absent", &file.role);
            continue;
        }
        h.str("role", &file.role);
        h.u64("sample", d.1);
    }
    Ok(h.digest())
}

fn read_exact_at(f: &File, mut off: u64, mut buf: &mut [u8]) -> Result<(), String> {
    while !buf.is_empty() {
        match f.read_at(buf, off) {
            Ok(0) => return Err("file ends early".to_string()),
            Ok(n) => {
                off += n as u64;
                buf = &mut buf[n..];
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(os_err(&e).1),
        }
    }
    Ok(())
}

/// The bound identity of a file (`SessionFileIdentity`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SessionFileIdentity {
    pub model: u64,
    pub config: u64,
}

// ---------------------------------------------------------------------------------------------------------------
// limits & preflights
// ---------------------------------------------------------------------------------------------------------------

#[inline]
pub fn sat_add(a: u64, b: u64) -> u64 {
    a.saturating_add(b)
}
#[inline]
pub fn sat_mul(a: u64, b: u64) -> u64 {
    a.saturating_mul(b)
}

/// What a read checks before it allocates (`SessionReadLimits`).
pub struct SessionReadLimits<'a> {
    pub max_file_bytes: u64,
    pub max_tokens: u64,
    pub max_checkpoints: u64,
    pub max_kv_layers: u64,
    pub geometry: Option<[i64; 18]>,
    pub layer_range: Option<(i64, i64)>,
    pub max_state_bytes: [u64; 5],
    pub max_kv_bytes: Vec<[u64; 5]>,
    pub admit: Option<&'a mut dyn FnMut(u64) -> Result<(), String>>,
    pub progress: Option<&'a mut dyn FnMut(u64, u64)>,
}

impl Default for SessionReadLimits<'_> {
    fn default() -> Self {
        Self {
            max_file_bytes: u64::MAX,
            max_tokens: u64::MAX,
            max_checkpoints: u64::MAX,
            max_kv_layers: u64::MAX,
            geometry: None,
            layer_range: None,
            max_state_bytes: [u64::MAX; 5],
            max_kv_bytes: Vec::new(),
            admit: None,
            progress: None,
        }
    }
}

pub fn session_read_max_file_bytes(l: &SessionReadLimits) -> u64 {
    let open = u64::MAX;
    if l.max_tokens == open || l.max_checkpoints == open || l.max_kv_layers == open {
        return l.max_file_bytes;
    }
    if l.max_state_bytes.contains(&open) {
        return l.max_file_bytes;
    }
    if (l.max_kv_bytes.len() as u64) < l.max_kv_layers {
        return l.max_file_bytes;
    }
    let mut cp = sat_add(16, sat_mul(l.max_tokens, 4 + 16));
    for &b in &l.max_state_bytes {
        cp = sat_add(cp, sat_add(8, b));
    }
    cp = sat_add(cp, 8);
    let mut n = HEADER + TRAILER + 18 * 8 + 3 * 8;
    n = sat_add(n, cp);
    n = sat_add(n, sat_add(8, sat_mul(l.max_checkpoints, cp)));
    n = sat_add(n, 8);
    for i in 0..l.max_kv_layers as usize {
        n = sat_add(n, 7 * 8);
        for &b in &l.max_kv_bytes[i] {
            n = sat_add(n, sat_add(8, b));
        }
    }
    n.min(l.max_file_bytes)
}

pub fn session_read_peak_bytes(file_bytes: u64) -> u64 {
    let segments = file_bytes / BLOCK as u64 + 1;
    sat_add(
        sat_add(sat_add(file_bytes, BLOCK as u64), sat_mul(segments, 4096)),
        1 << 20,
    )
}

/// What a SAVE copies into RAM besides what the engine holds (`SessionSaveLive`).
#[derive(Clone, Copy, Debug, Default)]
pub struct SessionSaveLive {
    pub state_bytes: u64,
    pub tokens: u64,
    pub images: u64,
    pub kv_layers: u64,
}

/// `sizeof(ConversationImageKey)` and `sizeof(SessionKvSource) + 256` of the C++ build (accounting only).
pub const SIZEOF_IMAGE_KEY: u64 = 16;
pub const SIZEOF_KV_SOURCE: u64 = 160;

pub fn session_save_peak_bytes(deepest: Option<&Checkpoint>, live: &SessionSaveLive) -> u64 {
    let mut n = sat_add(BLOCK as u64, 1 << 20);
    if let Some(d) = deepest {
        n = sat_add(n, sat_mul(2, d.bytes() as u64));
    }
    n = sat_add(n, live.state_bytes);
    n = sat_add(n, sat_mul(live.tokens, 4));
    n = sat_add(n, sat_mul(live.images, SIZEOF_IMAGE_KEY));
    n = sat_add(n, sat_mul(live.kv_layers, SIZEOF_KV_SOURCE + 256));
    n
}

/// The deepest checkpoint; with a pinned one in the chain, the deepest pinned one.
pub fn session_deepest_checkpoint(chain: &[Checkpoint]) -> Option<&Checkpoint> {
    let any_pinned = chain.iter().any(|c| c.pinned);
    let mut deepest: Option<&Checkpoint> = None;
    for c in chain {
        if (!any_pinned || c.pinned) && deepest.is_none_or(|d| c.ids.len() > d.ids.len()) {
            deepest = Some(c);
        }
    }
    deepest
}

pub fn session_phase_limit_s(bytes: u64) -> i64 {
    3600i64.min(60 + (bytes >> 22).min(3600) as i64)
}

/// The folder the free-space preflight asks about (POSIX rules).
pub fn session_free_space_dir(path: &str) -> String {
    match path.rfind('/') {
        None => ".".to_string(),
        Some(0) => "/".to_string(),
        Some(cut) => path[..cut].to_string(),
    }
}

fn free_space(path: &str) -> Option<u64> {
    let dir = std::ffi::CString::new(session_free_space_dir(path)).ok()?;
    let mut buf = [0u64; 16]; // struct statvfs (x86_64 glibc: 11 x u64 + 6 x i32 spare)
    if unsafe { statvfs(dir.as_ptr(), buf.as_mut_ptr()) } != 0 {
        return None;
    }
    Some(buf[4].wrapping_mul(buf[1])) // f_bavail * f_frsize
}

// ---------------------------------------------------------------------------------------------------------------
// raw file with O_DIRECT, aligned 16 MiB blocks
// ---------------------------------------------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
const O_DIRECT: i32 = 0o40000;
#[cfg(not(target_arch = "x86_64"))]
const O_DIRECT: i32 = 0;
const O_NOFOLLOW: i32 = 0o400000;
const O_NONBLOCK: i32 = 0o4000;
const F_GETFL: i32 = 3;
const F_SETFL: i32 = 4;

fn buffered_forced() -> bool {
    std::env::var("STRATA_SESSION_BUFFERED")
        .map(|v| v.starts_with('1'))
        .unwrap_or(false)
}

struct AlignedBlock {
    ptr: *mut u8,
}

impl AlignedBlock {
    fn new() -> Option<Self> {
        let layout = std::alloc::Layout::from_size_align(BLOCK, ALIGN).ok()?;
        let ptr = unsafe { std::alloc::alloc(layout) };
        (!ptr.is_null()).then_some(Self { ptr })
    }
    fn slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, BLOCK) }
    }
}

impl Drop for AlignedBlock {
    fn drop(&mut self) {
        unsafe {
            std::alloc::dealloc(
                self.ptr,
                std::alloc::Layout::from_size_align_unchecked(BLOCK, ALIGN),
            )
        }
    }
}

struct RawFile {
    f: Option<File>,
    direct: bool,
    kind: SessionError,
    error: String,
}

impl RawFile {
    fn new() -> Self {
        Self {
            f: None,
            direct: false,
            kind: SessionError::Io,
            error: String::new(),
        }
    }
    fn set_err(&mut self, e: &std::io::Error) {
        let (k, m) = os_err(e);
        self.kind = k;
        self.error = m;
    }
    fn try_direct(&mut self) {
        if O_DIRECT == 0 || buffered_forced() {
            return;
        }
        let fd = self.f.as_ref().unwrap().as_raw_fd();
        unsafe {
            let fl = fcntl(fd, F_GETFL);
            self.direct = fl >= 0 && fcntl(fd, F_SETFL, fl | O_DIRECT) == 0;
        }
    }
    fn leave_direct(&mut self) -> bool {
        let fd = self.f.as_ref().unwrap().as_raw_fd();
        unsafe {
            let fl = fcntl(fd, F_GETFL);
            if fl < 0 || fcntl(fd, F_SETFL, fl & !O_DIRECT) != 0 {
                return false;
            }
        }
        self.direct = false;
        true
    }
    fn create_new(&mut self, path: &str) -> Result<(), i32> {
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(O_NOFOLLOW)
            .open(path)
        {
            Ok(f) => {
                self.f = Some(f);
                self.try_direct();
                Ok(())
            }
            Err(e) => {
                self.set_err(&e);
                Err(e.raw_os_error().unwrap_or(0))
            }
        }
    }
    /// A session to restore: never through a symbolic link, a regular file with one name.
    fn open_session(&mut self, path: &str) -> Result<u64, bool> {
        let f = match OpenOptions::new()
            .read(true)
            .custom_flags(O_NOFOLLOW | O_NONBLOCK)
            .open(path)
        {
            Ok(f) => f,
            Err(e) => {
                let eloop = e.raw_os_error() == Some(ELOOP);
                self.set_err(&e);
                if eloop {
                    self.error = "a symbolic link, not a file".to_string();
                }
                return Err(eloop);
            }
        };
        let md = match f.metadata() {
            Ok(m) => m,
            Err(e) => {
                self.set_err(&e);
                return Err(false);
            }
        };
        if !md.file_type().is_file() {
            self.error = "not a regular file".to_string();
            return Err(true);
        }
        if md.nlink() != 1 {
            self.error = "the file has more than one name (hard link)".to_string();
            return Err(true);
        }
        unsafe {
            let fd = f.as_raw_fd();
            let fl = fcntl(fd, F_GETFL);
            if fl >= 0 {
                fcntl(fd, F_SETFL, fl & !O_NONBLOCK);
            }
        }
        self.f = Some(f);
        self.try_direct();
        Ok(md.len())
    }
    fn write_all(&mut self, mut p: &[u8]) -> bool {
        while !p.is_empty() {
            let r = self.f.as_mut().unwrap().write(p);
            match r {
                Ok(0) => {
                    self.kind = SessionError::Io;
                    self.error = "short write".to_string();
                    return false;
                }
                Ok(n) => p = &p[n..],
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e)
                    if e.raw_os_error() == Some(EINVAL) && self.direct && self.leave_direct() => {}
                Err(e) => {
                    self.set_err(&e);
                    return false;
                }
            }
        }
        true
    }
    fn read_some(&mut self, p: &mut [u8]) -> i64 {
        loop {
            match self.f.as_mut().unwrap().read(p) {
                Ok(n) => return n as i64,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e)
                    if e.raw_os_error() == Some(EINVAL) && self.direct && self.leave_direct() => {}
                Err(e) => {
                    self.set_err(&e);
                    return -1;
                }
            }
        }
    }
    fn truncate(&mut self, n: u64) -> bool {
        match self.f.as_ref().unwrap().set_len(n) {
            Ok(()) => true,
            Err(e) => {
                self.set_err(&e);
                false
            }
        }
    }
    fn sync(&mut self) -> bool {
        match self.f.as_ref().unwrap().sync_all() {
            Ok(()) => true,
            Err(e) => {
                self.set_err(&e);
                false
            }
        }
    }
    fn close(&mut self) -> bool {
        self.f.take();
        true
    }
}

fn random64() -> u64 {
    let mut b = [0u8; 8];
    if File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut b))
        .is_err()
    {
        b = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0))
        .to_le_bytes();
    }
    u64::from_le_bytes(b) ^ (std::process::id() as u64).wrapping_mul(P2)
}

fn temp_name(leaf: &str) -> String {
    let mut keep = leaf.len().min(64);
    while keep > 0 && keep < leaf.len() && !leaf.is_char_boundary(keep) {
        keep -= 1;
    }
    format!(".{}.{:016x}.tmp", &leaf[..keep], random64())
}

fn split_path(path: &str) -> Result<(String, String), String> {
    let (dir, leaf) = match path.rfind('/') {
        None => (String::new(), path.to_string()),
        Some(0) => ("/".to_string(), path[1..].to_string()),
        Some(c) => (path[..c].to_string(), path[c + 1..].to_string()),
    };
    if leaf.is_empty() || leaf == "." || leaf == ".." {
        return Err(format!("session file: {path} names no file"));
    }
    Ok((dir, leaf))
}

struct Publisher {
    dir: String,
    leaf: String,
    temp: String,
    own: bool,
}

impl Publisher {
    fn join(&self, name: &str) -> String {
        if self.dir.is_empty() {
            name.to_string()
        } else if self.dir.ends_with('/') {
            format!("{}{name}", self.dir)
        } else {
            format!("{}/{name}", self.dir)
        }
    }
    fn open(path: &str) -> Result<Self, (SessionError, String)> {
        let (dir, leaf) = split_path(path).map_err(|e| (SessionError::Invalid, e))?;
        let d = if dir.is_empty() { "." } else { dir.as_str() };
        match std::fs::metadata(d) {
            Ok(m) if m.is_dir() => {}
            Ok(_) => {
                return Err((
                    SessionError::Io,
                    format!(
                        "session file: cannot open the directory of {path}: {}",
                        strerror_text(20)
                    ),
                ))
            }
            Err(e) => {
                let (k, m) = os_err(&e);
                return Err((
                    k,
                    format!("session file: cannot open the directory of {path}: {m}"),
                ));
            }
        }
        Ok(Self {
            dir,
            leaf,
            temp: String::new(),
            own: false,
        })
    }
    fn create_temp(&mut self, f: &mut RawFile) -> Result<(), String> {
        for _ in 0..8 {
            self.temp = temp_name(&self.leaf);
            match f.create_new(&self.join(&self.temp)) {
                Ok(()) => {
                    self.own = true;
                    return Ok(());
                }
                Err(e) if e == EEXIST => continue,
                Err(_) => break,
            }
        }
        Err(f.error.clone())
    }
    fn publish(&mut self, durable: bool, st: &mut SessionStatus) -> Result<(), String> {
        if let Err(e) = std::fs::rename(self.join(&self.temp), self.join(&self.leaf)) {
            let (k, m) = os_err(&e);
            st.error = k;
            return Err(m);
        }
        self.own = false;
        st.published = true;
        if !durable {
            return Ok(());
        }
        let d = if self.dir.is_empty() {
            "."
        } else {
            self.dir.as_str()
        };
        let r = File::open(d).and_then(|f| f.sync_all());
        if let Err(e) = r {
            if e.raw_os_error() == Some(EINVAL) {
                st.dir_flush_unsupported = true;
                return Ok(());
            }
            let (k, m) = os_err(&e);
            st.error = k;
            return Err(format!("flushing the directory: {m}"));
        }
        Ok(())
    }
}

impl Drop for Publisher {
    fn drop(&mut self) {
        if self.own {
            let _ = std::fs::remove_file(self.join(&self.temp));
        }
    }
}

struct FileSink<'f, 'p> {
    buf: AlignedBlock,
    f: &'f mut RawFile,
    used: usize,
    total: u64,
    expected: u64,
    progress: Option<&'p mut dyn FnMut(u64, u64)>,
}

impl FileSink<'_, '_> {
    fn span(&mut self) -> &mut [u8] {
        let u = self.used;
        &mut self.buf.slice()[u..]
    }
    fn commit(&mut self, n: usize) -> bool {
        self.used += n;
        self.total += n as u64;
        self.used < BLOCK || self.drain()
    }
    fn write(&mut self, mut p: &[u8]) -> bool {
        while !p.is_empty() {
            let room = BLOCK - self.used;
            let c = room.min(p.len());
            let u = self.used;
            self.buf.slice()[u..u + c].copy_from_slice(&p[..c]);
            if !self.commit(c) {
                return false;
            }
            p = &p[c..];
        }
        true
    }
    fn drain(&mut self) -> bool {
        let used = self.used;
        let data = unsafe { std::slice::from_raw_parts(self.buf.ptr, used) };
        if !self.f.write_all(data) {
            return false;
        }
        self.used = 0;
        if let Some(p) = self.progress.as_mut() {
            p(self.total, self.expected);
        }
        true
    }
    fn finish(&mut self, durable: bool, phase: &mut Option<&mut dyn FnMut(&str, u64)>) -> bool {
        if self.f.f.is_none() {
            return false;
        }
        let padded = if self.f.direct {
            self.used.div_ceil(ALIGN) * ALIGN
        } else {
            self.used
        };
        let used = self.used;
        self.buf.slice()[used..padded].fill(0);
        let data = unsafe { std::slice::from_raw_parts(self.buf.ptr, padded) };
        let mut ok = self.f.write_all(data);
        if ok && used > 0 {
            if let Some(p) = self.progress.as_mut() {
                p(self.total, self.expected);
            }
        }
        ok = ok && (padded == used || self.f.truncate(self.total));
        if ok && durable {
            if let Some(ph) = phase.as_mut() {
                ph("flush", self.total);
            }
        }
        ok = ok && (!durable || self.f.sync());
        ok = self.f.close() && ok;
        ok
    }
}

struct FileSource<'f, 'p> {
    buf: AlignedBlock,
    f: &'f mut RawFile,
    at: usize,
    have: usize,
    total: u64,
    expected: u64,
    progress: Option<&'p mut dyn FnMut(u64, u64)>,
}

impl FileSource<'_, '_> {
    fn fill(&mut self) -> bool {
        self.at = 0;
        self.have = 0;
        while self.have < BLOCK {
            let h = self.have;
            let dst = unsafe { std::slice::from_raw_parts_mut(self.buf.ptr.add(h), BLOCK - h) };
            let r = self.f.read_some(dst);
            if r < 0 {
                return false;
            }
            if r == 0 {
                break;
            }
            self.have += r as usize;
            if self.f.direct && self.have % ALIGN != 0 {
                break;
            }
        }
        let before = self.total;
        self.total += self.have as u64;
        if self.total > before {
            if let Some(p) = self.progress.as_mut() {
                p(self.total, self.expected);
            }
        }
        self.have > 0
    }
    fn read(&mut self, mut d: &mut [u8]) -> bool {
        while !d.is_empty() {
            if self.at == self.have && !self.fill() {
                if self.f.error.is_empty() {
                    self.f.error = "file ends early".to_string();
                }
                return false;
            }
            let c = d.len().min(self.have - self.at);
            let src = unsafe { std::slice::from_raw_parts(self.buf.ptr.add(self.at), c) };
            d[..c].copy_from_slice(src);
            self.at += c;
            d = &mut d[c..];
        }
        true
    }
}

// ---------------------------------------------------------------------------------------------------------------
// serialization
// ---------------------------------------------------------------------------------------------------------------

/// One K/V layer read straight from its owner into the file block (`SessionKvSource`).
pub struct KvSource<'a> {
    pub format: i32,
    pub cells: i64,
    pub heads: i64,
    pub head_dim: i64,
    pub page_size: i64,
    pub pooled_rows: i64,
    pub idx_dim: i64,
    pub sizes: [usize; 5],
    /// `read(part, offset, dst)`: copies `dst.len()` bytes of part 0..4 from `offset`; Err = why it failed.
    pub read: Box<dyn FnMut(usize, usize, &mut [u8]) -> Result<(), String> + 'a>,
}

struct Out<'s, 'f, 'p> {
    f: Option<&'s mut FileSink<'f, 'p>>,
    hash: SessionHasher,
    n: u64,
    ok: bool,
    overflow: bool,
    read_failed: bool,
    read_error: String,
}

impl Out<'_, '_, '_> {
    fn count(&mut self, c: u64) {
        match self.n.checked_add(c) {
            Some(v) => self.n = v,
            None => {
                self.overflow = true;
                self.ok = false;
            }
        }
    }
    fn raw(&mut self, p: &[u8]) {
        self.count(p.len() as u64);
        let Some(f) = self.f.as_mut() else {
            return;
        };
        if !self.ok || p.is_empty() {
            return;
        }
        self.hash.update(p);
        self.ok = f.write(p);
    }
    fn u64(&mut self, v: u64) {
        self.raw(&v.to_le_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.raw(&v.to_le_bytes());
    }
    fn bytes_vec(&mut self, v: &[u8]) {
        self.u64(v.len() as u64);
        self.raw(v);
    }
    fn ids(&mut self, v: &[i32]) {
        self.u64(v.len() as u64);
        if self.f.is_none() {
            self.count(v.len() as u64 * 4);
            return;
        }
        // i32 little-endian in memory
        let b = unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 4) };
        self.raw(b);
    }
    fn buffer(&mut self, b: &ConvBuffer) {
        self.u64(b.size() as u64);
        if self.f.is_none() {
            self.count(b.size() as u64);
            return;
        }
        b.visit(0, b.size(), |p, _| {
            self.raw(p);
            self.ok
        });
    }
    fn source(&mut self, s: &mut KvSource, part: usize) {
        let size = s.sizes[part];
        self.u64(size as u64);
        if self.f.is_none() {
            self.count(size as u64);
            return;
        }
        let mut at = 0usize;
        while at < size && self.ok {
            let f = self.f.as_mut().unwrap();
            let span = f.span();
            let c = span.len().min(size - at);
            if let Err(e) = (s.read)(part, at, &mut span[..c]) {
                self.ok = false;
                self.read_failed = true;
                self.read_error = e;
                return;
            }
            self.hash.update(&span[..c]);
            self.count(c as u64);
            let f = self.f.as_mut().unwrap();
            self.ok = self.ok && f.commit(c);
            at += c;
        }
    }
}

fn put_checkpoint(o: &mut Out, c: &Checkpoint) {
    o.ids(&c.ids);
    o.u64(c.imgs.len() as u64);
    for k in &c.imgs {
        o.i64(k.start);
        o.u64(k.hash);
    }
    o.bytes_vec(&c.gdn);
    o.bytes_vec(&c.ple);
    o.bytes_vec(&c.tails);
    o.bytes_vec(&c.dead);
    o.bytes_vec(&c.block_pos);
    o.u64(c.used);
}

fn put_kv_header(o: &mut Out, h: [i64; 7]) {
    for v in h {
        o.i64(v);
    }
}

fn put_payload(o: &mut Out, s: &SavedConversation, sources: Option<&mut [KvSource]>) {
    for &g in &s.geometry {
        o.i64(g);
    }
    o.i64(s.layer_lo);
    o.i64(s.layer_hi);
    o.u64(s.cvec as u64);
    put_checkpoint(o, &s.live);
    o.u64(s.checkpoints.len() as u64);
    for c in &s.checkpoints {
        put_checkpoint(o, c);
    }
    if let Some(src) = sources {
        o.u64(src.len() as u64);
        for k in src.iter_mut() {
            put_kv_header(
                o,
                [
                    k.format as i64,
                    k.cells,
                    k.heads,
                    k.head_dim,
                    k.page_size,
                    k.pooled_rows,
                    k.idx_dim,
                ],
            );
            for part in 0..5 {
                o.source(k, part);
            }
        }
        return;
    }
    o.u64(s.kv.len() as u64);
    for k in &s.kv {
        put_kv_header(
            o,
            [
                k.format as i64,
                k.cells,
                k.heads,
                k.head_dim,
                k.page_size,
                k.pooled_rows,
                k.idx_dim,
            ],
        );
        for b in [&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled] {
            o.buffer(b);
        }
    }
}

fn header_bytes(id: &SessionFileIdentity, payload: u64) -> [u8; HEADER as usize] {
    let mut h = [0u8; HEADER as usize];
    h[..8].copy_from_slice(&MAGIC);
    h[8..12].copy_from_slice(&VERSION.to_le_bytes());
    h[12..16].copy_from_slice(&(HEADER as u32).to_le_bytes());
    h[16..24].copy_from_slice(&id.model.to_le_bytes());
    h[24..32].copy_from_slice(&id.config.to_le_bytes());
    h[32..40].copy_from_slice(&payload.to_le_bytes());
    let hh = session_hash64(&h[..56], 0);
    h[56..64].copy_from_slice(&hh.to_le_bytes());
    h
}

/// Knobs of a write (`SessionWriteOptions`).
pub struct SessionWriteOptions<'a> {
    pub min_free_bytes: u64,
    pub durable: bool,
    pub progress: Option<&'a mut dyn FnMut(u64, u64)>,
    pub phase: Option<&'a mut dyn FnMut(&str, u64)>,
}

impl Default for SessionWriteOptions<'_> {
    fn default() -> Self {
        Self {
            min_free_bytes: 0,
            durable: true,
            progress: None,
            phase: None,
        }
    }
}

/// Writes `image` (with `sources` as its K/V when given; `image.kv` must then be empty) to `path` through a
/// temporary file, flush and rename. Returns the file size.
pub fn session_file_write(
    path: &str,
    image: &SavedConversation,
    mut sources: Option<&mut [KvSource]>,
    id: &SessionFileIdentity,
    opt: SessionWriteOptions,
    st: &mut SessionStatus,
) -> Result<u64, String> {
    *st = SessionStatus {
        error: SessionError::Invalid,
        ..Default::default()
    };
    if sources.is_some() && !image.kv.is_empty() {
        return Err("session file: K/V given both in the image and as sources".to_string());
    }
    let mut sizing = Out {
        f: None,
        hash: SessionHasher::new(0),
        n: 0,
        ok: true,
        overflow: false,
        read_failed: false,
        read_error: String::new(),
    };
    put_payload(&mut sizing, image, sources.as_deref_mut());
    if sizing.overflow || sizing.n > u64::MAX - HEADER - TRAILER {
        return Err("session file: the state is too large for this build".to_string());
    }
    let payload = sizing.n;
    let total = HEADER + payload + TRAILER;
    st.error = SessionError::Io;
    if opt.min_free_bytes > 0 {
        let Some(avail) = free_space(path) else {
            return Err(format!(
                "session file: cannot read the free disk space for {path}"
            ));
        };
        if avail < total || avail - total < opt.min_free_bytes {
            st.error = SessionError::Storage;
            return Err(format!(
                "session file: not enough disk space ({} MiB plus a reserve of {} MiB needed, {} MiB free)",
                total >> 20,
                opt.min_free_bytes >> 20,
                avail >> 20
            ));
        }
    }
    let mut pubr = match Publisher::open(path) {
        Ok(p) => p,
        Err((k, e)) => {
            st.error = k;
            return Err(e);
        }
    };
    let mut raw = RawFile::new();
    if let Err(e) = pubr.create_temp(&mut raw) {
        st.error = raw.kind;
        return Err(format!(
            "session file: cannot create a temporary file beside {path}: {e}"
        ));
    }
    let Some(buf) = AlignedBlock::new() else {
        st.error = SessionError::Memory;
        return Err("session file: out of memory for the write buffer".to_string());
    };
    let SessionWriteOptions {
        durable,
        progress,
        mut phase,
        ..
    } = opt;
    let mut sink = FileSink {
        buf,
        f: &mut raw,
        used: 0,
        total: 0,
        expected: total,
        progress,
    };
    if let Some(p) = sink.progress.as_mut() {
        p(0, total);
    }
    let h = header_bytes(id, payload);
    let mut ok = sink.write(&h);
    let (out_ok, out_n, out_hash, read_failed, read_error) = {
        let mut out = Out {
            f: Some(&mut sink),
            hash: SessionHasher::new(0),
            n: 0,
            ok: true,
            overflow: false,
            read_failed: false,
            read_error: String::new(),
        };
        if ok {
            put_payload(&mut out, image, sources.take());
        }
        (
            out.ok,
            out.n,
            out.hash.digest(),
            out.read_failed,
            out.read_error,
        )
    };
    ok = ok && out_ok && out_n == payload;
    if ok {
        ok = sink.write(&out_hash.to_le_bytes()) && sink.write(&END);
    }
    ok = sink.finish(durable, &mut phase) && ok;
    let why;
    if !ok {
        if read_failed {
            why = if read_error.is_empty() {
                "K/V source read".to_string()
            } else {
                format!("K/V source read: {read_error}")
            };
            st.error = SessionError::Io;
        } else {
            why = raw.error.clone();
            st.error = raw.kind;
        }
    } else {
        if let Some(ph) = phase.as_mut() {
            ph("publish", total);
        }
        match pubr.publish(durable, st) {
            Ok(()) => {
                st.error = SessionError::None;
                return Ok(total);
            }
            Err(e) => why = e,
        }
    }
    let mut error = format!(
        "session file: writing {path} failed{}",
        if why.is_empty() {
            String::new()
        } else {
            format!(": {why}")
        }
    );
    if st.published {
        error.push_str(" (the new file has already replaced the old one; its folder entry may not survive a power loss)");
    }
    Err(error)
}

// ---------------------------------------------------------------------------------------------------------------
// parsing
// ---------------------------------------------------------------------------------------------------------------

struct In<'s, 'f, 'p, 'l> {
    f: &'s mut FileSource<'f, 'p>,
    hash: SessionHasher,
    left: u64,
    error: String,
    limits: &'s SessionReadLimits<'l>,
    kind: SessionError,
    kv_layer: usize,
}

impl In<'_, '_, '_, '_> {
    fn fail(&mut self, m: &str) -> bool {
        if self.error.is_empty() {
            self.error = format!("session file: {m}");
        }
        false
    }
    fn raw(&mut self, p: &mut [u8]) -> bool {
        if p.len() as u64 > self.left {
            return self.fail("payload ends early");
        }
        if !p.is_empty() && !self.f.read(p) {
            self.kind = self.f.f.kind;
            let e = format!("read error: {}", self.f.f.error);
            return self.fail(&e);
        }
        self.hash.update(p);
        self.left -= p.len() as u64;
        true
    }
    fn u64(&mut self, v: &mut u64) -> bool {
        let mut b = [0u8; 8];
        let ok = self.raw(&mut b);
        *v = u64::from_le_bytes(b);
        ok
    }
    fn i64(&mut self, v: &mut i64) -> bool {
        let mut b = [0u8; 8];
        let ok = self.raw(&mut b);
        *v = i64::from_le_bytes(b);
        ok
    }
    fn count(&mut self, n: &mut u64, elem: u64, limit: u64) -> bool {
        if !self.u64(n) {
            return false;
        }
        if elem > 0 && *n > self.left / elem {
            return self.fail("element count exceeds the payload");
        }
        if *n > limit {
            let m = format!("element count {} exceeds this engine's limit {}", *n, limit);
            return self.fail(&m);
        }
        true
    }
    fn bytes_vec(&mut self, v: &mut Vec<u8>, limit: u64) -> bool {
        let mut n = 0u64;
        if !self.count(&mut n, 1, limit) {
            return false;
        }
        *v = vec![0u8; n as usize];
        self.raw(v)
    }
    fn ids(&mut self, v: &mut Vec<i32>, limit: u64) -> bool {
        let mut n = 0u64;
        if !self.count(&mut n, 4, limit) {
            return false;
        }
        let mut b = vec![0u8; n as usize * 4];
        if !self.raw(&mut b) {
            return false;
        }
        *v = b
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        true
    }
    fn buffer(&mut self, b: &mut ConvBuffer, part: usize) -> bool {
        let mut n = 0u64;
        let limit = self
            .limits
            .max_kv_bytes
            .get(self.kv_layer)
            .map_or(u64::MAX, |l| l[part]);
        if !self.count(&mut n, 1, limit) {
            return false;
        }
        *b = ConvBuffer::default();
        b.resize(n as usize);
        b.visit_mut(0, n as usize, |p, _| self.raw(p))
    }
}

fn get_checkpoint(inp: &mut In, c: &mut Checkpoint) -> bool {
    let lim = inp.limits.max_tokens;
    if !inp.ids(&mut c.ids, lim) {
        return false;
    }
    let mut n = 0u64;
    if !inp.count(&mut n, 16, lim) {
        return false;
    }
    c.imgs = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let mut k = ImageKey::default();
        if !inp.i64(&mut k.start) || !inp.u64(&mut k.hash) {
            return false;
        }
        c.imgs.push(k);
    }
    let m = inp.limits.max_state_bytes;
    inp.bytes_vec(&mut c.gdn, m[0])
        && inp.bytes_vec(&mut c.ple, m[1])
        && inp.bytes_vec(&mut c.tails, m[2])
        && inp.bytes_vec(&mut c.dead, m[3])
        && inp.bytes_vec(&mut c.block_pos, m[4])
        && inp.u64(&mut c.used)
}

fn get_payload(inp: &mut In, s: &mut SavedConversation) -> bool {
    for g in s.geometry.iter_mut() {
        if !inp.i64(g) {
            return false;
        }
    }
    if let Some(g) = inp.limits.geometry {
        if g != s.geometry {
            return inp.fail("saved with another model geometry than this runtime's");
        }
    }
    let mut cvec = 0u64;
    if !inp.i64(&mut s.layer_lo) || !inp.i64(&mut s.layer_hi) || !inp.u64(&mut cvec) {
        return false;
    }
    if let Some((lo, hi)) = inp.limits.layer_range {
        if lo != s.layer_lo || hi != s.layer_hi {
            return inp.fail("saved with another layer range than this runtime's");
        }
    }
    if cvec > 1 {
        return inp.fail("invalid cvec flag");
    }
    s.cvec = cvec == 1;
    if !get_checkpoint(inp, &mut s.live) {
        return false;
    }
    let mut n = 0u64;
    let mc = inp.limits.max_checkpoints;
    if !inp.count(&mut n, 72, mc) {
        return false;
    }
    s.checkpoints = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let mut c = Checkpoint::default();
        if !get_checkpoint(inp, &mut c) {
            return false;
        }
        s.checkpoints.push(c);
    }
    let ml = inp.limits.max_kv_layers;
    if !inp.count(&mut n, 96, ml) {
        return false;
    }
    s.kv = Vec::with_capacity(n as usize);
    for i in 0..n as usize {
        inp.kv_layer = i;
        let mut k = ConvKv::default();
        let mut format = 0i64;
        if !inp.i64(&mut format)
            || !inp.i64(&mut k.cells)
            || !inp.i64(&mut k.heads)
            || !inp.i64(&mut k.head_dim)
            || !inp.i64(&mut k.page_size)
            || !inp.i64(&mut k.pooled_rows)
            || !inp.i64(&mut k.idx_dim)
        {
            return false;
        }
        if format < i32::MIN as i64 || format > i32::MAX as i64 {
            return inp.fail("invalid K/V format");
        }
        k.format = format as i32;
        if !inp.buffer(&mut k.k, 0)
            || !inp.buffer(&mut k.v, 1)
            || !inp.buffer(&mut k.k_scale, 2)
            || !inp.buffer(&mut k.v_scale, 3)
            || !inp.buffer(&mut k.pooled, 4)
        {
            return false;
        }
        s.kv.push(k);
    }
    true
}

/// Full validation (a regular file with one link, size, header, identity, bounded parse, payload hash).
/// Returns the image and the file size.
pub fn session_file_read(
    path: &str,
    id: &SessionFileIdentity,
    limits: &mut SessionReadLimits,
    st: &mut SessionStatus,
) -> Result<(SavedConversation, u64), String> {
    *st = SessionStatus {
        error: SessionError::Invalid,
        ..Default::default()
    };
    let mut raw = RawFile::new();
    let size = match raw.open_session(path) {
        Ok(s) => s,
        Err(refused) => {
            st.error = if refused {
                SessionError::Invalid
            } else {
                raw.kind
            };
            return Err(format!("session file: {path}: {}", raw.error));
        }
    };
    if size < HEADER + TRAILER {
        return Err(format!("session file: size {size} is below the minimum"));
    }
    let max_size = session_read_max_file_bytes(limits);
    if size > max_size {
        return Err(format!(
            "session file: size {size} exceeds this engine's limit{}",
            if max_size == u64::MAX {
                String::new()
            } else {
                format!(" of {max_size}")
            }
        ));
    }
    let Some(buf) = AlignedBlock::new() else {
        st.error = SessionError::Memory;
        return Err("session file: out of memory for the read buffer".to_string());
    };
    let progress = limits.progress.take();
    let mut src = FileSource {
        buf,
        f: &mut raw,
        at: 0,
        have: 0,
        total: 0,
        expected: size,
        progress,
    };
    let mut h = [0u8; HEADER as usize];
    if !src.read(&mut h) {
        st.error = src.f.kind;
        return Err(format!("session file: header read error: {}", src.f.error));
    }
    let rd = |o: usize| load64(&h[o..]);
    let version = u32::from_le_bytes([h[8], h[9], h[10], h[11]]);
    let hsize = u32::from_le_bytes([h[12], h[13], h[14], h[15]]);
    let (model, config, payload, r0, r1, hh) = (rd(16), rd(24), rd(32), rd(40), rd(48), rd(56));
    if h[..8] != MAGIC {
        return Err("session file: not a Strata session file (magic)".to_string());
    }
    if hh != session_hash64(&h[..56], 0) {
        return Err("session file: header checksum mismatch".to_string());
    }
    if version != VERSION {
        return Err(format!("session file: unsupported version {version}"));
    }
    if hsize as u64 != HEADER || r0 != 0 || r1 != 0 {
        return Err("session file: invalid header fields".to_string());
    }
    if model != id.model {
        return Err(
            "session file: saved with another model (model fingerprint differs)".to_string(),
        );
    }
    if config != id.config {
        return Err(
            "session file: saved with another engine configuration (config fingerprint differs)"
                .to_string(),
        );
    }
    if payload > size || size - HEADER - TRAILER != payload {
        return Err(format!(
            "session file: size {size} does not match the header ({})",
            HEADER + payload + TRAILER
        ));
    }
    if let Some(admit) = limits.admit.as_mut() {
        if let Err(why) = admit(session_read_peak_bytes(size)) {
            st.error = SessionError::Memory;
            return Err(format!("session file: {why}"));
        }
    }
    let mut parsed = SavedConversation::default();
    let (ok, left, kind, error, phash) = {
        let mut inp = In {
            f: &mut src,
            hash: SessionHasher::new(0),
            left: payload,
            error: String::new(),
            limits,
            kind: SessionError::Invalid,
            kv_layer: 0,
        };
        let ok = get_payload(&mut inp, &mut parsed);
        (ok, inp.left, inp.kind, inp.error, inp.hash.digest())
    };
    if !ok {
        st.error = kind;
        return Err(error);
    }
    if left != 0 {
        return Err(format!("session file: payload has {left} unparsed bytes"));
    }
    let mut t = [0u8; TRAILER as usize];
    if !src.read(&mut t) {
        st.error = src.f.kind;
        return Err(format!("session file: trailer read error: {}", src.f.error));
    }
    if load64(&t) != phash {
        return Err("session file: payload checksum mismatch".to_string());
    }
    if t[8..] != END {
        return Err("session file: bad end marker".to_string());
    }
    st.error = SessionError::None;
    Ok((parsed, size))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hasher_streaming_matches_one_shot() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 7 + 3) as u8).collect();
        let one = session_hash64(&data, 5);
        let mut h = SessionHasher::new(5);
        for c in data.chunks(13) {
            h.update(c);
        }
        assert_eq!(h.digest(), one);
        assert_ne!(session_hash64(&data, 0), one);
        assert_ne!(session_hash64(&[], 0), session_hash64(&[0], 0));
    }

    #[test]
    fn free_space_dir_follows_posix_rules() {
        assert_eq!(session_free_space_dir("chat.bin"), ".");
        assert_eq!(session_free_space_dir("/chat.bin"), "/");
        assert_eq!(session_free_space_dir("/a/b/chat.bin"), "/a/b");
    }

    #[test]
    fn phase_limit_matches_cpp() {
        assert_eq!(session_phase_limit_s(0), 60);
        assert_eq!(
            session_phase_limit_s(1_200_000_000),
            60 + (1_200_000_000u64 >> 22) as i64
        );
        assert_eq!(session_phase_limit_s(u64::MAX), 3600);
    }
}
