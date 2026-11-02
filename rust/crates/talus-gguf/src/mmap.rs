//! Zero-copy POSIX `mmap` wrapper and unified byte-slice backing store.
//!
//! Uses direct `extern "C"` FFI to POSIX `mmap`/`munmap` with `std::os::unix::io::AsRawFd`
//! so that `talus-gguf` requires zero external crate dependencies.

use std::ffi::c_void;
use std::fs::File;
use std::ops::Deref;
use std::path::Path;
use std::ptr::NonNull;

use crate::error::{GgufError, Result};

#[cfg(unix)]
use std::os::unix::io::AsRawFd;

#[cfg(unix)]
const PROT_READ: i32 = 1;
#[cfg(unix)]
const MAP_PRIVATE: i32 = 2;
#[cfg(unix)]
const MAP_FAILED: *mut c_void = !0usize as *mut c_void;

#[cfg(target_os = "linux")]
const PROT_NONE: i32 = 0;
#[cfg(target_os = "linux")]
const MAP_FIXED: i32 = 0x10;
#[cfg(target_os = "linux")]
const MAP_ANONYMOUS: i32 = 0x20;

#[cfg(unix)]
extern "C" {
    fn sysconf(name: i32) -> i64;
}
#[cfg(target_os = "linux")]
const SC_PAGESIZE: i32 = 30;

#[cfg(unix)]
extern "C" {
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
}

/// Read-only memory-mapped file region backed by OS `mmap`.
#[derive(Debug)]
pub struct Mmap {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: `Mmap` is a read-only (`PROT_READ`, `MAP_PRIVATE`) mapping owned for its lifetime.
unsafe impl Send for Mmap {}
// SAFETY: Concurrent read-only access to a `PROT_READ` mapping is thread-safe.
unsafe impl Sync for Mmap {}

impl Mmap {
    /// Opens `path` and maps its entire contents read-only into the process address space.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path_ref = path.as_ref();
        let display_path = Some(path_ref.display().to_string());
        let file = File::open(path_ref).map_err(|e| GgufError::Io {
            path: display_path.clone(),
            message: e.to_string(),
        })?;
        Self::from_file(&file, display_path)
    }

    /// Maps an already-open [`File`] read-only.
    pub fn from_file(file: &File, display_path: Option<String>) -> Result<Self> {
        let meta = file.metadata().map_err(|e| GgufError::Io {
            path: display_path.clone(),
            message: e.to_string(),
        })?;
        let file_len = meta.len();
        let len = usize::try_from(file_len).map_err(|_| GgufError::Io {
            path: display_path.clone(),
            message: format!("file size {file_len} exceeds address space"),
        })?;

        if len == 0 {
            return Ok(Self {
                ptr: NonNull::dangling(),
                len: 0,
            });
        }

        #[cfg(unix)]
        {
            let fd = file.as_raw_fd();
            // SAFETY: `fd` is a valid open file descriptor and `len > 0`.
            // `PROT_READ` + `MAP_PRIVATE` creates a read-only private mapping.
            let raw = unsafe { mmap(std::ptr::null_mut(), len, PROT_READ, MAP_PRIVATE, fd, 0) };
            if raw == MAP_FAILED || raw.is_null() {
                let err = std::io::Error::last_os_error();
                return Err(GgufError::Io {
                    path: display_path,
                    message: format!("mmap({len} bytes) failed: {err}"),
                });
            }
            let ptr = NonNull::new(raw.cast::<u8>()).ok_or_else(|| GgufError::Io {
                path: display_path,
                message: "mmap returned null pointer".to_string(),
            })?;
            Ok(Self { ptr, len })
        }

        #[cfg(not(unix))]
        {
            let _ = (file, display_path);
            Err(GgufError::Io {
                path: None,
                message: "POSIX mmap is only supported on unix targets".to_string(),
            })
        }
    }

    /// maps several files read-only back to back in ONE contiguous address range (each file starts on a
    /// page boundary; the tail of a file's last page reads as zeros). Returns the mapping and each file's start
    /// offset in it. Used to view a `llama-gguf-split` model as one flat GGUF without copying it.
    #[cfg(target_os = "linux")]
    pub fn open_concat<P: AsRef<Path>>(paths: &[P]) -> Result<(Self, Vec<usize>)> {
        let page = {
            // SAFETY: sysconf has no preconditions.
            let p = unsafe { sysconf(SC_PAGESIZE) };
            if p > 0 {
                p as usize
            } else {
                4096
            }
        };
        let mut files = Vec::with_capacity(paths.len());
        let mut bases = Vec::with_capacity(paths.len());
        let mut total = 0usize;
        for p in paths {
            let pr = p.as_ref();
            let display_path = Some(pr.display().to_string());
            let file = File::open(pr).map_err(|e| GgufError::Io {
                path: display_path.clone(),
                message: e.to_string(),
            })?;
            let len = file
                .metadata()
                .map_err(|e| GgufError::Io {
                    path: display_path.clone(),
                    message: e.to_string(),
                })?
                .len() as usize;
            bases.push(total);
            total = total
                .checked_add(len.div_ceil(page) * page)
                .ok_or_else(|| GgufError::Io {
                    path: display_path.clone(),
                    message: "concatenated size overflows".to_string(),
                })?;
            files.push((file, len, display_path));
        }
        if total == 0 {
            return Ok((
                Self {
                    ptr: NonNull::dangling(),
                    len: 0,
                },
                bases,
            ));
        }
        // SAFETY: an anonymous PROT_NONE reservation of `total` bytes; the files are then mapped over it with
        // MAP_FIXED inside the reservation only, so no foreign mapping is replaced.
        let base = unsafe {
            mmap(
                std::ptr::null_mut(),
                total,
                PROT_NONE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if base == MAP_FAILED || base.is_null() {
            let err = std::io::Error::last_os_error();
            return Err(GgufError::Io {
                path: None,
                message: format!("mmap reservation ({total} bytes) failed: {err}"),
            });
        }
        let ptr = NonNull::new(base.cast::<u8>()).ok_or_else(|| GgufError::Io {
            path: None,
            message: "mmap returned null pointer".to_string(),
        })?;
        let whole = Self { ptr, len: total }; // unmaps everything on an early return
        for ((file, len, display_path), &off) in files.iter().zip(bases.iter()) {
            if *len == 0 {
                continue;
            }
            // SAFETY: `base + off .. + len` lies inside the reservation; `file` is open for reading.
            let at = unsafe { base.cast::<u8>().add(off).cast::<c_void>() };
            let raw = unsafe {
                mmap(
                    at,
                    *len,
                    PROT_READ,
                    MAP_PRIVATE | MAP_FIXED,
                    file.as_raw_fd(),
                    0,
                )
            };
            if raw == MAP_FAILED || raw != at {
                let err = std::io::Error::last_os_error();
                return Err(GgufError::Io {
                    path: display_path.clone(),
                    message: format!(
                        "mmap({len} bytes) at offset {off} of the concatenated view failed: {err}"
                    ),
                });
            }
        }
        Ok((whole, bases))
    }

    /// Returns the length of the mapped region in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the mapped region has zero length.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the mapped bytes as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            &[]
        } else {
            // SAFETY: `ptr` points to a valid `PROT_READ` mapping of `len` bytes for the lifetime of `self`.
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }
}

impl Deref for Mmap {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsRef<[u8]> for Mmap {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        #[cfg(unix)]
        if self.len > 0 {
            // SAFETY: `self.ptr` and `self.len` were returned by a successful `mmap` call.
            unsafe {
                let _ = munmap(self.ptr.as_ptr().cast::<c_void>(), self.len);
            }
        }
    }
}

/// Backing storage for a GGUF file or binary pack: either an OS `mmap` or an in-memory buffer.
#[derive(Debug)]
pub enum Backing {
    /// Zero-copy OS memory mapping.
    Mmap(Mmap),
    /// In-memory byte vector.
    Owned(Vec<u8>),
}

impl Backing {
    /// Returns the backing bytes as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Self::Mmap(m) => m.as_slice(),
            Self::Owned(v) => v.as_slice(),
        }
    }

    /// Returns the byte length of the backing buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// Returns `true` if the backing buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }
}

impl Deref for Backing {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsRef<[u8]> for Backing {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}
