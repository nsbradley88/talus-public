//! The byte transport under a [`crate::Link`] (talus#48 / WAVES 1.18). The link owns the framing (header, payload
//! parts, optional CRC tail, counters); a transport only moves bytes in order, reliably, between two stages.
//!
//! - [`TcpTransport`] is today's data plane and the default: `TCP_NODELAY`, 4 MiB buffers, `TCP_QUICKACK` re-armed
//!   after every received header and payload, and optional socket busy-polling (`TALUS_CLUSTER_BUSY_POLL_US`, off by
//!   default).
//! - A later USB4STREAM backend (Linux 7.2+ `/dev/tbstreamN`, `rd/usb4x/DESIGN.md` option B) implements the same trait
//!   over one or more file descriptors and is chosen in HELLO ([`DataPlane`]); the pipeline code does not change.
//!
//! HELLO/INIT and the heartbeat stay on TCP whatever the data plane is ([`crate::Beat`] takes a [`Closer`] for the
//! data link, so it can close a non-socket transport too).

use std::io::{self, IoSlice, Read, Write};
use std::net::{Shutdown, TcpStream};
use std::os::fd::AsRawFd;
use std::time::Duration;

/// A reliable, ordered byte stream between two stages.
pub trait Transport: Send {
    /// Short name for logs and HELLO, e.g. `tcp`.
    fn kind(&self) -> &'static str;
    /// The peer, for logs.
    fn peer(&self) -> String;
    /// Write every byte of `bufs` (in order) or fail.
    fn write_all_vectored(&mut self, bufs: &mut [IoSlice<'_>]) -> io::Result<()>;
    /// Fill `buf` completely or fail (EOF is `UnexpectedEof`).
    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()>;
    /// Called once a whole header, and once a whole payload (with its CRC tail), has been read.
    fn after_recv(&mut self) {}
    /// Bound blocking reads (`None` = block forever).
    fn set_read_timeout(&self, t: Option<Duration>) -> io::Result<()>;
    /// A handle that closes this transport from another thread (heartbeat, worker teardown): any blocked or later
    /// send/receive on it fails at once.
    fn closer(&self) -> io::Result<Closer>;
}

/// Closes a transport from another thread; see [`Transport::closer`].
pub struct Closer(Box<dyn Fn() + Send + Sync>);

impl Closer {
    pub fn new(f: impl Fn() + Send + Sync + 'static) -> Self {
        Self(Box::new(f))
    }
    /// A closer for a TCP socket: `shutdown(Both)` on a duplicate of the fd.
    pub fn tcp(s: TcpStream) -> Self {
        Self::new(move || {
            let _ = s.shutdown(Shutdown::Both);
        })
    }
    pub fn close(&self) {
        (self.0)()
    }
}

/// The data plane a head asks for in HELLO (`dp=...`; absent = `tcp`). Only TCP exists today.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataPlane {
    Tcp,
}

impl DataPlane {
    pub fn as_str(self) -> &'static str {
        match self {
            DataPlane::Tcp => "tcp",
        }
    }

    /// Parse a HELLO `dp` value; `None` (an older head) means TCP.
    pub fn parse(s: Option<&str>) -> Result<Self, String> {
        match s.unwrap_or("tcp") {
            "tcp" => Ok(DataPlane::Tcp),
            other => Err(format!(
                "data plane {other:?} is not supported by this build (supported: tcp)"
            )),
        }
    }

    /// The data plane this process asks for: `TALUS_CLUSTER_DP` (default `tcp`).
    pub fn from_env() -> Result<Self, String> {
        let v = std::env::var("TALUS_CLUSTER_DP")
            .ok()
            .filter(|v| !v.is_empty());
        Self::parse(v.as_deref()).map_err(|e| format!("TALUS_CLUSTER_DP: {e}"))
    }
}

/// Optional socket busy-polling on the data link (`rd/usb4x/DESIGN.md` option A). Off by default.
///
/// - `TALUS_CLUSTER_BUSY_POLL_US=N` sets `SO_BUSY_POLL` to N µs: a blocking receive spins on the device queue for up
///   to N µs instead of sleeping until the (throttled) interrupt. Raising it above the socket's current value
///   needs `CAP_NET_ADMIN`; without it the ceiling is `net.core.busy_read` as it was when the socket (for an
///   accepted socket: the listener) was created. A refused option is reported once and the link runs without it.
/// - `TALUS_CLUSTER_PREFER_BUSY_POLL=1` also sets `SO_PREFER_BUSY_POLL` (needs `CAP_NET_ADMIN`; pair it with the
///   device's `napi_defer_hard_irqs` / `gro_flush_timeout`).
///
/// It changes timing only, never the bytes on the link. A spinning core takes package power from the GPU on
/// Strix Halo, so judge it on decode tok/s, not RTT alone.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BusyPoll {
    pub us: u32,
    pub prefer: bool,
}

impl BusyPoll {
    pub fn from_env() -> Self {
        let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        Self {
            us: env("TALUS_CLUSTER_BUSY_POLL_US")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            prefer: env("TALUS_CLUSTER_PREFER_BUSY_POLL").is_some_and(|v| v == "1"),
        }
    }
    pub fn enabled(&self) -> bool {
        self.us > 0
    }
}

// Linux asm-generic/socket.h (stable ABI); libc does not export the last one on every version.
const SO_BUSY_POLL: i32 = 46;
const SO_PREFER_BUSY_POLL: i32 = 69;

fn setopt(fd: i32, level: i32, name: i32, val: i32) -> io::Result<()> {
    let r = unsafe {
        libc::setsockopt(
            fd,
            level,
            name,
            (&val as *const i32).cast::<libc::c_void>(),
            std::mem::size_of::<i32>() as libc::socklen_t,
        )
    };
    if r == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

fn getopt(fd: i32, level: i32, name: i32) -> io::Result<i32> {
    let mut v: i32 = 0;
    let mut n = std::mem::size_of::<i32>() as libc::socklen_t;
    let r = unsafe {
        libc::getsockopt(
            fd,
            level,
            name,
            (&mut v as *mut i32).cast::<libc::c_void>(),
            &mut n,
        )
    };
    if r == 0 {
        Ok(v)
    } else {
        Err(io::Error::last_os_error())
    }
}

/// Today's data plane: one TCP connection.
pub struct TcpTransport {
    s: TcpStream,
    peer: String,
}

impl TcpTransport {
    /// Configure `s` as a stage data link (options from the environment: [`BusyPoll::from_env`]).
    pub fn new(s: TcpStream) -> io::Result<Self> {
        Self::with_busy_poll(s, BusyPoll::from_env())
    }

    pub fn with_busy_poll(s: TcpStream, bp: BusyPoll) -> io::Result<Self> {
        s.set_nodelay(true)?;
        let fd = s.as_raw_fd();
        let _ = setopt(fd, libc::SOL_SOCKET, libc::SO_SNDBUF, 4 << 20);
        let _ = setopt(fd, libc::SOL_SOCKET, libc::SO_RCVBUF, 4 << 20);
        let _ = setopt(fd, libc::IPPROTO_TCP, libc::TCP_QUICKACK, 1);
        if bp.enabled() {
            apply_busy_poll(fd, bp);
        }
        let peer = s.peer_addr().map(|a| a.to_string()).unwrap_or_default();
        Ok(Self { s, peer })
    }

    /// The `SO_BUSY_POLL` value in effect on the socket (µs; the kernel default is `net.core.busy_read`).
    pub fn busy_poll_us(&self) -> io::Result<i32> {
        getopt(self.s.as_raw_fd(), libc::SOL_SOCKET, SO_BUSY_POLL)
    }
}

fn apply_busy_poll(fd: i32, bp: BusyPoll) {
    static WARNED: std::sync::Once = std::sync::Once::new();
    let us = bp.us.min(i32::MAX as u32) as i32;
    let mut err = setopt(fd, libc::SOL_SOCKET, SO_BUSY_POLL, us).err();
    if err.is_none() && bp.prefer {
        err = setopt(fd, libc::SOL_SOCKET, SO_PREFER_BUSY_POLL, 1).err();
    }
    match err {
        None => WARNED.call_once(|| {
            eprintln!(
                "talus-cluster: data link busy-poll {us} us{}",
                if bp.prefer { " (prefer)" } else { "" }
            )
        }),
        Some(e) => WARNED.call_once(|| {
            eprintln!(
                "talus-cluster: WARN: busy-poll {us} us not applied ({e}); raising it (or prefer) needs \
                 CAP_NET_ADMIN or net.core.busy_read >= it before the socket was made. Running without it"
            )
        }),
    }
}

impl Transport for TcpTransport {
    fn kind(&self) -> &'static str {
        "tcp"
    }

    fn peer(&self) -> String {
        self.peer.clone()
    }

    fn write_all_vectored(&mut self, bufs: &mut [IoSlice<'_>]) -> io::Result<()> {
        let mut bufs: &mut [IoSlice<'_>] = bufs;
        while !bufs.is_empty() {
            let n = self.s.write_vectored(bufs)?;
            if n == 0 {
                return Err(io::Error::new(io::ErrorKind::WriteZero, "link closed"));
            }
            IoSlice::advance_slices(&mut bufs, n);
        }
        Ok(())
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        self.s.read_exact(buf)
    }

    fn after_recv(&mut self) {
        let _ = setopt(self.s.as_raw_fd(), libc::IPPROTO_TCP, libc::TCP_QUICKACK, 1);
    }

    fn set_read_timeout(&self, t: Option<Duration>) -> io::Result<()> {
        self.s.set_read_timeout(t)
    }

    fn closer(&self) -> io::Result<Closer> {
        Ok(Closer::tcp(self.s.try_clone()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn data_plane_parse() {
        assert_eq!(DataPlane::parse(None), Ok(DataPlane::Tcp));
        assert_eq!(DataPlane::parse(Some("tcp")), Ok(DataPlane::Tcp));
        assert!(DataPlane::parse(Some("tbstream:0,1")).is_err());
        assert_eq!(DataPlane::Tcp.as_str(), "tcp");
    }

    #[test]
    fn busy_poll_default_off_and_applied_within_limit() {
        assert!(!BusyPoll::default().enabled());
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let c = TcpStream::connect(l.local_addr().unwrap()).unwrap();
        let (s, _) = l.accept().unwrap();
        let off = TcpTransport::with_busy_poll(s, BusyPoll::default()).unwrap();
        let base = off.busy_poll_us().unwrap();
        // lowering (or keeping) the value never needs CAP_NET_ADMIN, so this exercises the setsockopt path
        let t = TcpTransport::with_busy_poll(
            c,
            BusyPoll {
                us: base.max(0) as u32,
                prefer: false,
            },
        )
        .unwrap();
        assert_eq!(t.busy_poll_us().unwrap(), base);
    }

    #[test]
    fn closer_fails_a_blocked_read() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let c = TcpStream::connect(l.local_addr().unwrap()).unwrap();
        let (_s, _) = l.accept().unwrap();
        let mut t = TcpTransport::new(c).unwrap();
        let k = t.closer().unwrap();
        let th = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            k.close();
        });
        let mut b = [0u8; 8];
        assert!(t.read_exact(&mut b).is_err());
        th.join().unwrap();
    }
}
