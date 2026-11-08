//! The byte transport under a [`crate::Link`] (talus#48 / WAVES 1.18). The link owns the framing (header, payload
//! parts, optional CRC tail, counters); a transport only moves bytes in order, reliably, between two stages.
//!
//! - [`TcpTransport`] is today's data plane and the default: `TCP_NODELAY`, 4 MiB buffers, `TCP_QUICKACK` re-armed
//!   after every received header and payload, and optional socket busy-polling (`TALUS_CLUSTER_BUSY_POLL_US`, off by
//!   default), and a bounded spin-receive pinned to the interface's RX IRQ CPU (`TALUS_CLUSTER_SPIN_US`, default ON at
//!   100 ms in cluster mode, 0 turns it off; [`crate::spin`]).
//! - A later USB4STREAM backend (Linux 7.2+ `/dev/tbstreamN`, `rd/usb4x/DESIGN.md` option B) implements the same trait
//!   over one or more file descriptors and is chosen in HELLO ([`DataPlane`]); the pipeline code does not change.
//!
//! HELLO/INIT and the heartbeat stay on TCP whatever the data plane is ([`crate::Beat`] takes a [`Closer`] for the
//! data link, so it can close a non-socket transport too).

use std::io::{self, IoSlice, Read, Write};
use std::net::{Shutdown, TcpStream};
use std::os::fd::AsRawFd;
use std::time::{Duration, Instant};

use crate::spin::{self, PinSpec, Predictor, SpinCfg};

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
    /// [`Transport::read_exact`] for a message header: the read that waits for the peer's next message.
    fn read_header(&mut self, buf: &mut [u8]) -> io::Result<()> {
        self.read_exact(buf)
    }
    /// The handshake is done and the engine is built: start any latency mode (spin-receive, pinning) on the
    /// calling thread, which is the thread that will receive from now on.
    fn steady(&mut self) {}
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
    spin: SpinCfg,
    /// spinning (set by [`Transport::steady`])
    active: bool,
    pinned: Option<(i64, usize)>, // (thread id, cpu)
    pred: Predictor,
    last_send: usize,
    st: SpinStats,
}

#[derive(Debug, Default)]
struct SpinStats {
    reads: u64,
    spun: u64,
    blocked: u64,
    sleeps: u64,
    overshoots: u64,
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
        Ok(Self {
            s,
            peer,
            spin: SpinCfg::from_env(),
            active: false,
            pinned: None,
            pred: Predictor::default(),
            last_send: 0,
            st: SpinStats::default(),
        })
    }

    /// Override the spin-receive settings (tests; the default comes from [`SpinCfg::from_env`]).
    pub fn set_spin(&mut self, c: SpinCfg) {
        self.spin = c;
    }

    /// Bounded spin-receive (see [`crate::spin`]): poll `recv(MSG_DONTWAIT)` until `buf` is full, re-arming the
    /// window on every byte; once a window passes with nothing, block. A header read in adapt mode first sleeps
    /// until shortly before the predicted arrival.
    fn read_spin(&mut self, buf: &mut [u8], header: bool) -> io::Result<()> {
        let fd = self.s.as_raw_fd();
        let win = Duration::from_micros(u64::from(self.spin.us));
        let class = self.last_send;
        let t0 = Instant::now();
        let mut off = 0usize;
        let mut first = true;
        let mut after_sleep = false;
        let mut deadline = t0 + win;
        if header && self.spin.adapt {
            if let Some(d) = self.pred.sleep_for(class, win) {
                if !readable(fd) {
                    std::thread::sleep(d);
                    self.st.sleeps += 1;
                    after_sleep = true;
                    deadline = Instant::now() + win * 2;
                }
            }
        }
        self.st.reads += 1;
        while off < buf.len() {
            let rest = &mut buf[off..];
            let r = unsafe {
                libc::recv(
                    fd,
                    rest.as_mut_ptr().cast::<libc::c_void>(),
                    rest.len(),
                    libc::MSG_DONTWAIT,
                )
            };
            let (n, how) = if r > 0 {
                (r as usize, 0)
            } else if r == 0 {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "link closed"));
            } else {
                let e = io::Error::last_os_error();
                match e.kind() {
                    io::ErrorKind::Interrupted => continue,
                    io::ErrorKind::WouldBlock => {
                        after_sleep = false;
                        if Instant::now() < deadline {
                            std::hint::spin_loop();
                            continue;
                        }
                        // the window passed: block (honours the read timeout)
                        let n = loop {
                            match self.s.read(rest) {
                                Ok(0) => {
                                    return Err(io::Error::new(
                                        io::ErrorKind::UnexpectedEof,
                                        "link closed",
                                    ))
                                }
                                Ok(n) => break n,
                                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                                Err(e) => return Err(e),
                            }
                        };
                        (n, 1)
                    }
                    _ => return Err(e),
                }
            };
            if first {
                first = false;
                if how == 0 {
                    self.st.spun += 1;
                } else {
                    self.st.blocked += 1;
                }
                if header && self.spin.adapt {
                    if after_sleep {
                        // the data was already there when the sleep ended: the prediction overshot
                        self.st.overshoots += 1;
                        self.pred.forget(class);
                    } else {
                        self.pred.record(class, t0.elapsed());
                    }
                }
            }
            off += n;
            deadline = Instant::now() + win;
        }
        Ok(())
    }
}

/// Whether a receive would return data now (`poll` with a zero timeout).
fn readable(fd: i32) -> bool {
    let mut p = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };
    unsafe { libc::poll(&mut p, 1, 0) > 0 && p.revents != 0 }
}

impl Drop for TcpTransport {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let st = &self.st;
        eprintln!(
            "talus-cluster: spin-receive: {} reads, {} served while spinning, {} blocked, {} sleeps ({} overshot)",
            st.reads, st.spun, st.blocked, st.sleeps, st.overshoots
        );
        if let Some((tid, _)) = self.pinned {
            if tid == spin::tid() {
                spin::unpin_thread();
            }
        }
    }
}

impl TcpTransport {
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
        self.last_send = bufs.iter().map(|b| b.len()).sum();
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
        if self.active {
            return self.read_spin(buf, false);
        }
        self.s.read_exact(buf)
    }

    fn read_header(&mut self, buf: &mut [u8]) -> io::Result<()> {
        if self.active {
            return self.read_spin(buf, true);
        }
        self.s.read_exact(buf)
    }

    fn steady(&mut self) {
        if self.active || !self.spin.enabled() {
            return;
        }
        self.active = true;
        let cpu = match self.spin.cpu {
            PinSpec::None => None,
            PinSpec::Cpu(c) => Some(c),
            PinSpec::Irq => self
                .s
                .local_addr()
                .ok()
                .and_then(|a| spin::irq_cpu_for(a.ip())),
        };
        let pin = match cpu {
            Some(c) => match spin::pin_thread(c) {
                Ok(()) => {
                    self.pinned = Some((spin::tid(), c));
                    format!("pinned to cpu {c}")
                }
                Err(e) => format!("not pinned (cpu {c}: {e})"),
            },
            None => "not pinned".to_string(),
        };
        eprintln!(
            "talus-cluster: data link spin-receive {} us{}, {pin}",
            self.spin.us,
            if self.spin.adapt { " (adaptive)" } else { "" }
        );
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
    use crate::{Header, Link, MsgType};
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

    /// Spin-receive (plain and adaptive) delivers the same bytes as a blocking link, including messages that
    /// arrive after the spin window (blocking fallback) and a closed peer.
    #[test]
    fn spin_receive_same_bytes() {
        for adapt in [false, true] {
            let l = Link::listen("127.0.0.1:0").unwrap();
            let addr = l.local_addr().unwrap().to_string();
            let th = std::thread::spawn(move || {
                let mut w = Link::accept(&l).unwrap();
                for i in 0..12u32 {
                    let h = w.recv_header().unwrap();
                    let b = w.recv_vec(&h).unwrap();
                    // every 4th reply comes after the 300 us window (blocking path / predictor overshoot)
                    std::thread::sleep(Duration::from_micros(if i % 4 == 3 { 5000 } else { 50 }));
                    let mut r = Header::new(MsgType::Ret);
                    r.step = h.step;
                    w.send(r, &[&b, &b]).unwrap();
                }
            });
            let s = TcpStream::connect(&addr).unwrap();
            let mut t = TcpTransport::new(s).unwrap();
            t.set_spin(SpinCfg {
                us: 300,
                adapt,
                cpu: PinSpec::None,
            });
            let mut c = Link::with_transport(Box::new(t));
            c.steady();
            for i in 0..12u32 {
                let a: Vec<u8> = (0..70_000u32 + i).map(|k| (k * 13 + i) as u8).collect();
                let mut h = Header::new(MsgType::Step);
                h.step = u64::from(i);
                c.send(h, &[&a]).unwrap();
                let (rh, body) = c.expect(MsgType::Ret).unwrap();
                assert_eq!(rh.step, u64::from(i));
                assert_eq!(body.len(), 2 * a.len());
                assert!(body[..a.len()] == a[..] && body[a.len()..] == a[..]);
            }
            th.join().unwrap();
            assert!(c.recv_header().is_err()); // the peer closed: EOF, not a hang
        }
    }
}
