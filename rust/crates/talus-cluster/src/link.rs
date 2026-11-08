//! A persistent connection between two stages. One message = a 64 B header + payload parts (+ an optional CRC32C
//! tail), written with one vectored write sequence. The bytes move over a [`Transport`]: TCP today
//! ([`TcpTransport`]: `TCP_NODELAY`, 4 MiB socket buffers, `TCP_QUICKACK` after each receive, optional busy-poll).

use std::io::{self, IoSlice};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

use crate::frame::{crc32c, crc32c_update, Header, MsgType, FLAG_CRC, HDR_BYTES};
use crate::transport::{Closer, TcpTransport, Transport};

pub struct Link {
    t: Box<dyn Transport>,
    crc: bool,
    pub peer: String,
    pub sent_msgs: u64,
    pub sent_bytes: u64,
    pub recv_bytes: u64,
}

impl Link {
    /// A link over any transport.
    pub fn with_transport(t: Box<dyn Transport>) -> Self {
        let peer = t.peer();
        Self {
            t,
            crc: crate::crc_enabled(),
            peer,
            sent_msgs: 0,
            sent_bytes: 0,
            recv_bytes: 0,
        }
    }

    /// A TCP link (the default data plane).
    pub fn from_stream(s: TcpStream) -> io::Result<Self> {
        Ok(Self::with_transport(Box::new(TcpTransport::new(s)?)))
    }

    /// The transport's name (`tcp`, ...).
    pub fn kind(&self) -> &'static str {
        self.t.kind()
    }

    /// Connect to `addr`, retrying until `wait` elapses (the worker may still be loading its weights).
    pub fn connect(addr: &str, wait: Duration) -> io::Result<Self> {
        let t0 = Instant::now();
        loop {
            let mut last =
                io::Error::new(io::ErrorKind::NotFound, format!("no address for {addr}"));
            for a in addr.to_socket_addrs()? {
                match TcpStream::connect_timeout(&a, Duration::from_secs(2)) {
                    Ok(s) => return Self::from_stream(s),
                    Err(e) => last = e,
                }
            }
            if t0.elapsed() >= wait {
                return Err(last);
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }

    pub fn listen(addr: &str) -> io::Result<TcpListener> {
        TcpListener::bind(addr)
    }

    pub fn accept(l: &TcpListener) -> io::Result<Self> {
        let (s, _) = l.accept()?;
        Self::from_stream(s)
    }

    /// A handle that closes the link from another thread (the heartbeat, the worker's session teardown).
    pub fn closer(&self) -> io::Result<Closer> {
        self.t.closer()
    }

    /// The handshake is done and the engine is built: from now on this thread receives the pipeline steps (starts
    /// the transport's latency mode, e.g. `TALUS_CLUSTER_SPIN_US`).
    pub fn steady(&mut self) {
        self.t.steady()
    }

    pub fn set_read_timeout(&self, t: Option<Duration>) -> io::Result<()> {
        self.t.set_read_timeout(t)
    }

    /// Send one message; `hdr.payload_len` and the CRC flag are set here from `parts`.
    pub fn send(&mut self, mut hdr: Header, parts: &[&[u8]]) -> io::Result<()> {
        let mut n: u64 = parts.iter().map(|p| p.len() as u64).sum();
        let mut tail = [0u8; 4];
        if self.crc {
            let mut c = 0u32;
            for p in parts {
                c = crc32c_update(c, p);
            }
            tail = c.to_le_bytes();
            hdr.flags |= FLAG_CRC;
            n += 4;
        }
        hdr.payload_len = n;
        let hb = hdr.encode();
        let mut sl: Vec<IoSlice<'_>> = Vec::with_capacity(parts.len() + 2);
        sl.push(IoSlice::new(&hb));
        for p in parts {
            if !p.is_empty() {
                sl.push(IoSlice::new(p));
            }
        }
        if self.crc {
            sl.push(IoSlice::new(&tail));
        }
        self.t.write_all_vectored(&mut sl)?;
        self.sent_msgs += 1;
        self.sent_bytes += HDR_BYTES as u64 + n;
        Ok(())
    }

    pub fn send_msg(&mut self, ty: MsgType, payload: &[u8]) -> io::Result<()> {
        self.send(Header::new(ty), &[payload])
    }

    pub fn recv_header(&mut self) -> io::Result<Header> {
        let mut b = [0u8; HDR_BYTES];
        self.t.read_header(&mut b)?;
        self.t.after_recv();
        Header::decode(&b).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// The payload size without the CRC tail.
    pub fn body_len(h: &Header) -> usize {
        (h.payload_len as usize).saturating_sub(if h.flags & FLAG_CRC != 0 { 4 } else { 0 })
    }

    /// Read the payload of `h` into `buf[..body_len]` (`buf` must be large enough) and check its CRC if present.
    pub fn recv_into(&mut self, h: &Header, buf: &mut [u8]) -> io::Result<usize> {
        let n = Self::body_len(h);
        if n > buf.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("payload {n} B exceeds the receive buffer ({} B)", buf.len()),
            ));
        }
        self.t.read_exact(&mut buf[..n])?;
        if h.flags & FLAG_CRC != 0 {
            let mut t = [0u8; 4];
            self.t.read_exact(&mut t)?;
            if u32::from_le_bytes(t) != crc32c(&buf[..n]) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "payload crc32c mismatch",
                ));
            }
        }
        self.t.after_recv();
        self.recv_bytes += h.payload_len + HDR_BYTES as u64;
        Ok(n)
    }

    pub fn recv_vec(&mut self, h: &Header) -> io::Result<Vec<u8>> {
        let mut v = vec![0u8; Self::body_len(h)];
        self.recv_into(h, &mut v)?;
        Ok(v)
    }

    /// Receive one message of type `want` (an `Err` message from the peer becomes an error).
    pub fn expect(&mut self, want: MsgType) -> io::Result<(Header, Vec<u8>)> {
        let h = self.recv_header()?;
        let body = self.recv_vec(&h)?;
        match h.msg() {
            Some(t) if t == want => Ok((h, body)),
            Some(MsgType::Err) => Err(io::Error::other(format!(
                "peer refused: {}",
                String::from_utf8_lossy(&body)
            ))),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("expected {want:?}, got {other:?}"),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_roundtrip_with_crc() {
        let l = Link::listen("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap().to_string();
        let th = std::thread::spawn(move || {
            let mut w = Link::accept(&l).unwrap();
            let h = w.recv_header().unwrap();
            let mut buf = vec![0u8; 1 << 20];
            let n = w.recv_into(&h, &mut buf).unwrap();
            let mut r = Header::new(MsgType::Ret);
            r.rows = h.rows;
            w.send(r, &[&buf[..n]]).unwrap();
        });
        let mut c = Link::connect(&addr, Duration::from_secs(5)).unwrap();
        c.crc = true;
        let a: Vec<u8> = (0..300_000u32).map(|i| (i * 7) as u8).collect();
        let mut h = Header::new(MsgType::Step);
        h.rows = 3;
        c.send(h, &[&a[..1000], &a[1000..]]).unwrap();
        let (rh, body) = c.expect(MsgType::Ret).unwrap();
        assert_eq!(rh.rows, 3);
        assert_eq!(body, a);
        th.join().unwrap();
    }
}
