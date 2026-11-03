//! Heartbeat on a control connection of its own (DESIGN.md 3.3/3.5), so heartbeats never queue behind a bundle.
//! Each side writes one byte every `period` and reads the peer's; after `timeout` without a byte (or on any error) the
//! peer is declared dead and the data connection is shut down, which fails the blocked send/receive at once instead
//! of hanging on a silent link (a pulled cable, a powered-off box).

use std::io::{Read, Write};
use std::net::{Shutdown, TcpStream};

use crate::transport::Closer;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

pub struct Beat {
    stop: Arc<AtomicBool>,
    dead: Arc<AtomicBool>,
    th: Option<JoinHandle<()>>,
}

impl Beat {
    /// Start beating on `ctl`; on peer death close the data link through `data`. Period and timeout from `TALUS_CLUSTER_HB_MS`
    /// (default 500) and `TALUS_CLUSTER_DEAD_MS` (default 3000).
    pub fn start(ctl: TcpStream, data: Closer, what: &'static str) -> std::io::Result<Self> {
        let ms = |k: &str, d: u64| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(d)
        };
        let period = Duration::from_millis(ms("TALUS_CLUSTER_HB_MS", 500).max(10));
        let timeout = Duration::from_millis(ms("TALUS_CLUSTER_DEAD_MS", 3000).max(100));
        ctl.set_nodelay(true)?;
        ctl.set_read_timeout(Some(period))?;
        let stop = Arc::new(AtomicBool::new(false));
        let dead = Arc::new(AtomicBool::new(false));
        let (s2, d2) = (stop.clone(), dead.clone());
        let th = std::thread::Builder::new().name("talus-cluster-hb".into()).spawn(move || {
            let mut ctl = ctl;
            let mut last = Instant::now();
            let mut next = Instant::now();
            let mut buf = [0u8; 64];
            while !s2.load(Ordering::Relaxed) {
                if Instant::now() >= next {
                    if ctl.write_all(&[0x48]).is_err() {
                        break;
                    }
                    next = Instant::now() + period;
                }
                match ctl.read(&mut buf) {
                    Ok(0) => break,
                    Ok(_) => last = Instant::now(),
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock || e.kind() == std::io::ErrorKind::TimedOut => {}
                    Err(_) => break,
                }
                if last.elapsed() > timeout {
                    break;
                }
            }
            if !s2.load(Ordering::Relaxed) {
                d2.store(true, Ordering::Relaxed);
                eprintln!(
                    "talus-cluster: {what}: peer silent for {:.1} s or control link lost: shutting the data link down",
                    last.elapsed().as_secs_f64()
                );
                data.close();
            }
            let _ = ctl.shutdown(Shutdown::Both);
        })?;
        Ok(Self {
            stop,
            dead,
            th: Some(th),
        })
    }

    pub fn is_dead(&self) -> bool {
        self.dead.load(Ordering::Relaxed)
    }
}

impl Drop for Beat {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.th.take() {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn silent_peer_shuts_the_data_link() {
        std::env::set_var("TALUS_CLUSTER_HB_MS", "20");
        std::env::set_var("TALUS_CLUSTER_DEAD_MS", "200");
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let a = l.local_addr().unwrap();
        let ctl_a = TcpStream::connect(a).unwrap();
        let (ctl_b, _) = l.accept().unwrap();
        let data_a = TcpStream::connect(a).unwrap();
        let (mut data_b, _) = l.accept().unwrap();
        // side A beats; side B never answers (a silent peer)
        let beat = Beat::start(ctl_a, Closer::tcp(data_a), "test").unwrap();
        let t0 = Instant::now();
        let mut b = [0u8; 1];
        // B's data read sees EOF once A shut its side down
        data_b
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let n = data_b.read(&mut b).unwrap_or(0);
        assert_eq!(n, 0);
        assert!(t0.elapsed() < Duration::from_secs(3));
        assert!(beat.is_dead());
        drop(ctl_b);
    }
}
