//! Bounded spin-receive on the data link (WAVES W2, talus#48 option A follow-up).
//!
//! .17 measured that the cost of a stage hop is the receiver's idle wake-up, not the wire: after a 13 ms gap a
//! blocking `recv` returns ~1.0 ms later than a receiver that is already running (1335 vs 361 us p50 for a 27B T=1
//! round trip), because the receiving core has to leave a deep C-state and the interrupt is throttled. A receiver
//! that polls a non-blocking `recv` sees the data as soon as the NAPI softirq queued it.
//!
//! - `TALUS_CLUSTER_SPIN_US=N` (default [`DEFAULT_SPIN_US`] = 100 ms, on; 0 = off): every data-link receive first
//!   polls `recv(MSG_DONTWAIT)` for up to
//!   N µs (re-armed whenever bytes arrive), then falls back to the normal blocking read. N must cover the peer's
//!   compute time for a hop to gain anything (tens of ms for 27B decode), so a large N keeps one core per host busy
//!   for the whole decode.
//! - `TALUS_CLUSTER_SPIN_ADAPT=1` (opt-in; no gain measured): instead of
//!   spinning the whole gap, sleep until shortly before the expected arrival
//!   and spin a window of ±N µs around it. The expected wait is the minimum of the last 8 header waits that followed
//!   a send of the same size (decode, each verify width and each prefill chunk are separate classes). A wait that
//!   ended before the sleep did (overshoot) clears that class, so the next hop measures again without sleeping.
//! - `TALUS_CLUSTER_SPIN_CPU=irq|<cpu>|none` (default `irq`): pin the receiving thread to the CPU that takes the data
//!   interface's receive-ring MSI-X vector (thunderbolt0's RX ring interrupt), so the softirq and the poller share a core and
//!   its cache. The thread's old affinity is restored when the last spinning link on it is dropped.
//!
//! Spinning and pinning start at [`crate::Link::steady`], called by the engine once the handshake is done and the
//! engine is built (threads the engine starts later on that thread would inherit the pin). Timing only: the bytes
//! read are the same.

use std::collections::HashMap;
use std::io;
use std::net::IpAddr;
use std::time::Duration;

/// Default spin window: covers the peer's whole compute gap for 27B two-box decode (bitwise identical;
/// costs one busy core per host while a session decodes).
pub const DEFAULT_SPIN_US: u32 = 100_000;

/// Spin-receive settings, from the environment ([`SpinCfg::from_env`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpinCfg {
    /// Spin window in µs (0 = off).
    pub us: u32,
    /// Sleep until shortly before the predicted arrival, then spin ±`us`.
    pub adapt: bool,
    pub cpu: PinSpec,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PinSpec {
    /// The CPU that takes the data interface's receive-ring MSI-X vector.
    #[default]
    Irq,
    Cpu(usize),
    None,
}

impl SpinCfg {
    pub fn from_env() -> Self {
        let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        Self::parse(
            env("TALUS_CLUSTER_SPIN_US").as_deref(),
            env("TALUS_CLUSTER_SPIN_ADAPT").as_deref(),
            env("TALUS_CLUSTER_SPIN_CPU").as_deref(),
        )
    }

    pub fn parse(us: Option<&str>, adapt: Option<&str>, cpu: Option<&str>) -> Self {
        let cpu = match cpu.map(str::trim) {
            None | Some("irq") | Some("") => PinSpec::Irq,
            Some("none") | Some("off") | Some("-1") => PinSpec::None,
            Some(v) => v.parse().map(PinSpec::Cpu).unwrap_or(PinSpec::Irq),
        };
        Self {
            us: match us {
                None => DEFAULT_SPIN_US,
                Some(v) => v.trim().parse().unwrap_or(DEFAULT_SPIN_US),
            },
            adapt: adapt.is_some_and(|v| v.trim() == "1"),
            cpu,
        }
    }

    pub fn enabled(&self) -> bool {
        self.us > 0
    }
}

/// The CPU whose interrupts deliver packets for local address `local`: the interface holding that address, its PCI
/// function (for thunderbolt-net: the NHI behind it), its receive vector in /proc/interrupts ([`rx_msix`]),
/// and that IRQ's first effective CPU. None when any step is missing (loopback, no MSI-X, no /proc).
pub fn irq_cpu_for(local: IpAddr) -> Option<usize> {
    let ifname = ifname_for(local)?;
    let dev = std::fs::canonicalize(format!("/sys/class/net/{ifname}/device")).ok()?;
    let pci = dev
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .rfind(|s| is_pci_addr(s))?
        .to_string();
    let irq = rx_msix(&std::fs::read_to_string("/proc/interrupts").ok()?, &pci)?;
    let aff = std::fs::read_to_string(format!("/proc/irq/{irq}/effective_affinity_list")).ok()?;
    first_cpu(&aff)
}

fn is_pci_addr(s: &str) -> bool {
    // 0000:c5:00.5
    let b = s.as_bytes();
    b.len() == 12 && b[4] == b':' && b[7] == b':' && b[10] == b'.'
}

/// The IRQ of the receive ring among the `PCI-MSIX-<pci>` lines: the highest MSI-X vector index that has taken
/// interrupts. thunderbolt-net allocates its TX ring and then its RX ring after the NHI's control rings (vectors 0/1),
/// so RX is the last vector in use (2 = TX, 3 = RX on the laptop and the desktop; the TX and RX counts are nearly equal, so the
/// busiest line is not a reliable pick). For a NIC with one queue pair the same rule picks its last vector.
pub fn rx_msix(interrupts: &str, pci: &str) -> Option<u32> {
    let tag = format!("PCI-MSIX-{pci}");
    let mut best: Option<(u32, u32)> = None; // (vector index, irq)
    for l in interrupts.lines() {
        if !l.contains(&tag) {
            continue;
        }
        let mut it = l.split_whitespace();
        let Some(irq) = it
            .next()
            .and_then(|s| s.trim_end_matches(':').parse::<u32>().ok())
        else {
            continue;
        };
        let toks: Vec<&str> = it.collect();
        let n: u64 = toks.iter().map_while(|s| s.parse::<u64>().ok()).sum();
        let vec = toks
            .iter()
            .position(|t| t.contains(&tag))
            .and_then(|p| toks.get(p + 1))
            .and_then(|t| t.split('-').next())
            .and_then(|v| v.parse::<u32>().ok());
        let Some(vec) = vec else { continue };
        if n > 0 && best.is_none_or(|(b, _)| vec > b) {
            best = Some((vec, irq));
        }
    }
    best.map(|(_, i)| i)
}

/// First CPU of a cpulist such as `0`, `3-5,8`.
pub fn first_cpu(list: &str) -> Option<usize> {
    list.trim()
        .split([',', '-'])
        .next()
        .and_then(|s| s.trim().parse().ok())
}

fn ifname_for(local: IpAddr) -> Option<String> {
    let mut head: *mut libc::ifaddrs = std::ptr::null_mut();
    if unsafe { libc::getifaddrs(&mut head) } != 0 {
        return None;
    }
    let mut out = None;
    let mut p = head;
    while !p.is_null() {
        let a = unsafe { &*p };
        if !a.ifa_addr.is_null() {
            let fam = unsafe { (*a.ifa_addr).sa_family } as i32;
            let ip = match fam {
                libc::AF_INET => {
                    let s = unsafe { &*(a.ifa_addr as *const libc::sockaddr_in) };
                    Some(IpAddr::from(u32::from_be(s.sin_addr.s_addr).to_be_bytes()))
                }
                libc::AF_INET6 => {
                    let s = unsafe { &*(a.ifa_addr as *const libc::sockaddr_in6) };
                    Some(IpAddr::from(s.sin6_addr.s6_addr))
                }
                _ => None,
            };
            if ip == Some(local) {
                out = Some(
                    unsafe { std::ffi::CStr::from_ptr(a.ifa_name) }
                        .to_string_lossy()
                        .into_owned(),
                );
                break;
            }
        }
        p = a.ifa_next;
    }
    unsafe { libc::freeifaddrs(head) };
    out
}

thread_local! {
    /// (links on this thread holding the pin, the affinity before the first pin)
    static PIN: std::cell::RefCell<(u32, Option<libc::cpu_set_t>)> = const { std::cell::RefCell::new((0, None)) };
}

/// Pin the calling thread to `cpu` (nested: only the first call saves the old mask).
pub fn pin_thread(cpu: usize) -> io::Result<()> {
    PIN.with(|p| {
        let mut p = p.borrow_mut();
        if p.0 == 0 {
            let mut old: libc::cpu_set_t = unsafe { std::mem::zeroed() };
            if unsafe {
                libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut old)
            } != 0
            {
                return Err(io::Error::last_os_error());
            }
            let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
            unsafe { libc::CPU_SET(cpu, &mut set) };
            if unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) }
                != 0
            {
                return Err(io::Error::last_os_error());
            }
            p.1 = Some(old);
        }
        p.0 += 1;
        Ok(())
    })
}

/// Undo one [`pin_thread`]; the last one restores the old affinity. Must run on the pinning thread.
pub fn unpin_thread() {
    PIN.with(|p| {
        let mut p = p.borrow_mut();
        if p.0 == 0 {
            return;
        }
        p.0 -= 1;
        if p.0 == 0 {
            if let Some(old) = p.1.take() {
                unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &old) };
            }
        }
    })
}

/// Thread id (to check that a drop runs on the pinning thread).
pub fn tid() -> i64 {
    unsafe { libc::syscall(libc::SYS_gettid) }
}

/// Header-wait predictor for [`SpinCfg::adapt`]: per class (the size of the last send), the last 8 waits.
#[derive(Debug, Default)]
pub struct Predictor {
    hist: HashMap<usize, ([Duration; 8], u8, u8)>, // ring, next slot, filled
}

/// The longest sleep the predictor will ask for.
pub const MAX_SLEEP: Duration = Duration::from_millis(250);

impl Predictor {
    /// The expected wait for class `k`: the minimum of its last waits once at least 2 are known.
    pub fn expect(&self, k: usize) -> Option<Duration> {
        let (r, _, n) = self.hist.get(&k)?;
        if *n < 2 {
            return None;
        }
        r[..*n as usize].iter().min().copied()
    }

    pub fn record(&mut self, k: usize, w: Duration) {
        if self.hist.len() > 256 && !self.hist.contains_key(&k) {
            self.hist.clear(); // bounded: a client that sends every size just never sleeps
        }
        let e = self.hist.entry(k).or_insert(([Duration::ZERO; 8], 0, 0));
        e.0[e.1 as usize] = w;
        e.1 = (e.1 + 1) % 8;
        e.2 = (e.2 + 1).min(8);
    }

    pub fn forget(&mut self, k: usize) {
        self.hist.remove(&k);
    }

    /// How long to sleep before spinning, for a window of `win`: the expected wait minus the window (none below 2
    /// windows), capped at [`MAX_SLEEP`].
    pub fn sleep_for(&self, k: usize, win: Duration) -> Option<Duration> {
        let e = self.expect(k)?;
        (e > win * 2).then(|| (e - win).min(MAX_SLEEP))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cfg() {
        assert_eq!(SpinCfg::parse(None, None, None).us, DEFAULT_SPIN_US);
        assert!(!SpinCfg::parse(Some("0"), None, None).enabled());
        let c = SpinCfg::parse(Some("200"), Some("1"), Some("3"));
        assert_eq!((c.us, c.adapt, c.cpu), (200, true, PinSpec::Cpu(3)));
        assert_eq!(
            SpinCfg::parse(Some("5"), None, Some("none")).cpu,
            PinSpec::None
        );
        assert_eq!(SpinCfg::parse(Some("5"), None, None).cpu, PinSpec::Irq);
        assert_eq!(SpinCfg::parse(Some("x"), None, None).us, DEFAULT_SPIN_US);
    }

    #[test]
    fn msix_and_cpulist() {
        let s = "           CPU0       CPU1\n\
                  59:          0      76074  PCI-MSIX-0000:c5:00.5    0-edge      thunderbolt\n\
                  61:          0          0  PCI-MSIX-0000:c5:00.5    2-edge      thunderbolt\n\
                  62:    7220502          0  PCI-MSIX-0000:c5:00.5    3-edge      thunderbolt\n\
                  77:          0    9999999  PCI-MSIX-0000:c5:00.6    0-edge      thunderbolt\n";
        assert_eq!(rx_msix(s, "0000:c5:00.5"), Some(62));
        assert_eq!(rx_msix(s, "0000:c5:00.6"), Some(77));
        assert_eq!(rx_msix(s, "0000:c5:00.7"), None);
        // the desktop's shape: TX (vector 2) busier than RX (vector 3); an unused vector 4 is skipped
        let h = "  66:    5838688          0  PCI-MSIX-0000:c7:00.6    2-edge      thunderbolt\n\
                   67:          0    5796244  PCI-MSIX-0000:c7:00.6    3-edge      thunderbolt\n\
                   68:          0          0  PCI-MSIX-0000:c7:00.6    4-edge      thunderbolt\n";
        assert_eq!(rx_msix(h, "0000:c7:00.6"), Some(67));
        assert_eq!(first_cpu("0\n"), Some(0));
        assert_eq!(first_cpu("3-5,8"), Some(3));
        assert!(is_pci_addr("0000:c5:00.5") && !is_pci_addr("domain0"));
    }

    #[test]
    fn predictor() {
        let ms = Duration::from_millis;
        let mut p = Predictor::default();
        let w = Duration::from_micros(500);
        assert_eq!(p.sleep_for(1, w), None);
        p.record(1, ms(30));
        assert_eq!(p.sleep_for(1, w), None); // one sample is not enough
        p.record(1, ms(20));
        p.record(1, ms(40));
        assert_eq!(p.expect(1), Some(ms(20)));
        assert_eq!(p.sleep_for(1, w), Some(ms(20) - w));
        assert_eq!(p.sleep_for(2, w), None); // other class
        for _ in 0..8 {
            p.record(1, ms(900));
        }
        assert_eq!(p.sleep_for(1, w), Some(MAX_SLEEP)); // old minimum aged out; capped
        p.forget(1);
        assert_eq!(p.sleep_for(1, w), None);
        p.record(3, Duration::from_micros(700));
        p.record(3, Duration::from_micros(700));
        assert_eq!(p.sleep_for(3, w), None); // shorter than two windows: spin only
    }

    #[test]
    fn pin_nesting_restores() {
        let mut before: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut before) };
        let first = (0..libc::CPU_SETSIZE as usize)
            .find(|&c| unsafe { libc::CPU_ISSET(c, &before) })
            .unwrap();
        std::thread::spawn(move || {
            pin_thread(first).unwrap();
            pin_thread(first).unwrap();
            unpin_thread();
            let mut now: libc::cpu_set_t = unsafe { std::mem::zeroed() };
            unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut now) };
            assert_eq!(unsafe { libc::CPU_COUNT(&now) }, 1);
            unpin_thread();
            unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut now) };
            assert!(unsafe { libc::CPU_EQUAL(&now, &before) });
        })
        .join()
        .unwrap();
    }

    /// On a host: `TALUS_TEST_LOCAL_IP=<the link IP> cargo test -p talus-cluster irq_cpu_probe -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn irq_cpu_probe() {
        let ip: IpAddr = std::env::var("TALUS_TEST_LOCAL_IP")
            .unwrap()
            .parse()
            .unwrap();
        println!("ifname {:?} irq cpu {:?}", ifname_for(ip), irq_cpu_for(ip));
    }
}
