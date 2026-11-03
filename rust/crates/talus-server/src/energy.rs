//! GPU energy counter for `GET /v1/talus/status` (`energy`; bench server extension E5).
//!
//! A thread samples the amdgpu hwmon power file (`power1_average`, else `power1_input`, microwatts) every
//! [`INTERVAL`] and integrates joules (trapezoids over the measured sample times). Read-only: sysfs reads, no
//! clock or power settings. The thread starts with the first status request (a client measuring J/token reads
//! status before and after its cell), so a server nobody asks does no sampling. `TALUS_ENERGY=0` turns it off.
//! What it measures: see [`NOTE`] (on Strix Halo the driver labels it `PPT`, the package power).

use crate::json::{unix_timestamp_secs, JsonValue};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const INTERVAL: Duration = Duration::from_millis(100);
/// What the number is: the amdgpu driver's power reading. On Strix Halo the file is labelled `PPT` (the APU's package
/// power tracking value as the SMU reports it), so it covers the APU package, not the wall and not the GPU alone.
pub const NOTE: &str =
    "amdgpu hwmon power as the driver reports it (APU: GPU/SoC package side), not wall power";

/// The first `card<N>/device/hwmon/hwmon*/power1_average` (else `power1_input`) under `drm` (normally
/// /sys/class/drm), cards in numeric order.
pub fn find_power_file(drm: &Path) -> Option<PathBuf> {
    let mut cards: Vec<(u32, PathBuf)> = std::fs::read_dir(drm)
        .ok()?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            let n = name.strip_prefix("card")?.parse::<u32>().ok()?;
            Some((n, e.path()))
        })
        .collect();
    cards.sort();
    for (_, card) in cards {
        let Ok(rd) = std::fs::read_dir(card.join("device/hwmon")) else {
            continue;
        };
        let mut hw: Vec<PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path()).collect();
        hw.sort();
        for file in ["power1_average", "power1_input"] {
            if let Some(p) = hw
                .iter()
                .map(|h| h.join(file))
                .find(|p| read_watts(p).is_some())
            {
                return Some(p);
            }
        }
    }
    None
}

/// Watts from a hwmon power file (microwatts).
pub fn read_watts(p: &Path) -> Option<f64> {
    std::fs::read_to_string(p)
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(|uw| uw as f64 / 1e6)
}

#[derive(Debug, Default, Clone)]
struct Acc {
    joules: f64,
    watts_last: Option<f64>,
    last_t: Option<Instant>,
    samples: u64,
    since_unix: i64,
}

impl Acc {
    fn add(&mut self, t: Instant, w: f64) {
        if let (Some(t0), Some(w0)) = (self.last_t, self.watts_last) {
            let dt = t.saturating_duration_since(t0).as_secs_f64();
            self.joules += 0.5 * (w0 + w) * dt;
        }
        self.last_t = Some(t);
        self.watts_last = Some(w);
        self.samples += 1;
    }
}

pub struct EnergyMeter {
    source: Option<PathBuf>,
    started: AtomicBool,
    acc: Mutex<Acc>,
}

impl EnergyMeter {
    /// Finds the power file now (a directory listing); samples nothing until [`Self::start`].
    pub fn new() -> Self {
        let off = std::env::var("TALUS_ENERGY")
            .map(|v| v.trim() == "0")
            .unwrap_or(false);
        let source = if off {
            None
        } else {
            find_power_file(Path::new("/sys/class/drm"))
        };
        Self::with_source(source)
    }

    pub fn with_source(source: Option<PathBuf>) -> Self {
        Self {
            source,
            started: AtomicBool::new(false),
            acc: Mutex::new(Acc::default()),
        }
    }

    /// One sample at time `t` (the sampler thread's step; public for tests).
    pub fn sample_at(&self, t: Instant) {
        if let Some(w) = self.source.as_deref().and_then(read_watts) {
            self.acc.lock().unwrap().add(t, w);
        }
    }

    /// Start the sampler thread once (no-op without a power file).
    pub fn start(self: &Arc<Self>) {
        if self.source.is_none() || self.started.swap(true, Ordering::SeqCst) {
            return;
        }
        self.acc.lock().unwrap().since_unix = unix_timestamp_secs();
        self.sample_at(Instant::now());
        let me = Arc::clone(self);
        let _ = std::thread::Builder::new()
            .name("talus-energy".to_string())
            .spawn(move || loop {
                std::thread::sleep(INTERVAL);
                me.sample_at(Instant::now());
            });
    }

    /// The `energy` object, or null without a power file.
    pub fn to_json(&self) -> JsonValue {
        let Some(src) = &self.source else {
            return JsonValue::Null;
        };
        let a = self.acc.lock().unwrap().clone();
        JsonValue::obj(vec![
            (
                "joules_total",
                JsonValue::from((a.joules * 1000.0).round() / 1000.0),
            ),
            (
                "watts_last",
                a.watts_last.map(JsonValue::from).unwrap_or(JsonValue::Null),
            ),
            ("source", JsonValue::from(src.to_string_lossy().as_ref())),
            (
                "label",
                std::fs::read_to_string(src.with_file_name("power1_label"))
                    .ok()
                    .map(|l| JsonValue::from(l.trim()))
                    .unwrap_or(JsonValue::Null),
            ),
            ("samples", JsonValue::from(a.samples)),
            ("interval_ms", JsonValue::from(INTERVAL.as_millis() as u64)),
            ("since_unix", JsonValue::from(a.since_unix)),
            ("note", JsonValue::from(NOTE)),
        ])
    }
}

impl Default for EnergyMeter {
    fn default() -> Self {
        Self::with_source(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_and_integrates() {
        let d = std::env::temp_dir().join(format!("talus-energy-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        // card0 has no hwmon; card1-DP-1 is a connector; card10 sorts after card2
        std::fs::create_dir_all(d.join("card0/device")).unwrap();
        std::fs::create_dir_all(d.join("card1-DP-1")).unwrap();
        let h2 = d.join("card2/device/hwmon/hwmon3");
        let h10 = d.join("card10/device/hwmon/hwmon1");
        std::fs::create_dir_all(&h2).unwrap();
        std::fs::create_dir_all(&h10).unwrap();
        std::fs::write(h10.join("power1_average"), "1000000\n").unwrap();
        std::fs::write(h2.join("power1_input"), "20000000\n").unwrap();
        assert_eq!(find_power_file(&d), Some(h2.join("power1_input")));
        std::fs::write(h2.join("power1_average"), "garbage").unwrap();
        assert_eq!(find_power_file(&d), Some(h2.join("power1_input")));
        std::fs::write(h2.join("power1_average"), "30000000").unwrap();
        let p = find_power_file(&d).unwrap();
        assert_eq!(p, h2.join("power1_average"));

        let m = EnergyMeter::with_source(Some(p.clone()));
        let t0 = Instant::now();
        m.sample_at(t0);
        std::fs::write(&p, "10000000").unwrap();
        m.sample_at(t0 + Duration::from_millis(500));
        m.sample_at(t0 + Duration::from_millis(1500));
        // (30+10)/2*0.5 + 10*1.0 = 20 J
        let j = m.to_json();
        assert_eq!(j.get_f64("joules_total"), Some(20.0));
        assert_eq!(j.get_f64("watts_last"), Some(10.0));
        assert_eq!(j.get_u64("samples"), Some(3));
        assert_eq!(j.get_str("note"), Some(NOTE));
        assert!(EnergyMeter::with_source(None).to_json().is_null());
        assert!(find_power_file(&d.join("missing")).is_none());
        let _ = std::fs::remove_dir_all(&d);
    }
}
