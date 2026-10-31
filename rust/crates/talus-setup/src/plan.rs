//! The arithmetic behind the choices: does a model fit in memory right now, and is there disk for the download.

use crate::manifest::{Choices, FileSpec, ModelInstall};
use crate::probe::SysInfo;
use std::path::Path;

/// Memory the release machine reports (MemTotal of a 128 GB Strix Halo, GiB).
pub const REF_TOTAL_GIB: f64 = 125.0;
/// What an idle headless system uses (GiB): the release runs' free figures are relative to that.
pub const HEADLESS_USED_GIB: f64 = 3.0;
/// The engines stop below this much free memory.
pub const HARD_FLOOR_GIB: f64 = 8.0;

#[derive(Clone, Debug, PartialEq)]
pub enum Fit {
    Fits,
    Tight,
    /// Does not fit as the machine is now; the reason says what to change.
    NotNow(String),
}

impl Fit {
    pub fn label(&self) -> &'static str {
        match self {
            Fit::Fits => "fits",
            Fit::Tight => "tight",
            Fit::NotNow(_) => "not now",
        }
    }
}

/// Free memory (GiB) to expect while `m` serves, given what this machine uses now.
pub fn predicted_free_gib(m: &ModelInstall, sys: &SysInfo) -> f64 {
    let total = sys.mem_total_gib();
    let used_now = (total - sys.mem_avail_gib()).max(0.0);
    let extra = (used_now - HEADLESS_USED_GIB).max(0.0);
    let short = (REF_TOTAL_GIB - total).max(0.0);
    // a smaller machine also gets a shorter automatic context (the 27B: down to 64K), and the engine drops its
    // optional speed copies (the 27B's wide tile-plane copies) first when memory is short or other programs use it
    let loss = extra + (short - m.auto_context_gib as f64).max(0.0);
    m.release_free_gib as f64 - (loss - m.auto_pack_gib as f64).max(0.0)
}

/// Whether `m` fits now. `pool_ok`: the GPU memory pool is already raised (only matters for the models that need it).
pub fn live_fit(m: &ModelInstall, sys: &SysInfo, pool_ok: bool) -> Fit {
    if m.needs_big_gtt && !pool_ok {
        return Fit::NotNow(
            "needs the GPU memory pool raised (kernel parameters, then a reboot)".into(),
        );
    }
    if let Some(min) = m.min_start_gib {
        let avail = sys.mem_avail_gib();
        if avail < min as f64 {
            return Fit::NotNow(format!(
                "starts only with {min} GiB free; {avail:.0} GiB free now (run it without a desktop session)"
            ));
        }
    }
    let floor = m.spec().min_free_gib as f64;
    let free = predicted_free_gib(m, sys);
    if free >= floor + 4.0 {
        Fit::Fits
    } else if free >= HARD_FLOOR_GIB {
        Fit::Tight
    } else {
        Fit::NotNow(format!(
            "about {:.0} GiB short with what runs now; close programs or run without a desktop session",
            (HARD_FLOOR_GIB - free).ceil().max(1.0)
        ))
    }
}

/// The name of a download in progress.
pub fn part_name(f: &FileSpec) -> String {
    format!("{}.part", f.local_name())
}

/// Bytes of `f` already in `dir`: the whole file when it has the right size, else what a partial download holds.
pub fn bytes_present(dir: &Path, f: &FileSpec) -> u64 {
    let size = |p: &Path| std::fs::metadata(p).map(|m| m.len()).ok();
    match size(&dir.join(f.local_name())) {
        Some(s) if s == f.size => f.size,
        _ => size(&dir.join(part_name(f))).unwrap_or(0).min(f.size),
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DiskNeed {
    /// Bytes of every selected file.
    pub total: u64,
    /// Bytes already on disk (whole files and partial downloads).
    pub present: u64,
    /// Bytes the preparation steps write.
    pub prep: u64,
}

impl DiskNeed {
    pub fn remaining(&self) -> u64 {
        self.total - self.present.min(self.total) + self.prep
    }
    /// Keep a margin: 2 GiB or 1 %, whichever is larger.
    pub fn margin(&self) -> u64 {
        (2u64 << 30).max(self.remaining() / 100)
    }
    pub fn fits(&self, free: u64) -> bool {
        self.remaining() + self.margin() <= free
    }
}

pub fn disk_need(models: &[(&ModelInstall, Choices)], models_dir: &Path) -> DiskNeed {
    let mut n = DiskNeed::default();
    for (m, c) in models {
        let dir = models_dir.join(m.id);
        for f in m.selected_files(c) {
            n.total += f.size;
            n.present += bytes_present(&dir, f);
        }
        let prepped = match m.prep {
            crate::manifest::Prep::None => true,
            crate::manifest::Prep::FlashNextPack => {
                dir.join("pack/native_experts.txt").exists()
                    && (!c.drafters || dir.join("mtp/rt/experts.bin").exists())
            }
        };
        if !prepped {
            n.prep += m.prep_bytes_for(c);
        }
    }
    n
}

pub fn gb(b: u64) -> String {
    let g = b as f64 / 1e9;
    if g >= 10.0 {
        format!("{g:.0} GB")
    } else if g >= 0.1 {
        format!("{g:.1} GB")
    } else {
        format!("{:.1} MB", b as f64 / 1e6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest;

    fn sys(total_gib: f64, avail_gib: f64) -> SysInfo {
        SysInfo {
            mem_total_kib: (total_gib * 1048576.0) as u64,
            mem_avail_kib: (avail_gib * 1048576.0) as u64,
            ..Default::default()
        }
    }

    #[test]
    fn fit_matches_the_docs_on_the_reference_machine() {
        let headless = sys(125.0, 122.0);
        let desktop = sys(125.0, 116.0); // a GNOME session uses about 6 GiB more
        let id = |i| manifest::get(i).unwrap();
        assert_eq!(live_fit(id("qwen3.8-27b"), &desktop, false), Fit::Fits);
        assert_eq!(
            live_fit(id("qwen3.8-flash-next"), &desktop, false),
            Fit::Fits
        );
        assert_eq!(live_fit(id("laguna-s-2.1"), &desktop, false), Fit::Fits);
        assert_eq!(live_fit(id("mimo-v2.6-flash"), &desktop, true), Fit::Tight);
        assert!(matches!(
            live_fit(id("glm-5.3-flash"), &desktop, true),
            Fit::NotNow(_)
        ));
        assert_eq!(live_fit(id("glm-5.3-flash"), &headless, true), Fit::Tight);
        // without the raised pool, GLM and MiMo never fit
        assert!(
            matches!(live_fit(id("mimo-v2.6-flash"), &headless, false), Fit::NotNow(r) if r.contains("pool"))
        );
        // a busy machine
        let busy = sys(125.0, 66.0);
        assert!(matches!(
            live_fit(id("laguna-s-2.1"), &busy, true),
            Fit::NotNow(_)
        ));
        // the 27B's 256K default context needs about 12 GiB more than the old 64K
        assert!(matches!(
            live_fit(id("qwen3.8-27b"), &busy, true),
            Fit::NotNow(_)
        ));
        assert_eq!(
            live_fit(id("qwen3.8-27b"), &sys(125.0, 78.0), true),
            Fit::Tight
        );
    }

    #[test]
    fn disk_math_counts_partial_downloads() {
        let dir = crate::testutil::tempdir("disk");
        let m = manifest::get("qwen3.8-27b").unwrap();
        let c = Choices::default();
        let n0 = disk_need(&[(m, c)], &dir);
        assert_eq!(n0.present, 0);
        assert_eq!(n0.total, m.download_bytes(&c));
        let md = dir.join(m.id);
        std::fs::create_dir_all(&md).unwrap();
        // the vocab complete, the MTP head half there
        let vocab = m.file(manifest::Role::DraftVocab).unwrap();
        std::fs::write(md.join(vocab.local_name()), vec![0u8; vocab.size as usize]).unwrap();
        let mtp = m.files[1];
        std::fs::File::create(md.join(part_name(&mtp)))
            .unwrap()
            .set_len(1000)
            .unwrap();
        let n1 = disk_need(&[(m, c)], &dir);
        assert_eq!(n1.present, vocab.size + 1000);
        assert_eq!(n1.remaining(), n0.remaining() - vocab.size - 1000);
        // a wrong-size final file does not count
        std::fs::write(md.join(m.files[0].local_name()), b"x").unwrap();
        assert_eq!(disk_need(&[(m, c)], &dir).present, n1.present);
        assert!(!n1.fits(n1.remaining()));
        assert!(n1.fits(n1.remaining() + n1.margin()));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn prep_space_for_flash_next() {
        let dir = crate::testutil::tempdir("prep");
        let m = manifest::get("qwen3.8-flash-next").unwrap();
        let on = disk_need(&[(m, Choices::default())], &dir);
        let off = disk_need(
            &[(
                m,
                Choices {
                    drafters: false,
                    ..Default::default()
                },
            )],
            &dir,
        );
        assert!(on.prep > off.prep && off.prep > 0);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn sizes_read_well() {
        assert_eq!(gb(14_252_845_984), "14 GB");
        assert_eq!(gb(2_142_248_960), "2.1 GB");
        assert_eq!(gb(262_144), "0.3 MB");
    }
}
