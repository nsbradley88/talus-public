//! The Bench screen: launch a quality smoke run (talus-bench) against the served model and follow it.
//!
//! The smoke run is Talus-vs-Talus: the served model at Balanced (our tuned defaults) is the reference and the
//! chosen level (Fast, or Reference when the model has no Fast opt-in) is the candidate. Each side is applied
//! through /v1/talus/apply (an engine restart each), then json_schema (20) and gsm8k (20) run at temperature 0.
//! On a read-only server the same settings run twice (a repeatability check). The run itself happens in
//! talus-bench-core on a worker thread; this module only holds state and builds the config.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use talus_bench_core::config::Config;
use talus_bench_core::gates::{GateStatus, Verdict};
use talus_bench_core::runner::{Event, RunOutcome};
use talus_presets::{Level, ModelSpec};

#[derive(Clone, Debug, PartialEq)]
pub struct BenchResult {
    pub dir: PathBuf,
    pub verdicts: Vec<Verdict>,
}

pub struct BenchState {
    /// The candidate level; None = pick for the served model (Fast if it has one, else Reference).
    pub level: Option<Level>,
    pub running: bool,
    pub phase: String,
    pub model: String,
    pub done: usize,
    pub total: usize,
    pub passes: usize,
    pub fails: usize,
    pub log: VecDeque<String>,
    pub result: Option<BenchResult>,
    pub error: Option<String>,
    pub cancel: Arc<AtomicBool>,
    pub started: Option<Instant>,
}

impl Default for BenchState {
    fn default() -> Self {
        Self {
            level: None,
            running: false,
            phase: String::new(),
            model: String::new(),
            done: 0,
            total: 0,
            passes: 0,
            fails: 0,
            log: VecDeque::new(),
            result: None,
            error: None,
            cancel: Arc::new(AtomicBool::new(false)),
            started: None,
        }
    }
}

/// Where TUI runs keep their artifacts: `$XDG_DATA_HOME/talus-bench/runs` or `~/.local/share/talus-bench/runs`.
pub fn runs_dir() -> PathBuf {
    let base = std::env::var("XDG_DATA_HOME")
        .ok()
        .filter(|x| !x.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into())).join(".local/share")
        });
    base.join("talus-bench/runs")
}

impl BenchState {
    pub fn level_for(&self, spec: Option<&ModelSpec>) -> Level {
        self.level.unwrap_or(match spec {
            Some(s) if s.has_fast() => Level::Fast,
            _ => Level::Reference,
        })
    }

    pub fn cycle_level(&mut self, spec: Option<&ModelSpec>) {
        let cur = self.level_for(spec);
        let has_fast = spec.map(|s| s.has_fast()).unwrap_or(false);
        self.level = Some(match cur {
            Level::Fast => Level::Reference,
            _ if has_fast => Level::Fast,
            _ => Level::Reference,
        });
    }

    fn push_log(&mut self, s: String) {
        self.log.push_back(s);
        while self.log.len() > 200 {
            self.log.pop_front();
        }
    }

    /// Reset for a new run and hand out the cancel flag the worker will watch.
    pub fn start(&mut self) -> Arc<AtomicBool> {
        let level = self.level;
        *self = BenchState {
            level,
            running: true,
            phase: "starting".into(),
            started: Some(Instant::now()),
            ..Default::default()
        };
        Arc::clone(&self.cancel)
    }

    pub fn request_cancel(&mut self) {
        if self.running {
            self.cancel.store(true, Ordering::Relaxed);
            self.phase = "cancelling after the current request...".into();
        }
    }

    pub fn on_event(&mut self, e: Event) {
        match e {
            Event::Phase(p) => {
                self.push_log(p.clone());
                self.phase = p;
            }
            Event::Log(l) => self.push_log(l),
            Event::ModelStart { model, total } => {
                self.model = model;
                self.total = total;
                self.done = 0;
                self.passes = 0;
                self.fails = 0;
            }
            Event::Case {
                done,
                pass,
                status,
                case_id,
                ..
            } => {
                self.done = done;
                if pass {
                    self.passes += 1;
                } else {
                    self.fails += 1;
                    if !status.scored() {
                        self.push_log(format!("{case_id}: {}", status.as_str()));
                    }
                }
            }
            Event::ModelDone {
                model,
                correct,
                total,
            } => self.push_log(format!("{model}: {correct}/{total} passed")),
        }
    }

    pub fn on_done(&mut self, res: Result<RunOutcome, String>) {
        self.running = false;
        match res {
            Ok(o) => {
                self.phase = format!(
                    "done in {} s",
                    self.started.map(|t| t.elapsed().as_secs()).unwrap_or(0)
                );
                self.result = Some(BenchResult {
                    dir: o.output_dir,
                    verdicts: o.verdicts,
                });
            }
            Err(e) => {
                self.phase = "stopped".into();
                self.error = Some(e);
            }
        }
    }

    /// The smoke config for the served model. `control` = the server accepts /v1/talus/apply.
    pub fn smoke_config(&self, server_url: &str, preset: Option<&str>, control: bool) -> Config {
        let spec = preset.and_then(talus_presets::model);
        let level = self.level_for(spec);
        let out = runs_dir();
        let model = if control { spec.map(|s| s.id) } else { None };
        let mut cfg = talus_bench_core::smoke_config(server_url, model, Some(level.id()), None, "");
        cfg.experiment.output_dir = Some(out.join(&cfg.experiment.id).display().to_string());
        if !control {
            cfg.experiment.description = Some(
                "TUI smoke on a read-only server: the served settings twice (repeatability check)"
                    .into(),
            );
            cfg.reference.id = "served-a".into();
            cfg.candidates[0].id = "served-b".into();
        }
        cfg
    }

    /// One-line verdict per candidate.
    pub fn summary_lines(&self) -> Vec<(GateStatus, String)> {
        let Some(r) = &self.result else {
            return Vec::new();
        };
        r.verdicts
            .iter()
            .map(|v| {
                (
                    v.overall,
                    format!(
                        "{}: {}  {}",
                        v.candidate_id,
                        v.overall.as_str(),
                        v.reasons.join("; ")
                    ),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use talus_bench_core::case::CaseStatus;

    #[test]
    fn level_choice_follows_the_model() {
        let mut b = BenchState::default();
        let mimo = talus_presets::model("mimo-v2.6-flash");
        assert_eq!(b.level_for(mimo), Level::Fast);
        b.cycle_level(mimo);
        assert_eq!(b.level_for(mimo), Level::Reference);
        b.cycle_level(mimo);
        assert_eq!(b.level_for(mimo), Level::Fast);
        // a model without a Fast opt-in compares against Reference (the catalog decides which models those are)
        if let Some(plain) = talus_presets::MODELS.iter().find(|m| !m.has_fast()) {
            let mut c = BenchState::default();
            assert_eq!(c.level_for(Some(plain)), Level::Reference, "{}", plain.id);
            c.cycle_level(Some(plain));
            assert_eq!(c.level_for(Some(plain)), Level::Reference);
        }
        assert_eq!(BenchState::default().level_for(None), Level::Reference);
    }

    #[test]
    fn events_update_progress() {
        let mut b = BenchState::default();
        let cancel = b.start();
        b.on_event(Event::ModelStart {
            model: "fast".into(),
            total: 4,
        });
        b.on_event(Event::Case {
            model: "fast".into(),
            done: 1,
            total: 4,
            case_id: "a".into(),
            status: CaseStatus::Ok,
            pass: true,
        });
        b.on_event(Event::Case {
            model: "fast".into(),
            done: 2,
            total: 4,
            case_id: "b".into(),
            status: CaseStatus::Timeout,
            pass: false,
        });
        assert_eq!((b.done, b.passes, b.fails), (2, 1, 1));
        assert!(b.log.back().unwrap().contains("timeout"));
        b.request_cancel();
        assert!(cancel.load(Ordering::Relaxed));
        b.on_done(Err("cancelled".into()));
        assert!(!b.running);
        assert_eq!(b.error.as_deref(), Some("cancelled"));
    }

    #[test]
    fn smoke_config_is_valid() {
        let b = BenchState::default();
        let c = b.smoke_config("http://127.0.0.1:8000", Some("mimo-v2.6-flash"), true);
        assert!(c.validate().ok(), "{:?}", c.validate());
        assert_eq!(
            c.candidates[0].talus.as_ref().unwrap().level.as_deref(),
            Some("fast")
        );
        assert!(c.reference.talus.as_ref().unwrap().apply);
        let ro = b.smoke_config("http://127.0.0.1:8000", Some("mimo-v2.6-flash"), false);
        assert!(ro.validate().ok());
        assert!(ro.reference.talus.is_none());
        assert_eq!(ro.candidates[0].id, "served-b");
    }
}
