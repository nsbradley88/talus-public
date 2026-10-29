//! Guided install, health checks and updates for Talus: the `talus` command.
//!
//! - `talus setup`: hardware report, model choice, downloads, kernel parameters (asked, never silent), run configs,
//!   the optional user service.
//! - `talus doctor`: a verdict per check and the fixes (`--fix`), or a redacted report for an issue (`--report`).
//! - `talus update`: new release, rebuild, migrated configs, doctor.
//! - `talus serve`: what the service runs (talus-server on the control directory).
//!
//! The checks live in [`checks`] so talus-tui can show the same health panel.

pub mod boot;
pub mod checks;
pub mod config;
pub mod doctor;
pub mod download;
pub mod exec;
pub mod manifest;
pub mod paths;
pub mod plan;
pub mod prepare;
pub mod probe;
pub mod report;
pub mod service;
pub mod state;
pub mod support;
pub mod ui;
pub mod update;
pub mod wizard;

#[cfg(test)]
pub(crate) mod testutil;
