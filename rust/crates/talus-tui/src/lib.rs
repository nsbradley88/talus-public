//! talus-tui: a terminal UI for talus-server. It shows the serving statistics (sampled every one to two seconds
//! from counters the server already keeps), lets you pick which model the server runs, and tune it with the
//! per-model presets from `talus-presets`. Changes go through the server's control endpoint, which restarts the
//! engine; the TUI itself never touches the GPU.

pub mod app;
pub mod client;
pub mod fake;
pub mod settings;
pub mod stats;
pub mod ui;
