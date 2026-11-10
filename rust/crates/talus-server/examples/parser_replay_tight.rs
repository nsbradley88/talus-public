//! `common/replay.rs` with tight tool-call recovery off and on.
#[path = "common/replay.rs"]
mod replay;

fn main() {
    replay::run(&[("off", |p| p), ("tight", |p| p.with_tight_recovery(true))]);
}
