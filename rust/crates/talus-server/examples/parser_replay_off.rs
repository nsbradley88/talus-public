//! `common/replay.rs` with the parser as the server runs it by default.
#[path = "common/replay.rs"]
mod replay;

fn main() {
    replay::run(&[("off", |p| p)]);
}
