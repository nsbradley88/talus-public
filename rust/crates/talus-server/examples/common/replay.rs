//! Replays recorded model output through the parser: one JSON case per input line (`id`, `stream`, `tools`, `finish`, `thinking`),
//! one JSON result per configuration per output line, with a digest of the events at each chunk width (call ids left out).
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::{BufRead, Write};
use talus_server::frontend::{Event, EventKind, OutputParser};
use talus_server::json::JsonValue;

pub type Configure = fn(OutputParser) -> OutputParser;

fn parse_events(mut p: OutputParser, stream: &str, width: usize, finish: &str) -> Vec<Event> {
    let mut evs = Vec::new();
    let mut i = 0;
    while i < stream.len() {
        let mut e = if width == 0 {
            stream.len()
        } else {
            (i + width).min(stream.len())
        };
        while !stream.is_char_boundary(e) {
            e += 1;
        }
        evs.extend(p.feed(&stream[i..e]));
        i = e;
    }
    evs.extend(p.finish_with_reason(finish));
    evs
}

/// Neighbouring text of one kind joined, so the digest does not depend on how text was cut into events.
fn normal(evs: &[Event]) -> Vec<(u8, String)> {
    let mut out: Vec<(u8, String)> = Vec::new();
    for ev in evs {
        let (kind, text) = match ev.kind {
            EventKind::Reasoning => (0, ev.text.clone()),
            EventKind::Content => (1, ev.text.clone()),
            EventKind::ToolStart => (
                2,
                ev.call.as_ref().map(|c| c.name.clone()).unwrap_or_default(),
            ),
            EventKind::ToolArgs => (3, ev.text.clone()),
            EventKind::ToolCall => (
                4,
                ev.call
                    .as_ref()
                    .map(|c| format!("{} {}", c.name, c.arguments.to_json_string()))
                    .unwrap_or_default(),
            ),
        };
        match out.last_mut() {
            Some(last) if last.0 == kind && kind < 2 => last.1.push_str(&text),
            _ => out.push((kind, text)),
        }
    }
    out
}

fn digest(n: &[(u8, String)]) -> String {
    let mut h = Sha256::new();
    for (k, t) in n {
        h.update([*k]);
        h.update((t.len() as u64).to_le_bytes());
        h.update(t.as_bytes());
    }
    h.finalize()
        .iter()
        .take(12)
        .map(|b| format!("{b:02x}"))
        .collect()
}

pub fn run(configs: &[(&str, Configure)]) {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    for line in stdin.lock().lines() {
        let case: Value = serde_json::from_str(&line.expect("read")).expect("a JSON case");
        let stream = case["stream"].as_str().expect("stream");
        let finish = case["finish"].as_str().unwrap_or("stop");
        let thinking = case["thinking"].as_bool().unwrap_or(true);
        let names: Vec<Value> = case["tools"]
            .as_array()
            .map(|t| t.iter().map(|n| json!({"name": n})).collect())
            .unwrap_or_default();
        let tools = JsonValue::parse(&Value::Array(names).to_string()).expect("tools");
        let tools = tools.as_array().map(|t| t.to_vec());
        for (name, configure) in configs {
            let make = || configure(OutputParser::new(thinking, tools.clone(), true, true, true));
            let whole = normal(&parse_events(make(), stream, 0, finish));
            let w1 = digest(&normal(&parse_events(make(), stream, 1, finish)));
            let w7 = digest(&normal(&parse_events(make(), stream, 7, finish)));
            let calls: Vec<&str> = whole
                .iter()
                .filter(|(k, _)| *k == 4)
                .map(|(_, t)| t.as_str())
                .collect();
            let text_len = |kind: u8| {
                whole
                    .iter()
                    .filter(|(k, _)| *k == kind)
                    .map(|(_, t)| t.len())
                    .sum::<usize>()
            };
            let row = json!({"id": case["id"], "cfg": name, "whole": digest(&whole), "w1": w1, "w7": w7, "calls": calls,
                "reasoning": text_len(0), "content": text_len(1)});
            writeln!(out, "{row}").expect("write");
        }
    }
}
