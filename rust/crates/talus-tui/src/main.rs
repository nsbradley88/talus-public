//! `talus-tui`: terminal UI for talus-server. See docs/TUI.md.

use crossterm::event::{self, Event, KeyEventKind};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use talus_tui::app::{Action, App, Modal, Screen};
use talus_tui::client::{error_message, Client};
use talus_tui::fake::{self, FakeState};
use talus_tui::settings::{self, clamp_interval, Settings};
use talus_tui::stats::{self, fmt_tokens, summarize, Snapshot};
use talus_tui::ui;

const USAGE: &str = "\
Usage:
  talus-tui [--server URL] [--interval SECONDS] [--api-key-file PATH]
  talus-tui --once [--server URL]          print the status once and exit
  talus-tui fake-daemon [--port N] [--model ID] [--read-only]
                                          a fake server with synthetic stats (no GPU), for trying the TUI

The API key is read from TALUS_API_KEY, else STRATA_API_KEY, else --api-key-file, else ~/.config/talus/api-key.
Settings are kept in ~/.config/talus/tui.json.";

struct Args {
    server: Option<String>,
    interval_s: Option<f64>,
    key_file: Option<PathBuf>,
    once: bool,
    screenshot: Option<String>,
    fake: bool,
    port: u16,
    model: String,
    read_only: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        server: None,
        interval_s: None,
        key_file: None,
        once: false,
        screenshot: None,
        fake: false,
        port: 8000,
        model: "qwen3.8-27b".into(),
        read_only: false,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let val = |i: &mut usize, name: &str| -> Result<String, String> {
        *i += 1;
        argv.get(*i)
            .cloned()
            .ok_or_else(|| format!("{name} needs a value"))
    };
    while i < argv.len() {
        match argv[i].as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            "--server" | "-s" => a.server = Some(val(&mut i, "--server")?),
            "--interval" => {
                a.interval_s = Some(
                    val(&mut i, "--interval")?
                        .parse()
                        .map_err(|_| "bad --interval")?,
                )
            }
            "--api-key-file" => a.key_file = Some(PathBuf::from(val(&mut i, "--api-key-file")?)),
            "--once" => a.once = true,
            "--screenshot" => a.screenshot = Some(val(&mut i, "--screenshot")?),
            "fake-daemon" => a.fake = true,
            "--port" => a.port = val(&mut i, "--port")?.parse().map_err(|_| "bad --port")?,
            "--model" => a.model = val(&mut i, "--model")?,
            "--read-only" => a.read_only = true,
            other => return Err(format!("unknown argument {other:?} (try --help)")),
        }
        i += 1;
    }
    Ok(a)
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("talus-tui: {e}");
            std::process::exit(2);
        }
    };
    if args.fake {
        run_fake(&args);
        return;
    }
    if let Some(screen) = &args.screenshot {
        print!("{}", screenshot(screen));
        return;
    }

    let cfg_dir = settings::config_dir();
    let settings_path = cfg_dir.as_ref().map(|d| d.join("tui.json"));
    let key_path = args
        .key_file
        .clone()
        .or_else(|| cfg_dir.as_ref().map(|d| d.join("api-key")));
    let mut st = settings_path
        .as_deref()
        .map(Settings::load)
        .unwrap_or_default();
    if let Some(s) = &args.server {
        st.server = s.clone();
    }
    if let Some(i) = args.interval_s {
        st.interval_ms = clamp_interval((i * 1000.0) as u64);
    }
    let key = ["TALUS_API_KEY", "STRATA_API_KEY"]
        .iter()
        .filter_map(|k| std::env::var(k).ok())
        .map(|k| k.trim().to_string())
        .find(|k| !k.is_empty())
        .or_else(|| key_path.as_deref().and_then(settings::read_key_file));
    let client = match Client::new(&st.server, key) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("talus-tui: {e}");
            std::process::exit(2);
        }
    };

    if args.once {
        std::process::exit(print_once(&client));
    }

    let mut app = App::new(st);
    app.settings_path = settings_path;
    app.key_path = key_path;
    if let Err(e) = run(app, client) {
        eprintln!("talus-tui: {e}");
        std::process::exit(1);
    }
}

fn print_once(client: &Client) -> i32 {
    match client.get("/v1/talus/status") {
        Ok((200, v)) => {
            let s = stats::parse_status(&v);
            let sum = summarize(&s.recent, ui::SUMMARY_N);
            let name = s
                .preset
                .as_deref()
                .and_then(talus_presets::model)
                .map(|m| m.name.to_string())
                .unwrap_or(s.model.clone());
            println!("model     {name} ({})", s.state);
            println!(
                "level     {}",
                s.selection
                    .as_ref()
                    .map(|x| x.level.as_str())
                    .unwrap_or("balanced")
            );
            println!(
                "context   {} of {} tokens",
                fmt_tokens(s.ctx_in_use),
                fmt_tokens(s.ctx_max)
            );
            if let Some(a) = s.mem_avail_gib {
                println!("memory    {a:.1} GiB free");
            }
            let f = |v: Option<f64>| v.map(|x| format!("{x:.1}")).unwrap_or_else(|| "-".into());
            println!(
                "speed     answer {} tok/s · prompt {} tok/s · 1st token {} s · accepted {} %  (last {} requests)",
                f(sum.decode_tps),
                f(sum.prompt_tps),
                f(sum.ttft_ms.map(|m| m / 1000.0)),
                f(sum.acceptance.map(|a| a * 100.0)),
                sum.n
            );
            println!(
                "control   {}",
                if s.control_enabled {
                    "on"
                } else {
                    "off (read-only)"
                }
            );
            if let Some(e) = &s.last_error {
                println!("last change failed: {e}");
            }
            0
        }
        Ok((code, v)) => {
            eprintln!(
                "talus-tui: {} answered {code}: {}",
                client.url(),
                error_message(&v)
            );
            1
        }
        Err(e) => {
            eprintln!("talus-tui: {e}");
            1
        }
    }
}

enum Msg {
    Status(Box<Result<Snapshot, String>>),
    Presets(Value),
}

fn run(mut app: App, client: Client) -> std::io::Result<()> {
    let url = client.url();
    let client = Arc::new(Mutex::new(client));
    let interval = Duration::from_millis(app.settings.interval_ms);
    let (tx, rx) = mpsc::channel::<Msg>();
    let (wake_tx, wake_rx) = mpsc::channel::<()>();

    // The poller: one status request per interval (and on demand), presets now and then.
    let pc = Arc::clone(&client);
    std::thread::spawn(move || {
        let mut n: u64 = 0;
        loop {
            let c = pc.lock().unwrap().clone();
            let status = match c.get("/v1/talus/status") {
                Ok((200, v)) => Ok(stats::parse_status(&v)),
                Ok((401, _)) => {
                    Err("the server wants an API key (set TALUS_API_KEY or --api-key-file)".into())
                }
                Ok((404, _)) => {
                    Err("this server has no /v1/talus/status (an older talus-server?)".into())
                }
                Ok((code, v)) => Err(format!("status {code}: {}", error_message(&v))),
                Err(e) => Err(e),
            };
            let ok = status.is_ok();
            if tx.send(Msg::Status(Box::new(status))).is_err() {
                return;
            }
            if ok && n.is_multiple_of(15) {
                if let Ok((200, v)) = c.get("/v1/talus/presets") {
                    let _ = tx.send(Msg::Presets(v));
                }
            }
            n = if ok { n + 1 } else { 0 };
            match wake_rx.recv_timeout(interval) {
                Ok(()) | Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => return,
            }
        }
    });

    let mut terminal = ratatui::init();
    let res = (|| -> std::io::Result<()> {
        loop {
            while let Ok(m) = rx.try_recv() {
                match m {
                    Msg::Status(s) => app.on_status(*s),
                    Msg::Presets(v) => app.on_presets(&v),
                }
            }
            terminal.draw(|f| ui::draw(f, &app, &url))?;
            if !event::poll(Duration::from_millis(150))? {
                continue;
            }
            let Event::Key(k) = event::read()? else {
                continue;
            };
            if k.kind != KeyEventKind::Press {
                continue;
            }
            match app.on_key(k) {
                Action::None => {}
                Action::Quit => {
                    if let Some(p) = &app.settings_path {
                        let _ = app.settings.save(p);
                    }
                    return Ok(());
                }
                Action::Refresh => {
                    let _ = wake_tx.send(());
                }
                Action::Apply { body, new_key } => {
                    let c = client.lock().unwrap().clone();
                    let reply = c.post("/v1/talus/apply", &body);
                    if let Some(k) = app.on_apply_result(reply, new_key) {
                        client.lock().unwrap().set_api_key(Some(k));
                    }
                    let _ = wake_tx.send(());
                }
            }
        }
    })();
    ratatui::restore();
    res
}

fn run_fake(args: &Args) {
    let Some(spec) = talus_presets::model(&args.model) else {
        eprintln!("talus-tui: unknown model {:?}", args.model);
        std::process::exit(2);
    };
    let mut st = FakeState::new(spec.id);
    st.control = !args.read_only;
    for _ in 0..12 {
        st.tick();
    }
    match fake::spawn(args.port, st) {
        Ok((addr, _state)) => {
            println!(
                "fake talus-server (synthetic stats, no GPU) on http://{addr}  model {}",
                spec.name
            );
            println!("try:  talus-tui --server http://{addr}");
            loop {
                std::thread::sleep(Duration::from_secs(3600));
            }
        }
        Err(e) => {
            eprintln!("talus-tui: cannot listen on port {}: {e}", args.port);
            std::process::exit(1);
        }
    }
}

/// Text renders of each screen from fake data (used for docs/TUI.md).
fn screenshot(which: &str) -> String {
    let model = match which {
        "models" => "qwen3.8-27b",
        _ => "mimo-v2.6-flash",
    };
    let mut fs = FakeState::new(model);
    fs.port = 8000;
    for _ in 0..24 {
        fs.tick();
    }
    let mut app = App::new(Settings::default());
    app.on_status(Ok(stats::parse_status(&fs.status())));
    let mut presets = fs.presets();
    if let Some(a) = presets["models"].as_array_mut() {
        for m in a.iter_mut() {
            if m["id"] == "glm-5.3-flash" {
                m["available"] = Value::Bool(false);
            }
        }
    }
    app.on_presets(&presets);
    match which {
        "models" => {
            app.screen = Screen::Models;
            app.model_cursor = 3;
        }
        "tuning" | "confirm" => {
            app.select_model(3);
            app.on_key(key(crossterm::event::KeyCode::Right)); // Fast
            if which == "confirm" {
                app.on_key(key(crossterm::event::KeyCode::Char('a')));
                if let Modal::Confirm { .. } = app.modal {
                } else {
                    return "no confirmation\n".into();
                }
            } else {
                app.tune_cursor = app
                    .rows()
                    .iter()
                    .position(|r| matches!(r, talus_tui::app::Row::Opt(o) if o.id == "context"))
                    .unwrap_or(0);
                app.on_key(key(crossterm::event::KeyCode::Left)); // context 192K -> 128K
            }
        }
        _ => app.screen = Screen::Status,
    }
    ui::render_to_string(&app, "http://127.0.0.1:8000", 100, 32)
}

fn key(code: crossterm::event::KeyCode) -> crossterm::event::KeyEvent {
    crossterm::event::KeyEvent::new(code, crossterm::event::KeyModifiers::NONE)
}
