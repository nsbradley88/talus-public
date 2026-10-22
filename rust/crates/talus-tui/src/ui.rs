//! Drawing. One function per screen; `draw` picks the screen and overlays a dialog when one is open.

use crate::app::{App, Modal, Row, Screen};
use crate::stats::{fmt_tokens, fmt_uptime, summarize, Snapshot};
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Cell, Clear, Gauge, Paragraph, Row as TRow, Sparkline, Table, Wrap};
use ratatui::Frame;
use talus_presets::{Bind, Group, Level, Quality, MODELS};

const ACCENT: Color = Color::Cyan;
const GOOD: Color = Color::Green;
const WARN: Color = Color::Yellow;
const BAD: Color = Color::Red;
const DIM: Color = Color::DarkGray;

/// Requests the speed summary averages over.
pub const SUMMARY_N: usize = 10;

pub fn draw(f: &mut Frame, app: &App, server_url: &str) {
    let [head, body, foot] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(5),
        Constraint::Length(1),
    ])
    .areas(f.area());
    draw_header(f, head, app, server_url);
    match app.screen {
        Screen::Status => draw_status(f, body, app, server_url),
        Screen::Models => draw_models(f, body, app),
        Screen::Tuning => draw_tuning(f, body, app),
    }
    draw_footer(f, foot, app);
    match &app.modal {
        Modal::None => {}
        Modal::Confirm {
            title,
            lines,
            warning,
            ..
        } => {
            let mut text: Vec<Line> = lines.iter().map(|l| Line::from(l.as_str())).collect();
            if let Some(w) = warning {
                text.push(Line::from(""));
                text.push(Line::from(Span::styled(
                    format!("! {w}"),
                    Style::new().fg(WARN).bold(),
                )));
            }
            text.push(Line::from(""));
            text.push(Line::from(vec![
                Span::styled("[y]", Style::new().fg(GOOD).bold()),
                Span::raw(" apply   "),
                Span::styled("[n]", Style::new().fg(BAD).bold()),
                Span::raw(" cancel"),
            ]));
            dialog(f, title, text);
        }
        Modal::Message { title, lines } => {
            let mut text: Vec<Line> = lines.iter().map(|l| Line::from(l.as_str())).collect();
            text.push(Line::from(""));
            text.push(Line::from(Span::styled(
                "press any key",
                Style::new().fg(DIM),
            )));
            dialog(f, title, text);
        }
        Modal::Help => dialog(f, "Keys", help_lines()),
    }
}

fn help_lines() -> Vec<Line<'static>> {
    [
        ("1 2 3 / Tab", "switch screen: Status, Models, Tuning"),
        ("Up Down / j k", "move"),
        ("Enter", "Models: tune this model"),
        ("Left Right / h l", "Tuning: change the value"),
        ("d", "Tuning: this option back to our tuned default"),
        ("r", "Tuning: every option back to our tuned defaults"),
        ("a", "Tuning: apply (asks first; restarts the engine)"),
        ("g / F5", "refresh now"),
        ("q / Esc", "quit (the server keeps running)"),
    ]
    .into_iter()
    .map(|(k, v)| {
        Line::from(vec![
            Span::styled(format!("{k:<18}"), Style::new().fg(ACCENT).bold()),
            Span::raw(v),
        ])
    })
    .collect()
}

fn dialog(f: &mut Frame, title: &str, text: Vec<Line>) {
    let area = f.area();
    let max_inner = area.width.saturating_sub(6).clamp(20, 76);
    let inner = text
        .iter()
        .map(|l| l.width() as u16)
        .max()
        .unwrap_or(20)
        .clamp(30, max_inner);
    let width = (inner + 2).min(area.width);
    // lines longer than the box wrap
    let rows: u16 = text
        .iter()
        .map(|l| (l.width() as u16).div_ceil(inner).max(1))
        .sum();
    let height = (rows + 2).min(area.height);
    let r = Rect {
        x: area.x + (area.width - width) / 2,
        y: area.y + (area.height - height) / 2,
        width,
        height,
    };
    f.render_widget(Clear, r);
    f.render_widget(
        Paragraph::new(text).wrap(Wrap { trim: false }).block(
            Block::bordered()
                .title(format!(" {title} "))
                .border_style(Style::new().fg(ACCENT)),
        ),
        r,
    );
}

fn draw_header(f: &mut Frame, area: Rect, app: &App, url: &str) {
    let mut spans = vec![
        Span::styled(" Talus ", Style::new().fg(Color::Black).bg(ACCENT).bold()),
        Span::raw(" "),
    ];
    for (i, s) in Screen::ALL.iter().enumerate() {
        let label = format!(" {} {} ", i + 1, s.title());
        if *s == app.screen {
            spans.push(Span::styled(
                label,
                Style::new().add_modifier(Modifier::REVERSED | Modifier::BOLD),
            ));
        } else {
            spans.push(Span::raw(label));
        }
    }
    f.render_widget(Paragraph::new(Line::from(spans)), area);
    let conn = match (&app.last_error, &app.snapshot) {
        (None, Some(_)) => Span::styled(format!("● {url} "), Style::new().fg(GOOD)),
        (Some(_), _) if app.restart_model.is_some() => {
            Span::styled(format!("◌ restarting {url} "), Style::new().fg(WARN))
        }
        (Some(_), _) => Span::styled(format!("✗ no answer from {url} "), Style::new().fg(BAD)),
        (None, None) => Span::styled(format!("… {url} "), Style::new().fg(DIM)),
    };
    f.render_widget(
        Paragraph::new(Line::from(conn)).alignment(Alignment::Right),
        area,
    );
}

fn draw_footer(f: &mut Frame, area: Rect, app: &App) {
    let hints = match app.screen {
        Screen::Status => "1-3 screens · g refresh · ? keys · q quit (the server keeps running)",
        Screen::Models => "↑↓ move · Enter tune this model · ? keys · q quit",
        Screen::Tuning => "↑↓ move · ←→ change · d default · r reset all · a apply · ? keys",
    };
    let line = match &app.notice {
        Some(n) => Line::from(vec![
            Span::styled(format!(" {n} "), Style::new().fg(WARN).bold()),
            Span::styled(hints, Style::new().fg(DIM)),
        ]),
        None => Line::from(Span::styled(format!(" {hints}"), Style::new().fg(DIM))),
    };
    f.render_widget(Paragraph::new(line), area);
}

fn model_name(id: Option<&str>, fallback: &str) -> String {
    id.and_then(talus_presets::model)
        .map(|m| m.name.to_string())
        .unwrap_or_else(|| fallback.to_string())
}

fn level_text(s: &Snapshot) -> String {
    match &s.selection {
        Some(sel) => match Level::parse(&sel.level) {
            Some(Level::Balanced) => "Balanced (our tuned defaults)".into(),
            Some(l) => l.label().into(),
            None => "Custom".into(),
        },
        None => "Balanced (our tuned defaults)".into(),
    }
}

fn kv(k: &str, v: Vec<Span<'static>>) -> Line<'static> {
    let mut spans = vec![Span::styled(format!(" {k:<11}"), Style::new().fg(DIM))];
    spans.extend(v);
    Line::from(spans)
}

fn draw_status(f: &mut Frame, area: Rect, app: &App, url: &str) {
    let Some(s) = app.snapshot.as_ref() else {
        let mut lines = vec![
            Line::from(""),
            Line::from(format!("  Waiting for talus-server at {url} ...")),
            Line::from(""),
            Line::from("  Start the server first, for example:"),
            Line::from("    talus-server --control-dir /etc/talus"),
            Line::from("  or point the TUI elsewhere:  talus-tui --server http://HOST:PORT"),
        ];
        if let Some(e) = &app.last_error {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                format!("  {e}"),
                Style::new().fg(BAD),
            )));
        }
        f.render_widget(
            Paragraph::new(lines).block(Block::bordered().title(" Status ")),
            area,
        );
        return;
    };
    let [top, mid, bottom] = Layout::vertical([
        Constraint::Length(7),
        Constraint::Length(6),
        Constraint::Min(4),
    ])
    .areas(area);

    // Serving
    let (state_txt, state_col) = match s.state.as_str() {
        "idle" => ("● idle", GOOD),
        "generating" => ("● generating", ACCENT),
        "restarting" => ("◌ restarting: serving pauses", WARN),
        "unloaded" => ("○ unloaded", DIM),
        other => (other, DIM),
    };
    let spec = s.preset.as_deref().and_then(talus_presets::model);
    let mem = match (s.mem_avail_gib, s.mem_total_gib) {
        (Some(a), Some(t)) => {
            let floor = spec.map(|m| m.min_free_gib as f64).unwrap_or(8.0);
            let col = if a < floor {
                BAD
            } else if a < floor + 4.0 {
                WARN
            } else {
                GOOD
            };
            vec![
                Span::styled(format!("{a:.1} GiB free"), Style::new().fg(col)),
                Span::raw(format!(
                    " of {t:.0} GiB   (keep at least {floor:.0} GiB free)"
                )),
            ]
        }
        _ => vec![Span::styled("not reported", Style::new().fg(DIM))],
    };
    let key = if s.api_key {
        "API key required"
    } else {
        "no API key"
    };
    let accuracy = match &s.last_error {
        Some(e) => vec![
            Span::raw(level_text(s)),
            Span::styled(format!("   last change failed: {e}"), Style::new().fg(WARN)),
        ],
        None => vec![Span::raw(level_text(s))],
    };
    let lines = vec![
        kv(
            "Model",
            vec![
                Span::styled(
                    model_name(s.preset.as_deref(), &s.model),
                    Style::new().bold(),
                ),
                Span::raw("   "),
                Span::styled(state_txt.to_string(), Style::new().fg(state_col)),
                Span::styled(
                    format!("   up {}", fmt_uptime(s.uptime_s)),
                    Style::new().fg(DIM),
                ),
            ],
        ),
        kv("Accuracy", accuracy),
        kv("Memory", mem),
        kv(
            "Server",
            vec![Span::raw(format!(
                "{}:{} · {} · {}",
                s.host,
                s.port,
                Bind::from_host(&s.host).label(),
                key
            ))],
        ),
        kv(
            "Requests",
            vec![Span::raw(format!(
                "{} served · {} running · {} waiting",
                s.totals.requests, s.running, s.queued
            ))],
        ),
    ];
    let block = Block::bordered().title(" Serving ");
    let inner = block.inner(top);
    f.render_widget(block, top);
    let [info, ctx] = Layout::vertical([Constraint::Length(5), Constraint::Length(1)]).areas(inner);
    f.render_widget(Paragraph::new(lines), info);
    let ratio = if s.ctx_max > 0 {
        (s.ctx_in_use as f64 / s.ctx_max as f64).min(1.0)
    } else {
        0.0
    };
    let [cl, cg] = Layout::horizontal([Constraint::Length(12), Constraint::Min(10)]).areas(ctx);
    f.render_widget(
        Paragraph::new(Span::styled(" Context", Style::new().fg(DIM))),
        cl,
    );
    f.render_widget(
        Gauge::default()
            .ratio(ratio)
            .label(format!(
                "{} of {} tokens (last request)",
                fmt_tokens(s.ctx_in_use),
                fmt_tokens(s.ctx_max)
            ))
            .gauge_style(Style::new().fg(ACCENT).bg(Color::Black)),
        cg,
    );

    // Speed
    let sum = summarize(&s.recent, SUMMARY_N);
    let fmt = |v: Option<f64>, unit: &str, prec: usize| match v {
        Some(x) => format!("{x:.prec$} {unit}"),
        None => "-".to_string(),
    };
    let speed = vec![
        kv(
            "Answer",
            vec![
                Span::styled(fmt(sum.decode_tps, "tok/s", 1), Style::new().bold()),
                Span::styled("  generating the reply", Style::new().fg(DIM)),
            ],
        ),
        kv(
            "Prompt",
            vec![
                Span::styled(fmt(sum.prompt_tps, "tok/s", 0), Style::new().bold()),
                Span::styled("  reading new prompt text", Style::new().fg(DIM)),
            ],
        ),
        kv(
            "1st token",
            vec![
                Span::styled(
                    fmt(sum.ttft_ms.map(|m| m / 1000.0), "s", 2),
                    Style::new().bold(),
                ),
                Span::styled("  wait before the reply starts", Style::new().fg(DIM)),
            ],
        ),
        kv(
            "Accepted",
            vec![
                Span::styled(
                    fmt(sum.acceptance.map(|a| a * 100.0), "%", 0),
                    Style::new().bold(),
                ),
                Span::styled("  drafted tokens kept", Style::new().fg(DIM)),
            ],
        ),
    ];
    let [sl, sr] =
        Layout::horizontal([Constraint::Percentage(58), Constraint::Percentage(42)]).areas(mid);
    f.render_widget(
        Paragraph::new(speed)
            .block(Block::bordered().title(format!(" Speed, last {} requests ", sum.n))),
        sl,
    );
    let mut spark: Vec<u64> = s
        .recent
        .iter()
        .rev()
        .filter_map(|r| r.decode_tps())
        .map(|v| v.round() as u64)
        .collect();
    if spark.is_empty() {
        spark.push(0);
    }
    f.render_widget(
        Sparkline::default()
            .data(&spark)
            .style(Style::new().fg(ACCENT))
            .block(Block::bordered().title(" Answer tok/s per request ")),
        sr,
    );

    // Recent requests
    let header = TRow::new([
        "prompt",
        "reused",
        "answer",
        "prompt tok/s",
        "answer tok/s",
        "1st token",
        "accepted",
        "end",
    ])
    .style(Style::new().fg(DIM));
    let rows: Vec<TRow> = s
        .recent
        .iter()
        .take(bottom.height.saturating_sub(3) as usize)
        .map(|r| {
            TRow::new(vec![
                Cell::from(fmt_tokens(r.prompt_tokens)),
                Cell::from(fmt_tokens(r.reused)),
                Cell::from(fmt_tokens(r.output_tokens)),
                Cell::from(
                    r.prompt_tps()
                        .map(|v| format!("{v:.0}"))
                        .unwrap_or_else(|| "-".into()),
                ),
                Cell::from(
                    r.decode_tps()
                        .map(|v| format!("{v:.1}"))
                        .unwrap_or_else(|| "-".into()),
                ),
                Cell::from(if r.ttft_ms > 0.0 {
                    format!("{:.2} s", r.ttft_ms / 1000.0)
                } else {
                    "-".into()
                }),
                Cell::from(
                    r.acceptance()
                        .map(|a| format!("{:.0}%", a * 100.0))
                        .unwrap_or_else(|| "-".into()),
                ),
                Cell::from(r.finish.clone()),
            ])
        })
        .collect();
    let widths = [
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Length(7),
        Constraint::Length(13),
        Constraint::Length(13),
        Constraint::Length(10),
        Constraint::Length(9),
        Constraint::Min(6),
    ];
    let title = if s.recent.is_empty() {
        " Recent requests (none yet) ".to_string()
    } else {
        " Recent requests, newest first ".to_string()
    };
    f.render_widget(
        Table::new(rows, widths)
            .header(header)
            .block(Block::bordered().title(title)),
        bottom,
    );
}

fn draw_models(f: &mut Frame, area: Rect, app: &App) {
    let [list, detail] = Layout::vertical([
        Constraint::Length(MODELS.len() as u16 + 3),
        Constraint::Min(6),
    ])
    .areas(area);
    let served = app.served_id();
    let header = TRow::new([
        "",
        "Model",
        "Engine",
        "Weights",
        "Context",
        "On this server",
    ])
    .style(Style::new().fg(DIM));
    let rows: Vec<TRow> = MODELS
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let status = if served == Some(m.id) {
                Span::styled("● serving", Style::new().fg(GOOD).bold())
            } else {
                match app.available.get(m.id) {
                    Some(true) => Span::raw("installed"),
                    Some(false) => Span::styled("not installed", Style::new().fg(DIM)),
                    None => Span::styled("?", Style::new().fg(DIM)),
                }
            };
            let ctx = m
                .option("context")
                .map(|o| o.default_choice().label)
                .unwrap_or("-");
            let weights = m.weights.split(',').next().unwrap_or(m.weights);
            let weights = weights.split(" (").next().unwrap_or(weights).to_string();
            let row = TRow::new(vec![
                Cell::from(if i == app.model_cursor { "▸" } else { " " }),
                Cell::from(m.name),
                Cell::from(m.engine),
                Cell::from(weights),
                Cell::from(ctx),
                Cell::from(status),
            ]);
            if i == app.model_cursor {
                row.style(Style::new().add_modifier(Modifier::REVERSED))
            } else {
                row
            }
        })
        .collect();
    let widths = [
        Constraint::Length(1),
        Constraint::Length(20),
        Constraint::Length(9),
        Constraint::Length(34),
        Constraint::Length(8),
        Constraint::Min(10),
    ];
    f.render_widget(
        Table::new(rows, widths)
            .header(header)
            .block(Block::bordered().title(" Choose a model ")),
        list,
    );

    let m = &MODELS[app.model_cursor];
    let saved = app.settings.models.get(m.id).map(|s| s.len()).unwrap_or(0);
    let lines = vec![
        Line::from(Span::styled(m.summary, Style::new().bold())),
        Line::from(""),
        kv("Drafter", vec![Span::raw(m.drafter)]),
        kv("Weights", vec![Span::raw(m.weights)]),
        kv(
            "Memory",
            vec![Span::raw(format!(
                "keep at least {} GiB free while serving",
                m.min_free_gib
            ))],
        ),
        kv("License", vec![Span::raw(m.license)]),
        kv("Fast", vec![Span::raw(m.fast_note)]),
        kv(
            "Preset",
            vec![Span::raw(if saved == 0 {
                "our tuned defaults".to_string()
            } else {
                format!("our tuned defaults + {saved} change(s) you saved")
            })],
        ),
        Line::from(""),
        Line::from(Span::styled(
            "Enter: open its tuning panel. Applying there switches the server to it.",
            Style::new().fg(DIM),
        )),
    ];
    f.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: true })
            .block(Block::bordered().title(format!(" {} ", m.name))),
        detail,
    );
}

fn value_label(app: &App, row: Row) -> (String, bool) {
    let spec = app.spec();
    match row {
        Row::Level => {
            let cur = app.level();
            let parts: Vec<String> = Level::ALL
                .iter()
                .map(|l| {
                    if Some(*l) == cur {
                        format!("[{}]", l.label())
                    } else {
                        l.label().to_string()
                    }
                })
                .collect();
            let text = if cur.is_none() {
                format!("Custom   ({})", parts.join(" · "))
            } else {
                parts.join(" · ")
            };
            (text, cur == Some(Level::Balanced))
        }
        Row::Opt(o) => {
            let v = spec.value(&app.draft, o.id).unwrap_or(o.default);
            let label = o.choice(v).map(|c| c.label).unwrap_or(v).to_string();
            (label, v == o.default)
        }
        Row::Bind => {
            let b = app.bind_choice();
            (b.label().to_string(), b == app.served_bind())
        }
        Row::Key => {
            let has = app.snapshot.as_ref().map(|s| s.api_key).unwrap_or(false);
            let t = match (app.gen_key, has) {
                (true, _) => "Generate a new key",
                (false, true) => "Keep the current key",
                (false, false) => "None",
            };
            (t.to_string(), !app.gen_key)
        }
    }
}

fn row_label(row: Row) -> &'static str {
    match row {
        Row::Level => "Accuracy ↔ Speed",
        Row::Opt(o) => o.label,
        Row::Bind => "Network access",
        Row::Key => "API key",
    }
}

fn draw_tuning(f: &mut Frame, area: Rect, app: &App) {
    let spec = app.spec();
    let rows = app.rows();
    let served = app.served_id() == Some(spec.id);
    let mut lines: Vec<Line> = Vec::new();
    let mut last_group: Option<&str> = None;
    for (i, row) in rows.iter().enumerate() {
        let group = match row {
            Row::Level => "Accuracy",
            Row::Opt(o) => match o.group {
                Group::Common => "Common",
                Group::Accuracy => "Accuracy",
                Group::Advanced => "Advanced",
            },
            Row::Bind | Row::Key => "Server",
        };
        if last_group != Some(group) {
            if last_group.is_some() {
                lines.push(Line::from(""));
            }
            lines.push(Line::from(Span::styled(
                format!(" {group}"),
                Style::new().fg(ACCENT).bold(),
            )));
            last_group = Some(group);
        }
        let (val, is_default) = value_label(app, *row);
        let inactive = matches!(row, Row::Opt(o) if !spec.active(&app.draft, o));
        let selected = i == app.tune_cursor;
        let mark = match (row, is_default) {
            (Row::Bind | Row::Key, true) => "  current",
            (_, true) => "  our tuned default",
            (_, false) => "  changed",
        };
        let mut style = Style::new();
        if inactive {
            style = style.fg(DIM);
        }
        let mut spans = vec![
            Span::raw(if selected { " ▸ " } else { "   " }),
            Span::styled(format!("{:<30}", row_label(*row)), style),
            Span::styled(
                format!("‹ {val} ›"),
                if selected {
                    style.add_modifier(Modifier::REVERSED)
                } else {
                    style.bold()
                },
            ),
        ];
        if inactive {
            spans.push(Span::styled(
                "  (needs speculative decoding)",
                Style::new().fg(DIM),
            ));
        } else if *row != Row::Level || !is_default {
            spans.push(Span::styled(
                mark,
                Style::new().fg(if is_default { DIM } else { WARN }),
            ));
        } else {
            spans.push(Span::styled("  recommended", Style::new().fg(DIM)));
        }
        if let Row::Opt(o) = row {
            if o.choices
                .iter()
                .any(|c| c.value != o.default && app.missing_file(o.id, c.value).is_some())
            {
                spans.push(Span::styled(
                    "  (weights not installed)",
                    Style::new().fg(DIM),
                ));
            }
            if o.quality == Quality::Reduced && !is_default {
                spans.push(Span::styled(
                    "  ! may lower quality",
                    Style::new().fg(WARN).bold(),
                ));
            }
        }
        lines.push(Line::from(spans));
    }
    let [banner, list, detail] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(lines.len() as u16 + 2),
        Constraint::Min(6),
    ])
    .areas(area);

    let banner_line = if app.snapshot.is_some() && !app.control_enabled() {
        Line::from(Span::styled(
            " Read-only: this server does not accept changes (start talus-server with --control-dir).",
            Style::new().fg(WARN),
        ))
    } else if served {
        Line::from(vec![
            Span::raw(" "),
            Span::styled(spec.name, Style::new().bold()),
            Span::styled("  ● serving now", Style::new().fg(GOOD)),
        ])
    } else {
        Line::from(vec![
            Span::raw(" "),
            Span::styled(spec.name, Style::new().bold()),
            Span::styled(
                "  not serving: applying switches the server to it",
                Style::new().fg(WARN),
            ),
        ])
    };
    f.render_widget(Paragraph::new(banner_line), banner);

    f.render_widget(
        Paragraph::new(lines).block(Block::bordered().title(format!(" Tuning: {} ", spec.name))),
        list,
    );

    let row = app.current_row();
    let detail_lines: Vec<Line> = match row {
        Row::Level => vec![
            Line::from(Span::styled(
                "How exact or fast the model runs. Each level sets the Accuracy options below.",
                Style::new().bold(),
            )),
            Line::from(""),
            kv("Reference", vec![Span::raw(spec.reference_note)]),
            kv(
                "Balanced",
                vec![Span::raw("Our release defaults: the recommended level.")],
            ),
            kv(
                "Fast",
                if app.fast_unavailable() {
                    vec![Span::raw(
                        "Its weights file is not installed next to the model, so Fast is the same as Balanced.",
                    )]
                } else {
                    vec![Span::styled(
                        spec.fast_note,
                        Style::new().fg(if spec.has_fast() { WARN } else { Color::Reset }),
                    )]
                },
            ),
        ],
        Row::Opt(o) => {
            let def = o.default_choice().label;
            let mut v = vec![
                Line::from(Span::styled(o.description, Style::new().bold())),
                Line::from(""),
                kv(
                    "Default",
                    vec![Span::raw(format!("{def} (our tuned default)"))],
                ),
                kv("Trade-off", vec![Span::raw(o.tradeoff)]),
                kv(
                    "Output",
                    vec![Span::styled(
                        o.quality.label(),
                        Style::new().fg(match o.quality {
                            Quality::Exact => GOOD,
                            Quality::Gated => Color::Reset,
                            Quality::Reduced => WARN,
                        }),
                    )],
                ),
                kv("Switch", vec![Span::styled(o.switch, Style::new().fg(DIM))]),
            ];
            for c in o.choices.iter().filter(|c| c.value != o.default) {
                if let Some(f) = app.missing_file(o.id, c.value) {
                    v.push(kv(
                        "Weights",
                        vec![Span::styled(
                            format!("not installed: {f}"),
                            Style::new().fg(WARN),
                        )],
                    ));
                }
            }
            if !spec.active(&app.draft, o) {
                v.push(Line::from(Span::styled(
                    "Has no effect while speculative decoding is off.",
                    Style::new().fg(DIM),
                )));
            }
            v
        }
        Row::Bind => vec![
            Line::from(Span::styled(
                "Who can reach the server.",
                Style::new().bold(),
            )),
            Line::from(""),
            kv(
                "Local",
                vec![Span::raw(
                    "127.0.0.1: only programs on this computer (the default).",
                )],
            ),
            kv(
                "Network",
                vec![Span::raw(
                    "0.0.0.0: other computers on your network too. Needs an API key.",
                )],
            ),
        ],
        Row::Key => vec![
            Line::from(Span::styled(
                "The password clients send with each request.",
                Style::new().bold(),
            )),
            Line::from(""),
            kv(
                "Generate",
                vec![Span::raw(
                    "Makes a new random key, shows it once and saves it for this TUI.",
                )],
            ),
            kv(
                "Note",
                vec![Span::raw(
                    "A key set with STRATA_API_KEY in the service environment wins.",
                )],
            ),
        ],
    };
    f.render_widget(
        Paragraph::new(detail_lines)
            .wrap(Wrap { trim: true })
            .block(Block::bordered().title(format!(" {} ", row_label(row)))),
        detail,
    );
}

/// Render the current state as plain text (for docs screenshots and tests).
pub fn render_to_string(app: &App, url: &str, width: u16, height: u16) -> String {
    let backend = ratatui::backend::TestBackend::new(width, height);
    let mut term = ratatui::Terminal::new(backend).expect("test backend");
    term.draw(|f| draw(f, app, url)).expect("draw");
    let buf = term.backend().buffer();
    let mut out = String::new();
    for y in 0..height {
        let mut line = String::new();
        for x in 0..width {
            line.push_str(buf[(x, y)].symbol());
        }
        out.push_str(line.trim_end());
        out.push('\n');
    }
    out
}
