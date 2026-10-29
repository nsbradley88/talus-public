//! TUI state and key handling. No terminal or network here: the runner feeds in status replies and key presses,
//! and carries out the `Action`s this returns, so all of it is unit-testable.

use crate::settings::Settings;
use crate::stats::Snapshot;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;
use talus_presets::{Bind, Group, Level, ModelSpec, Quality, Selection, TuneOption, MODELS};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Screen {
    Status,
    Models,
    Tuning,
}

impl Screen {
    pub const ALL: [Screen; 3] = [Screen::Status, Screen::Models, Screen::Tuning];

    pub fn title(self) -> &'static str {
        match self {
            Screen::Status => "Status",
            Screen::Models => "Models",
            Screen::Tuning => "Tuning",
        }
    }
}

/// One line of the tuning panel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Row {
    Level,
    Opt(&'static TuneOption),
    Bind,
    Key,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Modal {
    None,
    Confirm {
        title: String,
        lines: Vec<String>,
        warning: Option<String>,
        body: Value,
        new_key: Option<String>,
    },
    Message {
        title: String,
        lines: Vec<String>,
    },
    Help,
}

/// What the runner should do after a key press.
#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    None,
    Quit,
    Refresh,
    /// POST this body to `/v1/talus/apply`; pass the reply to `App::on_apply_result`.
    Apply {
        body: Value,
        new_key: Option<String>,
    },
}

pub struct App {
    pub screen: Screen,
    pub settings: Settings,
    pub settings_path: Option<PathBuf>,
    pub key_path: Option<PathBuf>,
    pub snapshot: Option<Snapshot>,
    pub last_error: Option<String>,
    /// Model id -> a run config exists in the server's control directory.
    pub available: BTreeMap<String, bool>,
    /// Model id -> choices (option, value, file) whose weights file is not installed next to the model's GGUF.
    pub missing: BTreeMap<String, Vec<(String, String, String)>>,
    pub model_cursor: usize,
    /// Index into `MODELS` of the model being tuned.
    pub tune_model: usize,
    /// Set once the user picked a model, so status updates stop following the served model.
    pub tune_chosen: bool,
    pub draft: Selection,
    pub bind: Option<Bind>,
    pub gen_key: bool,
    pub tune_cursor: usize,
    pub modal: Modal,
    pub notice: Option<String>,
    pub restarting_since: Option<Instant>,
    /// Restart reported by apply; cleared when the server answers again.
    pub restart_model: Option<String>,
    /// The server was seen restarting or down since the apply.
    restart_seen_down: bool,
}

impl App {
    pub fn new(settings: Settings) -> Self {
        Self {
            screen: Screen::Status,
            settings,
            settings_path: None,
            key_path: None,
            snapshot: None,
            last_error: None,
            available: BTreeMap::new(),
            missing: BTreeMap::new(),
            model_cursor: 0,
            tune_model: 0,
            tune_chosen: false,
            draft: Selection::new(),
            bind: None,
            gen_key: false,
            tune_cursor: 0,
            modal: Modal::None,
            notice: None,
            restarting_since: None,
            restart_model: None,
            restart_seen_down: false,
        }
    }

    pub fn spec(&self) -> &'static ModelSpec {
        &MODELS[self.tune_model]
    }

    pub fn served_id(&self) -> Option<&str> {
        self.snapshot.as_ref().and_then(|s| s.preset.as_deref())
    }

    pub fn control_enabled(&self) -> bool {
        self.snapshot
            .as_ref()
            .map(|s| s.control_enabled)
            .unwrap_or(false)
    }

    pub fn served_bind(&self) -> Bind {
        self.snapshot
            .as_ref()
            .map(|s| Bind::from_host(&s.host))
            .unwrap_or(Bind::Local)
    }

    pub fn bind_choice(&self) -> Bind {
        self.bind.unwrap_or_else(|| self.served_bind())
    }

    /// The choices in effect on the server for `id` (empty = tuned defaults), when `id` is being served.
    pub fn applied_for(&self, id: &str) -> Option<Selection> {
        let snap = self.snapshot.as_ref()?;
        if snap.preset.as_deref() != Some(id) {
            return None;
        }
        Some(
            snap.selection
                .as_ref()
                .filter(|s| s.model == id)
                .map(|s| s.choices.clone())
                .unwrap_or_default(),
        )
    }

    fn load_draft(&mut self) {
        let id = self.spec().id;
        self.draft = self
            .applied_for(id)
            .or_else(|| self.settings.models.get(id).cloned())
            .unwrap_or_default();
        self.bind = None;
        self.gen_key = false;
        self.tune_cursor = 0;
    }

    pub fn select_model(&mut self, idx: usize) {
        self.tune_model = idx.min(MODELS.len() - 1);
        self.tune_chosen = true;
        self.load_draft();
        self.screen = Screen::Tuning;
    }

    pub fn on_status(&mut self, res: Result<Snapshot, String>) {
        match res {
            Ok(snap) => {
                let first = self.snapshot.is_none();
                if self.restart_model.is_some() && snap.state == "restarting" {
                    self.restart_seen_down = true;
                }
                let back = self.restart_model.is_some()
                    && self.restart_seen_down
                    && snap.state != "restarting"
                    && snap.loaded;
                self.snapshot = Some(snap);
                self.last_error = None;
                if back {
                    let name = self.restart_model.take().unwrap_or_default();
                    self.restarting_since = None;
                    match self.snapshot.as_ref().and_then(|s| s.last_error.clone()) {
                        Some(err) => self.message(
                            "Not applied",
                            vec![
                                err,
                                String::new(),
                                "The server went back to its previous settings.".into(),
                            ],
                        ),
                        None => self.notice = Some(format!("Serving {name} again.")),
                    }
                }
                if (first || back) && !self.tune_chosen {
                    if let Some(i) = self
                        .served_id()
                        .and_then(|id| MODELS.iter().position(|m| m.id == id))
                    {
                        self.tune_model = i;
                        self.model_cursor = i;
                    }
                    self.load_draft();
                }
            }
            Err(e) => {
                if self.restart_model.is_some() {
                    self.restart_seen_down = true;
                }
                self.last_error = Some(e);
            }
        }
    }

    pub fn on_presets(&mut self, v: &Value) {
        self.available = v
            .get("models")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|m| {
                        Some((
                            m.get("id")?.as_str()?.to_string(),
                            m.get("available")?.as_bool()?,
                        ))
                    })
                    .collect()
            })
            .unwrap_or_default();
        self.missing = v
            .get("models")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|m| {
                        let id = m.get("id")?.as_str()?.to_string();
                        let list = m
                            .get("missing")?
                            .as_array()?
                            .iter()
                            .filter_map(|x| {
                                Some((
                                    x.get("option")?.as_str()?.to_string(),
                                    x.get("value")?.as_str()?.to_string(),
                                    x.get("file")?.as_str()?.to_string(),
                                ))
                            })
                            .collect();
                        Some((id, list))
                    })
                    .collect()
            })
            .unwrap_or_default();
    }

    /// The weights file `option = value` of the tuned model needs and the server reported missing.
    pub fn missing_file(&self, option: &str, value: &str) -> Option<&str> {
        self.missing
            .get(self.spec().id)?
            .iter()
            .find(|(o, v, _)| o == option && v == value)
            .map(|(_, _, f)| f.as_str())
    }

    /// The level's Fast choices cannot be applied: every one of them needs a missing weights file.
    pub fn fast_unavailable(&self) -> bool {
        let spec = self.spec();
        spec.has_fast()
            && spec
                .fast
                .iter()
                .all(|(o, v)| self.missing_file(o, v).is_some())
    }

    /// Drop non-default choices whose weights file is missing (they would only fall back in the engine).
    fn drop_missing(&mut self) {
        let spec = self.spec();
        let drop: Vec<&'static str> = spec
            .options
            .iter()
            .filter(|o| {
                let v = spec.value(&self.draft, o.id).unwrap_or(o.default);
                v != o.default && self.missing_file(o.id, v).is_some()
            })
            .map(|o| o.id)
            .collect();
        for id in drop {
            self.draft.remove(id);
        }
    }

    pub fn rows(&self) -> Vec<Row> {
        let spec = self.spec();
        let mut rows = vec![Row::Level];
        for g in [Group::Accuracy, Group::Common, Group::Advanced] {
            rows.extend(spec.options.iter().filter(|o| o.group == g).map(Row::Opt));
        }
        rows.push(Row::Bind);
        rows.push(Row::Key);
        rows
    }

    pub fn current_row(&self) -> Row {
        let rows = self.rows();
        rows[self.tune_cursor.min(rows.len() - 1)]
    }

    pub fn level(&self) -> Option<Level> {
        self.spec().level_of(&self.draft)
    }

    /// Options the draft turns on that may lower output quality.
    pub fn reduced_on(&self) -> Vec<&'static str> {
        let spec = self.spec();
        spec.options
            .iter()
            .filter(|o| o.quality == Quality::Reduced)
            .filter(|o| spec.value(&self.draft, o.id) != Some(o.default))
            .map(|o| o.label)
            .collect()
    }

    fn change(&mut self, dir: i32) {
        let spec = self.spec();
        match self.current_row() {
            Row::Level => {
                let cur = self.level().unwrap_or(Level::Balanced);
                let i = Level::ALL.iter().position(|&l| l == cur).unwrap_or(1) as i32;
                let n = (i + dir).clamp(0, 2) as usize;
                self.draft = spec.apply_level(&self.draft, Level::ALL[n]);
                if Level::ALL[n] == Level::Fast && self.fast_unavailable() {
                    self.notice = Some(
                        "Fast needs weights that are not installed: it stays Balanced.".to_string(),
                    );
                }
                self.drop_missing();
            }
            Row::Opt(o) => {
                let cur = spec.value(&self.draft, o.id).unwrap_or(o.default);
                let i = o.choice_index(cur) as i32;
                let len = o.choices.len() as i32;
                // skip choices whose weights file is not installed
                let mut n = i;
                for _ in 0..len {
                    n = ((n + dir) % len + len) % len;
                    let v = o.choices[n as usize].value;
                    if v == o.default || self.missing_file(o.id, v).is_none() {
                        break;
                    }
                    self.notice = Some(format!(
                        "{}: weights not installed ({})",
                        o.label,
                        self.missing_file(o.id, v).unwrap_or("")
                    ));
                }
                self.draft
                    .insert(o.id.to_string(), o.choices[n as usize].value.to_string());
                self.draft = spec.normalize(&self.draft);
            }
            Row::Bind => {
                self.bind = Some(match self.bind_choice() {
                    Bind::Local => Bind::Lan,
                    Bind::Lan => Bind::Local,
                });
            }
            Row::Key => self.gen_key = !self.gen_key,
        }
    }

    fn reset_row(&mut self) {
        match self.current_row() {
            Row::Level => self.draft = self.spec().apply_level(&self.draft, Level::Balanced),
            Row::Opt(o) => {
                self.draft.remove(o.id);
            }
            Row::Bind => self.bind = None,
            Row::Key => self.gen_key = false,
        }
    }

    pub fn reset_all(&mut self) {
        self.draft.clear();
        self.notice = Some(format!(
            "{}: back to our tuned defaults (press a to apply).",
            self.spec().name
        ));
    }

    /// Build the confirmation for applying the draft, or a message saying why not.
    pub fn prepare_apply(&mut self) {
        let spec = self.spec();
        let Some(snap) = self.snapshot.as_ref() else {
            self.message(
                "Not connected",
                vec!["The server is not answering yet.".into()],
            );
            return;
        };
        if !snap.control_enabled {
            self.message(
                "Read-only",
                vec![
                    "This server does not accept changes.".into(),
                    "Start talus-server with --control-dir DIR to switch models and apply tuning."
                        .into(),
                    "See docs/TUI.md.".into(),
                ],
            );
            return;
        }
        if self.available.get(spec.id) == Some(&false) {
            self.message(
                "Model not installed",
                vec![format!(
                    "The server has no run config for {}: copy one to <control-dir>/{}.json.",
                    spec.name, spec.id
                )],
            );
            return;
        }
        let switching = snap.preset.as_deref() != Some(spec.id);
        let from = self.applied_for(spec.id).unwrap_or_default();
        let mut lines = Vec::new();
        if switching {
            let cur = snap
                .preset
                .as_deref()
                .and_then(talus_presets::model)
                .map(|m| m.name.to_string())
                .unwrap_or_else(|| snap.model.clone());
            lines.push(format!("Switch model: {cur} -> {}", spec.name));
        }
        lines.extend(spec.diff(&from, &self.draft));
        let bind = self.bind_choice();
        let bind_changed = bind != self.served_bind();
        if bind_changed {
            lines.push(format!("Network access: {}", bind.label()));
        }
        if self.gen_key {
            lines.push("API key: generate a new one".into());
        }
        if lines.is_empty() {
            self.message(
                "Nothing to apply",
                vec![format!("{} already runs with these settings.", spec.name)],
            );
            return;
        }
        if bind == Bind::Lan && !snap.api_key && !self.gen_key {
            self.message(
                "API key needed",
                vec![
                    "Local network access needs an API key.".into(),
                    "Set \"API key\" to \"Generate new\" first.".into(),
                ],
            );
            return;
        }
        if self.gen_key && snap.api_key_from_env {
            self.message(
                "API key set by the service",
                vec!["The key comes from TALUS_API_KEY (or STRATA_API_KEY) in the service environment; change it there.".into()],
            );
            return;
        }
        let new_key = if self.gen_key {
            match crate::settings::generate_key() {
                Ok(k) => Some(k),
                Err(e) => {
                    self.message("Cannot make a key", vec![e.to_string()]);
                    return;
                }
            }
        } else {
            None
        };
        lines.push(String::new());
        lines.push(if switching {
            format!(
                "Serving stops while {} loads ({}); this can take a few minutes.",
                spec.name, spec.weights
            )
        } else {
            "Serving pauses briefly while the engine restarts.".into()
        });
        let reduced = self.reduced_on();
        let warning = (!reduced.is_empty()).then(|| {
            format!(
                "Output quality may drop slightly: {} did not pass our quality gate.",
                reduced.join(", ")
            )
        });
        let choices: Map<String, Value> = self
            .draft
            .iter()
            .map(|(k, v)| (k.clone(), Value::String(v.clone())))
            .collect();
        let mut body = json!({"model": spec.id, "choices": choices});
        if bind_changed {
            body["bind"] = Value::String(bind.id().into());
        }
        if let Some(k) = &new_key {
            body["api_key"] = Value::String(k.clone());
        }
        self.modal = Modal::Confirm {
            title: format!("Apply to {}?", spec.name),
            lines,
            warning,
            body,
            new_key,
        };
    }

    fn message(&mut self, title: &str, lines: Vec<String>) {
        self.modal = Modal::Message {
            title: title.to_string(),
            lines,
        };
    }

    /// The server's reply to an apply. Returns the new API key to use from now on, if one was set.
    pub fn on_apply_result(
        &mut self,
        res: Result<(u16, Value), String>,
        new_key: Option<String>,
    ) -> Option<String> {
        match res {
            Ok((202, _)) => {
                let spec = self.spec();
                self.settings
                    .models
                    .insert(spec.id.to_string(), spec.normalize(&self.draft));
                if let Some(p) = &self.settings_path {
                    if let Err(e) = self.settings.save(p) {
                        self.notice = Some(format!("Could not save settings: {e}"));
                    }
                }
                self.restarting_since = Some(Instant::now());
                self.restart_model = Some(spec.name.to_string());
                self.restart_seen_down = false;
                self.tune_chosen = false;
                self.screen = Screen::Status;
                self.bind = None;
                self.gen_key = false;
                if let Some(k) = &new_key {
                    let saved = match &self.key_path {
                        Some(p) => match crate::settings::write_key_file(p, k) {
                            Ok(()) => format!("Saved to {} (readable by you only).", p.display()),
                            Err(e) => format!("Could not save it ({e}): copy it now."),
                        },
                        None => "Copy it now; it is not shown again.".to_string(),
                    };
                    self.message(
                        "New API key",
                        vec![
                            k.clone(),
                            String::new(),
                            saved,
                            "Clients send it as the header  Authorization: Bearer <key>".into(),
                        ],
                    );
                } else {
                    self.notice = Some(format!("Restarting with {}...", spec.name));
                }
                new_key
            }
            Ok((code, v)) => {
                self.message(
                    "Not applied",
                    vec![format!(
                        "The server said ({code}): {}",
                        crate::client::error_message(&v)
                    )],
                );
                None
            }
            Err(e) => {
                self.message("Not applied", vec![e]);
                None
            }
        }
    }

    pub fn on_key(&mut self, k: KeyEvent) -> Action {
        if k.modifiers.contains(KeyModifiers::CONTROL) && k.code == KeyCode::Char('c') {
            return Action::Quit;
        }
        match std::mem::replace(&mut self.modal, Modal::None) {
            Modal::Confirm {
                title,
                lines,
                warning,
                body,
                new_key,
            } => {
                return match k.code {
                    KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter => {
                        Action::Apply { body, new_key }
                    }
                    KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => Action::None,
                    _ => {
                        self.modal = Modal::Confirm {
                            title,
                            lines,
                            warning,
                            body,
                            new_key,
                        };
                        Action::None
                    }
                };
            }
            Modal::Message { .. } | Modal::Help => return Action::None,
            Modal::None => {}
        }
        self.notice = None;
        match k.code {
            KeyCode::Char('q') | KeyCode::Esc => return Action::Quit,
            KeyCode::Char('?') => {
                self.modal = Modal::Help;
                return Action::None;
            }
            KeyCode::Char('1') => self.screen = Screen::Status,
            KeyCode::Char('2') => self.screen = Screen::Models,
            KeyCode::Char('3') => self.screen = Screen::Tuning,
            KeyCode::Tab => self.screen = next_screen(self.screen, 1),
            KeyCode::BackTab => self.screen = next_screen(self.screen, -1),
            KeyCode::F(5) | KeyCode::Char('g') => return Action::Refresh,
            _ => match self.screen {
                Screen::Status => {}
                Screen::Models => self.models_key(k.code),
                Screen::Tuning => self.tuning_key(k.code),
            },
        }
        Action::None
    }

    fn models_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.model_cursor = self.model_cursor.saturating_sub(1)
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.model_cursor = (self.model_cursor + 1).min(MODELS.len() - 1)
            }
            KeyCode::Enter | KeyCode::Char(' ') => self.select_model(self.model_cursor),
            _ => {}
        }
    }

    fn tuning_key(&mut self, code: KeyCode) {
        let n = self.rows().len();
        match code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.tune_cursor = self.tune_cursor.saturating_sub(1)
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.tune_cursor = (self.tune_cursor + 1).min(n - 1)
            }
            KeyCode::Left | KeyCode::Char('h') => self.change(-1),
            KeyCode::Right | KeyCode::Char('l') | KeyCode::Enter | KeyCode::Char(' ') => {
                self.change(1)
            }
            KeyCode::Char('d') => self.reset_row(),
            KeyCode::Char('r') => self.reset_all(),
            KeyCode::Char('a') => self.prepare_apply(),
            KeyCode::Char('m') => self.screen = Screen::Models,
            _ => {}
        }
    }
}

fn next_screen(s: Screen, d: i32) -> Screen {
    let i = Screen::ALL.iter().position(|&x| x == s).unwrap_or(0) as i32;
    Screen::ALL[((i + d).rem_euclid(3)) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::{AppliedSelection, Snapshot};
    use crossterm::event::KeyEventKind;

    fn key(c: KeyCode) -> KeyEvent {
        KeyEvent {
            code: c,
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Press,
            state: crossterm::event::KeyEventState::NONE,
        }
    }

    fn snap(preset: &str, control: bool) -> Snapshot {
        Snapshot {
            model: preset.into(),
            preset: Some(preset.into()),
            state: "idle".into(),
            loaded: true,
            host: "127.0.0.1".into(),
            port: 8000,
            control_enabled: control,
            ..Default::default()
        }
    }

    fn app_on(preset: &str) -> App {
        let mut a = App::new(Settings::default());
        a.on_status(Ok(snap(preset, true)));
        a
    }

    fn row_of(a: &App, id: &str) -> usize {
        a.rows()
            .iter()
            .position(|r| matches!(r, Row::Opt(o) if o.id == id))
            .unwrap()
    }

    #[test]
    fn follows_served_model_until_chosen() {
        let a = app_on("laguna-s-2.1");
        assert_eq!(a.spec().id, "laguna-s-2.1");
        assert_eq!(a.level(), Some(Level::Balanced));
        let mut a = a;
        a.screen = Screen::Models;
        a.on_key(key(KeyCode::Up));
        a.on_key(key(KeyCode::Enter));
        assert_eq!(a.screen, Screen::Tuning);
        assert_eq!(a.spec().id, "mimo-v2.6-flash");
        a.on_status(Ok(snap("laguna-s-2.1", true)));
        assert_eq!(a.spec().id, "mimo-v2.6-flash", "a chosen model stays");
    }

    #[test]
    fn draft_starts_from_applied_selection() {
        let mut s = snap("glm-5.3-flash", true);
        s.selection = Some(AppliedSelection {
            model: "glm-5.3-flash".into(),
            level: "custom".into(),
            choices: [("context".to_string(), "32768".to_string())].into(),
            bind: "local".into(),
        });
        let mut a = App::new(Settings::default());
        a.on_status(Ok(s));
        assert_eq!(a.draft.get("context").map(String::as_str), Some("32768"));
        a.screen = Screen::Tuning;
        a.on_key(key(KeyCode::Char('a')));
        assert!(matches!(&a.modal, Modal::Message { title, .. } if title == "Nothing to apply"));
    }

    #[test]
    fn level_control_and_fast_warning() {
        let mut a = app_on("mimo-v2.6-flash");
        a.screen = Screen::Tuning;
        assert_eq!(a.current_row(), Row::Level);
        a.on_key(key(KeyCode::Right));
        assert_eq!(a.level(), Some(Level::Fast));
        assert_eq!(a.reduced_on(), vec!["Low-bit expert prompt kernel"]);
        a.on_key(key(KeyCode::Left));
        a.on_key(key(KeyCode::Left));
        assert_eq!(a.level(), Some(Level::Reference));
        a.on_key(key(KeyCode::Right));
        a.on_key(key(KeyCode::Right));
        a.on_key(key(KeyCode::Char('a')));
        match &a.modal {
            Modal::Confirm {
                lines,
                warning,
                body,
                ..
            } => {
                assert!(
                    lines[0].contains("Low-bit expert prompt kernel: Off -> On"),
                    "{lines:?}"
                );
                assert!(warning.as_deref().unwrap().contains("quality may drop"));
                assert_eq!(body["choices"]["iq1m"], "on");
                assert!(body.get("bind").is_none());
            }
            m => panic!("{m:?}"),
        }
        let act = a.on_key(key(KeyCode::Char('y')));
        assert!(matches!(act, Action::Apply { .. }));
        assert_eq!(a.modal, Modal::None);
    }

    #[test]
    fn model_without_fast_stays_balanced() {
        let mut a = app_on("glm-5.3-flash");
        a.screen = Screen::Tuning;
        a.on_key(key(KeyCode::Right));
        assert_eq!(a.level(), Some(Level::Balanced));
        assert!(a.draft.is_empty());
    }

    #[test]
    fn fast_weights_not_installed_stays_balanced() {
        let mut a = app_on("qwen3.8-27b");
        a.on_presets(
            &json!({"models": [{"id": "qwen3.8-27b", "available": true, "missing": [
            {"option": "w4_fast", "value": "on", "file": "/m/Qwen3.8-27B-UD-IQ4_XS.fast.w4i"}]}]}),
        );
        a.screen = Screen::Tuning;
        a.on_key(key(KeyCode::Right)); // Balanced -> Fast
        assert!(a.fast_unavailable());
        assert_eq!(a.level(), Some(Level::Balanced));
        assert!(!a.draft.contains_key("w4_fast"));
        // with the file installed, Fast turns the all-layer profile on
        let mut b = app_on("qwen3.8-27b");
        b.on_presets(&json!({"models": [{"id": "qwen3.8-27b", "available": true, "missing": []}]}));
        b.screen = Screen::Tuning;
        b.on_key(key(KeyCode::Right));
        assert_eq!(b.level(), Some(Level::Fast));
        assert_eq!(b.draft.get("w4_fast").map(String::as_str), Some("on"));
    }

    #[test]
    fn options_cycle_reset_and_apply_flow() {
        let mut a = app_on("qwen3.8-27b");
        a.screen = Screen::Tuning;
        let r = row_of(&a, "context");
        for _ in 0..r {
            a.on_key(key(KeyCode::Down));
        }
        a.on_key(key(KeyCode::Left)); // 64K -> 32K
        assert_eq!(a.draft.get("context").map(String::as_str), Some("32768"));
        a.on_key(key(KeyCode::Char('d')));
        assert!(a.draft.is_empty());
        a.on_key(key(KeyCode::Left));
        a.on_key(key(KeyCode::Char('a')));
        let Modal::Confirm { body, lines, .. } = a.modal.clone() else {
            panic!("{:?}", a.modal)
        };
        assert_eq!(lines[0], "Context length: 64K -> 32K");
        assert!(lines.last().unwrap().contains("pauses briefly"));
        a.modal = Modal::None;
        let k = a.on_apply_result(Ok((202, json!({"status": "restarting"}))), None);
        assert!(k.is_none());
        assert_eq!(a.screen, Screen::Status);
        assert_eq!(a.settings.models["qwen3.8-27b"], body_choices(&body));
        // the server comes back
        let mut s = snap("qwen3.8-27b", true);
        s.state = "restarting".into();
        a.on_status(Ok(s));
        assert!(a.restart_model.is_some());
        a.on_status(Ok(snap("qwen3.8-27b", true)));
        assert!(a.restart_model.is_none());
        assert!(a.notice.as_deref().unwrap().contains("again"));
        a.screen = Screen::Tuning;
        a.on_key(key(KeyCode::Char('r')));
        assert!(a.draft.is_empty());
    }

    fn body_choices(b: &Value) -> Selection {
        b["choices"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
            .collect()
    }

    #[test]
    fn switching_and_guards() {
        let mut a = app_on("qwen3.8-27b");
        a.on_presets(&json!({"models": [{"id": "glm-5.3-flash", "available": false}, {"id": "laguna-s-2.1", "available": true}]}));
        a.select_model(2); // GLM, not installed
        a.on_key(key(KeyCode::Char('a')));
        assert!(matches!(&a.modal, Modal::Message { title, .. } if title == "Model not installed"));
        a.modal = Modal::None;
        a.select_model(4); // Laguna
        a.on_key(key(KeyCode::Char('a')));
        match &a.modal {
            Modal::Confirm { lines, .. } => {
                assert_eq!(lines[0], "Switch model: Qwen3.8-27B -> Laguna-S-2.1");
                assert!(lines.last().unwrap().contains("loads"));
            }
            m => panic!("{m:?}"),
        }
        // LAN needs a key
        a.modal = Modal::None;
        let r = a.rows().iter().position(|r| *r == Row::Bind).unwrap();
        a.tune_cursor = r;
        a.on_key(key(KeyCode::Right));
        assert_eq!(a.bind_choice(), Bind::Lan);
        a.on_key(key(KeyCode::Char('a')));
        assert!(matches!(&a.modal, Modal::Message { title, .. } if title == "API key needed"));
        a.modal = Modal::None;
        a.tune_cursor = r + 1;
        a.on_key(key(KeyCode::Right));
        a.on_key(key(KeyCode::Char('a')));
        let Modal::Confirm { body, new_key, .. } = a.modal.clone() else {
            panic!("{:?}", a.modal)
        };
        assert_eq!(body["bind"], "lan");
        assert_eq!(body["api_key"].as_str(), new_key.as_deref());
        // read-only server
        let mut b = App::new(Settings::default());
        b.on_status(Ok(snap("qwen3.8-27b", false)));
        b.select_model(1);
        b.on_key(key(KeyCode::Char('a')));
        assert!(matches!(&b.modal, Modal::Message { title, .. } if title == "Read-only"));
    }

    #[test]
    fn failed_apply_is_reported() {
        let mut a = app_on("qwen3.8-27b");
        a.select_model(3);
        a.on_apply_result(Ok((202, json!({"status": "restarting"}))), None);
        a.on_status(Err("connection refused".into()));
        let mut s = snap("qwen3.8-27b", true);
        s.last_error = Some("MiMo-V2.6-Flash did not start: out of memory".into());
        a.on_status(Ok(s));
        match &a.modal {
            Modal::Message { title, lines } => {
                assert_eq!(title, "Not applied");
                assert!(lines[0].contains("did not start"));
            }
            m => panic!("{m:?}"),
        }
    }

    #[test]
    fn errors_and_quit() {
        let mut a = app_on("qwen3.8-27b");
        a.on_apply_result(Ok((409, json!({"error": {"message": "busy"}}))), None);
        assert!(matches!(&a.modal, Modal::Message { lines, .. } if lines[0].contains("busy")));
        assert_eq!(a.on_key(key(KeyCode::Char('x'))), Action::None);
        assert_eq!(a.modal, Modal::None);
        a.on_status(Err("cannot reach".into()));
        assert!(a.last_error.is_some());
        assert!(a.snapshot.is_some(), "the last snapshot stays on screen");
        assert_eq!(a.on_key(key(KeyCode::Char('q'))), Action::Quit);
        assert_eq!(a.on_key(key(KeyCode::Tab)), Action::None);
        assert_eq!(a.screen, Screen::Models);
    }
}
