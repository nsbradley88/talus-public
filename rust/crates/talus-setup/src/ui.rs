//! Terminal output and questions, in talus-tui's colors (cyan accent, green / yellow / red verdicts).
//!
//! Questions read from /dev/tty, not stdin: `curl ... | bash` hands the script to bash on stdin. Enter takes the
//! default shown in brackets. With `--yes` nothing is asked and every default is taken.

use std::io::{BufRead, BufReader, IsTerminal, Write};

fn color() -> bool {
    std::env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal()
}

fn paint(code: &str, s: &str) -> String {
    if color() {
        format!("\x1b[{code}m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}

pub fn accent(s: &str) -> String {
    paint("36", s)
}
pub fn good(s: &str) -> String {
    paint("32", s)
}
pub fn warn_c(s: &str) -> String {
    paint("33", s)
}
pub fn bad(s: &str) -> String {
    paint("31", s)
}
pub fn dim(s: &str) -> String {
    paint("90", s)
}
pub fn bold(s: &str) -> String {
    paint("1", s)
}

/// The header line, like the TUI's (" Talus " in black on cyan).
pub fn header(title: &str) {
    println!();
    println!("{} {}", paint("30;46;1", " Talus "), bold(title));
}

/// A numbered step heading.
pub fn step(n: usize, total: usize, title: &str) {
    println!();
    println!("{} {}", accent(&format!("[{n}/{total}]")), bold(title));
}

pub fn say(s: &str) {
    println!("  {s}");
}
pub fn note(s: &str) {
    println!("  {}", dim(s));
}
pub fn warn(s: &str) {
    println!("  {} {s}", warn_c("!"));
}
pub fn error(s: &str) {
    eprintln!("  {} {s}", bad("error:"));
}
pub fn dry(s: &str) {
    println!("  {} {s}", accent("[dry-run] would"));
}

/// A block of text to copy, indented and set apart.
pub fn block(text: &str) {
    for l in text.lines() {
        println!("      {l}");
    }
}

pub struct Prompter {
    pub yes: bool,
    tty: Option<BufReader<std::fs::File>>,
}

impl Prompter {
    pub fn new(yes: bool) -> Prompter {
        let tty = if yes {
            None
        } else {
            std::fs::File::open("/dev/tty").ok().map(BufReader::new)
        };
        Prompter { yes, tty }
    }

    pub fn interactive(&self) -> bool {
        !self.yes && self.tty.is_some()
    }

    fn read_line(&mut self) -> Option<String> {
        let tty = self.tty.as_mut()?;
        let mut s = String::new();
        match tty.read_line(&mut s) {
            Ok(0) | Err(_) => None,
            Ok(_) => Some(s.trim().to_string()),
        }
    }

    /// A free-form answer; Enter (or --yes, or no terminal) takes `default`.
    pub fn ask(&mut self, q: &str, default: &str) -> String {
        if !self.interactive() {
            println!("  {q} [{default}] {}", dim("(default)"));
            return default.to_string();
        }
        print!("  {q} [{}] ", accent(default));
        let _ = std::io::stdout().flush();
        match self.read_line() {
            Some(s) if !s.is_empty() => s,
            _ => default.to_string(),
        }
    }

    /// Yes / no with a default.
    pub fn confirm(&mut self, q: &str, default_yes: bool) -> bool {
        let hint = if default_yes { "Y/n" } else { "y/N" };
        if !self.interactive() {
            println!(
                "  {q} [{hint}] {}",
                dim(if default_yes {
                    "yes (default)"
                } else {
                    "no (default)"
                })
            );
            return default_yes;
        }
        loop {
            print!("  {q} [{}] ", accent(hint));
            let _ = std::io::stdout().flush();
            match self
                .read_line()
                .as_deref()
                .map(str::to_ascii_lowercase)
                .as_deref()
            {
                None | Some("") => return default_yes,
                Some("y") | Some("yes") => return true,
                Some("n") | Some("no") => return false,
                _ => println!("  {}", dim("please answer y or n")),
            }
        }
    }

    /// A change that needs the word typed out. Never true without a terminal unless `preapproved`
    /// (`--allow-system-changes` with `--yes`).
    pub fn confirm_typed(&mut self, q: &str, word: &str, preapproved: bool) -> bool {
        if self.yes || self.tty.is_none() {
            if preapproved {
                println!("  {q} {}", dim("(approved by --allow-system-changes)"));
            } else {
                println!(
                    "  {q} {}",
                    dim("(skipped: needs a person to type yes, or --allow-system-changes)")
                );
            }
            return preapproved;
        }
        print!(
            "  {q} Type {} to go ahead, anything else skips: ",
            bold(word)
        );
        let _ = std::io::stdout().flush();
        self.read_line().as_deref() == Some(word)
    }
}

/// A verdict tag, fixed width.
pub fn tag(v: crate::checks::Verdict) -> String {
    use crate::checks::Verdict::*;
    match v {
        Pass => good("[ ok ]"),
        Warn => warn_c("[warn]"),
        Fail => bad("[FAIL]"),
        Info => accent("[info]"),
        Skip => dim("[ -- ]"),
    }
}
