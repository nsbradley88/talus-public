//! Follows Markdown code in model text, to tell a tool call from a quoted one: code fences (``` or ~~~), the current line,
//! and the backticks of the current paragraph. Same rules as `talus_text::parser::OutputParser::track`, in constant space.

/// Where the text pushed so far ends. `Copy`, so a decision can look ahead on a copy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CodeSpans {
    /// The open fence's character, 0 for none.
    fence: u8,
    /// Backticks on the finished lines of this paragraph.
    ticks: usize,
    line_ticks: usize,
    /// The line has something other than whitespace.
    nonblank: bool,
    /// The line's first three characters after its indent: how many so far, and the fence character they all are (0 if not).
    head_len: u8,
    head_char: u8,
    /// After those three: the current run of that character, and whether a run of three came up.
    run: u8,
    triple: bool,
}

impl CodeSpans {
    pub fn push(&mut self, text: &str) {
        for c in text.chars() {
            if c == '\n' {
                self.end_line();
                continue;
            }
            if c == '`' {
                self.line_ticks += 1;
            }
            if !self.nonblank && c.is_whitespace() {
                continue;
            }
            self.nonblank = true;
            let fence_char = u8::try_from(c).ok().filter(|b| *b == b'`' || *b == b'~');
            if self.head_len < 3 {
                self.head_char = match (self.head_len, fence_char) {
                    (0, Some(b)) => b,
                    (_, Some(b)) if b == self.head_char => b,
                    _ => 0,
                };
                self.head_len += 1;
            } else if self.head_char != 0 && fence_char == Some(self.head_char) {
                self.run += 1;
                self.triple |= self.run >= 3;
            } else {
                self.run = 0;
            }
        }
    }

    fn end_line(&mut self) {
        let fence_head = self.head_len == 3 && self.head_char != 0;
        if self.fence != 0 {
            if fence_head && self.head_char == self.fence {
                self.fence = 0;
            }
        } else if fence_head && !self.triple {
            self.fence = self.head_char;
        } else if !self.nonblank {
            self.ticks = 0;
        } else {
            self.ticks += self.line_ticks;
        }
        *self = Self {
            fence: self.fence,
            ticks: self.ticks,
            ..Self::default()
        };
    }

    /// The next character is inside a code fence or inline code.
    pub fn in_code(&self) -> bool {
        self.fence != 0 || !(self.ticks + self.line_ticks).is_multiple_of(2)
    }

    /// The next character starts a line (after any indent), outside a code fence and inline code.
    pub fn at_line_start_outside_code(&self) -> bool {
        self.fence == 0 && !self.nonblank && self.ticks.is_multiple_of(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(text: &str) -> CodeSpans {
        let mut c = CodeSpans::default();
        c.push(text);
        c
    }

    #[test]
    fn fences_lines_and_ticks() {
        assert!(at("").at_line_start_outside_code());
        assert!(at("text\n").at_line_start_outside_code());
        assert!(at("text\n   ").at_line_start_outside_code());
        assert!(!at("text ").at_line_start_outside_code());
        assert!(at("```\n").in_code());
        assert!(at("  ```rust\ncode\n").in_code());
        assert!(!at("```\ncode\n```\n").in_code());
        assert!(at("~~~\n```\n").in_code());
        assert!(!at("```a```\n").in_code());
        assert!(at("a `b").in_code());
        assert!(at("a `b\nc").in_code());
        assert!(!at("a `b\n\nc").in_code());
        assert!(!at("a `b` c").in_code());
    }

    /// The reference: the tracker the server's other parser uses, on text cut in many ways.
    #[test]
    fn same_as_the_reference_tracker() {
        let mut rng = 0x2545_f491_4f6c_dd1du64;
        let mut next = move |n: usize| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng % n as u64) as usize
        };
        let parts = [
            "`", "```", "~~~", "````", "\n", "\n\n", " ", "  ", "a", "é", "~", "``", "```x", "\t",
            "b `c` d",
        ];
        for _ in 0..20_000 {
            let text: String = (0..next(14)).map(|_| parts[next(parts.len())]).collect();
            let mut mine = CodeSpans::default();
            let mut reference = talus_text::parser::OutputParser::default();
            let cut = text
                .char_indices()
                .map(|(i, _)| i)
                .nth(next(text.chars().count() + 1))
                .unwrap_or(text.len());
            for piece in [&text[..cut], &text[cut..]] {
                mine.push(piece);
                reference.track(piece);
            }
            assert_eq!(mine.in_code(), reference.in_code(), "{text:?}");
            assert_eq!(
                mine.at_line_start_outside_code(),
                reference.opener_ok(),
                "{text:?}"
            );
        }
    }
}
