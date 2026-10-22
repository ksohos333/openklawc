// ui/chat.rs — Phase 2 chat: scrollback + input box + vim/emacs line editing.
//
// Two modes: Normal and Insert. In Normal you scroll / change mode.
// In Insert you type, with emacs-style line editing + Enter to submit.
//
// The submit handler in Phase 2 is a stub: appends a placeholder assistant
// response. Phase 3 will route submits through the MCP client.

use ratatui::{
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Frame,
};

use super::theme::Theme;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Normal,
    Insert,
}

/// State for the `/` command palette. When `active` is true, normal-mode
/// `:`/`/` style is engaged and a single-line command input is shown.
#[derive(Clone, Debug, Default)]
pub struct CommandState {
    pub active: bool,
    pub input: String,
    pub cursor: usize,
}

pub struct Chat {
    pub input: String,
    pub cursor: usize,
    pub scrollback: Vec<ScrollEntry>,
    pub scroll_offset: usize,
    pub mode: Mode,
    pub command: CommandState,
    pub theme: Theme,
}

/// One rendered line in the scrollback.
#[derive(Clone, Debug)]
pub enum ScrollEntry {
    User(String),
    Assistant(String),
    System(String),
}

impl Chat {
    pub fn new() -> Self {
        Self {
            input: String::new(),
            cursor: 0,
            scrollback: vec![ScrollEntry::System(
                "openklawc v0.2 — Phase 4 (LLM + file context + RAG).".to_string(),
            ),
            ScrollEntry::System("type a message and press Enter to chat with your LLM".to_string()),
            ScrollEntry::System("use @file.rs to attach files, / for commands".to_string()),
            ],
            scroll_offset: 0,
            mode: Mode::Insert,
            command: CommandState::default(),
            theme: Theme::gold(),
        }
    }

    /// Insert a single character at the cursor.
    pub fn input_insert(&mut self, c: char) {
        self.input.insert(self.cursor, c);
        self.cursor += c.len_utf8();
    }

    /// Backspace at the cursor (removes the char before the cursor).
    pub fn input_backspace(&mut self) {
        if self.cursor > 0 {
            let prev = self.input[..self.cursor]
                .char_indices()
                .last()
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.input.remove(prev);
            self.cursor = prev;
        }
    }

    /// Delete the character at the cursor (Delete key, not Backspace).
    pub fn input_delete_forward(&mut self) {
        if self.cursor < self.input.len() {
            // Find the next char boundary past the cursor.
            let next = self.input[self.cursor..]
                .char_indices()
                .nth(1)
                .map(|(i, _)| self.cursor + i)
                .unwrap_or(self.input.len());
            self.input.drain(self.cursor..next);
        }
    }

    /// Move cursor to start of line. (Ctrl-a / Home)
    pub fn input_home(&mut self) {
        self.cursor = 0;
    }

    /// Move cursor to end of line. (Ctrl-e / End)
    pub fn input_end(&mut self) {
        self.cursor = self.input.len();
    }

    /// Move cursor one char left.
    pub fn input_left(&mut self) {
        if self.cursor > 0 {
            self.cursor = self.input[..self.cursor]
                .char_indices()
                .last()
                .map(|(i, _)| i)
                .unwrap_or(0);
        }
    }

    /// Move cursor one char right.
    pub fn input_right(&mut self) {
        if self.cursor < self.input.len() {
            let next = self.input[self.cursor..]
                .char_indices()
                .nth(1)
                .map(|(i, _)| self.cursor + i)
                .unwrap_or(self.input.len());
            self.cursor = next;
        }
    }

    /// Kill from cursor to end of line. (Ctrl-k)
    pub fn input_kill_to_end(&mut self) {
        self.input.truncate(self.cursor);
    }

    /// Kill from start of line to cursor. (Ctrl-u)
    pub fn input_kill_to_start(&mut self) {
        self.input.drain(..self.cursor);
        self.cursor = 0;
    }

    /// Kill previous word. (Ctrl-w)
    pub fn input_kill_word(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let prefix = &self.input[..self.cursor];
        // Walk back over non-whitespace, then over whitespace, then cut.
        let mut cut = self.cursor;
        let mut in_word = false;
        for (i, c) in prefix.char_indices().rev() {
            if c.is_whitespace() {
                if in_word {
                    cut = i + c.len_utf8();
                    break;
                }
            } else {
                in_word = true;
                cut = i;
            }
        }
        if !in_word {
            cut = 0;
        }
        self.input.replace_range(cut..self.cursor, "");
        self.cursor = cut;
    }

    /// Submit current input. Adds the user's message to scrollback, clears
    /// the input. Returns the submitted text. The assistant response is
    /// the caller's responsibility (main.rs routes to an MCP and pushes the
    /// real result).
    pub fn submit(&mut self) -> String {
        let msg = self.input.trim().to_string();
        if !msg.is_empty() {
            self.scrollback.push(ScrollEntry::User(msg.clone()));
            self.scroll_offset = 0;
        }
        self.input.clear();
        self.cursor = 0;
        msg
    }

    /// Scroll up one line (toward older messages).
    pub fn scroll_up(&mut self) {
        let max = self.scrollback.len().saturating_sub(1);
        if self.scroll_offset < max {
            self.scroll_offset += 1;
        }
    }

    /// Scroll down one line (toward newer messages).
    pub fn scroll_down(&mut self) {
        if self.scroll_offset > 0 {
            self.scroll_offset -= 1;
        }
    }

    /// Render the full chat: scrollback on top, input box, status bar.
    /// Layout: vertical split — chat area gets remaining space, input is 3 lines, status 1.
    pub fn render(&self, frame: &mut Frame, area: ratatui::layout::Rect) {
        use ratatui::layout::{Constraint, Direction, Layout};
        let t = &self.theme;

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(3),
                Constraint::Length(3),
                Constraint::Length(1),
            ])
            .split(area);

        // --- Scrollback ---
        let items: Vec<ListItem> = self
            .scrollback
            .iter()
            .map(|entry| {
                let (label, text, label_style) = match entry {
                    ScrollEntry::User(s) => (
                        "> ",
                        s.as_str(),
                        Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
                    ),
                    ScrollEntry::Assistant(s) => (
                        "< ",
                        s.as_str(),
                        Style::default().fg(t.fg),
                    ),
                    ScrollEntry::System(s) => (
                        "* ",
                        s.as_str(),
                        Style::default().fg(t.fg_dim).add_modifier(Modifier::ITALIC),
                    ),
                };
                ListItem::new(Line::from(vec![
                    Span::styled(label, label_style),
                    Span::styled(text, Style::default().fg(t.fg)),
                ]))
            })
            .collect();

        let total = self.scrollback.len();
        let scroll = self.scroll_offset;
        let title = if total == 0 {
            " scrollback ".to_string()
        } else {
            format!(" scrollback — {} entries — {} from bottom ", total, scroll)
        };
        let list = List::new(items)
            .block(
                Block::default()
                    .title(Span::styled(title, Style::default().fg(t.accent)))
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(t.accent_dim))
                    .style(Style::default().bg(t.bg)),
            )
            .style(Style::default().fg(t.fg))
            .highlight_style(Style::default().bg(t.bg).fg(t.accent))
            .scroll_padding(0);
        let mut state = ratatui::widgets::ListState::default();
        if total > 0 && scroll < total {
            state.select(Some(total - 1 - scroll));
        }
        frame.render_stateful_widget(list, chunks[0], &mut state);

        // --- Input box (chat or command) ---
        if self.command.active {
            // Command palette is engaged. Show a single-line prompt.
            let cmd_block = Block::default()
                .title(Span::styled(
                    " / — Enter to run, Esc to cancel ",
                    Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.accent))
                .style(Style::default().bg(t.bg));
            let (left, right) = self.command.input
                .split_at(self.command.cursor.min(self.command.input.len()));
            let cmd_lines: Vec<Line> = vec![Line::from(vec![
                Span::styled(left, Style::default().fg(t.fg)),
                Span::styled("│", Style::default().fg(t.accent).add_modifier(Modifier::BOLD)),
                Span::styled(right, Style::default().fg(t.fg)),
            ])];
            let para = Paragraph::new(cmd_lines)
                .block(cmd_block)
                .style(Style::default().bg(t.bg).fg(t.fg))
                .wrap(Wrap { trim: false });
            frame.render_widget(para, chunks[1]);
        } else {
            // Normal chat input.
            let mode_tag = match self.mode {
                Mode::Normal => " NORMAL ",
                Mode::Insert => " INSERT ",
            };
            let mode_style = match self.mode {
                Mode::Normal => Style::default().fg(t.fg_dim),
                Mode::Insert => Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
            };
            let input_block = Block::default()
                .title(Span::styled(
                    format!(" {} — Enter to submit, Esc to toggle mode ", mode_tag),
                    mode_style,
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.accent))
                .style(Style::default().bg(t.bg));

            let (left, right) = self.input.split_at(self.cursor.min(self.input.len()));
            let input_lines: Vec<Line> = vec![Line::from(vec![
                Span::styled(left, Style::default().fg(t.fg)),
                Span::styled("│", Style::default().fg(t.accent).add_modifier(Modifier::BOLD)),
                Span::styled(right, Style::default().fg(t.fg)),
            ])];
            let para = Paragraph::new(input_lines)
                .block(input_block)
                .style(Style::default().bg(t.bg).fg(t.fg))
                .wrap(Wrap { trim: false });
            frame.render_widget(para, chunks[1]);
        }

        // --- Status / mode bar ---
        let input_chars = self.input.chars().count();
        let status = if self.command.active {
            format!(" openklawc v0.1 — phase 3.5: chat + commands — mode: COMMAND — model: <default> ")
        } else {
            format!(
                " openklawc v0.1 — phase 3.5: chat + commands — mode: {} — input: {} chars — model: <default> ",
                match self.mode {
                    Mode::Normal => "NORMAL",
                    Mode::Insert => "INSERT",
                },
                input_chars
            )
        };
        let status_line = Paragraph::new(Line::from(Span::styled(
            status,
            Style::default().fg(t.fg_dim),
        )))
        .style(Style::default().bg(t.bg));
        frame.render_widget(status_line, chunks[2]);
    }
}
