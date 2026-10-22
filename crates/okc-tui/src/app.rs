use std::collections::VecDeque;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::Frame;

use okc_core::{ReasoningStep, AutonomyLevel};
use crate::theme::Theme;
use crate::widgets::{chat_widget, input_widget, reasoning_panel, status_bar};

#[derive(Debug, Clone)]
pub enum ScrollEntry {
    User(String),
    Assistant(String),
    System(String),
    Reasoning(String),
    ToolCall { name: String, args: String },
    ToolResult { name: String, success: bool, result: String },
    Error(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Normal,
    Insert,
    Command,
}

#[derive(Debug, Clone, Default)]
pub struct CommandState {
    pub active: bool,
    pub input: String,
    pub cursor: usize,
}

pub struct App {
    pub input: String,
    pub cursor: usize,
    pub scrollback: Vec<ScrollEntry>,
    pub scroll_offset: usize,
    pub mode: Mode,
    pub command: CommandState,
    pub theme: Theme,
    pub reasoning_steps: VecDeque<ReasoningStep>,
    pub show_reasoning: bool,
    pub model: String,
    pub autonomy: AutonomyLevel,
    pub mcp_count: usize,
}

impl App {
    pub fn new() -> Self {
        let mut app = Self {
            input: String::new(),
            cursor: 0,
            scrollback: vec![
                ScrollEntry::System("openklawc v0.2 — autonomous Rust agent".to_string()),
                ScrollEntry::System("Type a message and press Enter. Use / for commands.".to_string()),
                ScrollEntry::System("Autonomy: read_only (use /autonomy to change)".to_string()),
            ],
            scroll_offset: 0,
            mode: Mode::Insert,
            command: CommandState::default(),
            theme: Theme::gold(),
            reasoning_steps: VecDeque::new(),
            show_reasoning: true,
            model: "ollama/qwen3".to_string(),
            autonomy: AutonomyLevel::ReadOnly,
            mcp_count: 0,
        };
        app.scroll_to_bottom();
        app
    }

    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        let main_chunks: std::rc::Rc<[Rect]> = if self.show_reasoning {
            Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
                .split(area)
        } else {
            std::rc::Rc::from([area])
        };

        let left_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(3),
                Constraint::Length(3),
                Constraint::Length(1),
            ])
            .split(main_chunks[0]);

        chat_widget::render_chat(
            frame,
            left_chunks[0],
            &self.scrollback,
            self.scroll_offset,
            &self.theme,
            self.mode,
        );

        let prompt = match self.mode {
            Mode::Command => "/",
            _ => "> ",
        };
        input_widget::render_input(
            frame,
            left_chunks[1],
            if self.command.active { &self.command.input } else { &self.input },
            if self.command.active { self.command.cursor } else { self.cursor },
            &self.theme,
            prompt,
        );

        status_bar::render_status(
            frame,
            left_chunks[2],
            &self.model,
            &format!("{:?}", self.autonomy).to_lowercase(),
            self.mcp_count,
            &self.theme,
        );

        if self.show_reasoning && main_chunks.len() > 1 {
            reasoning_panel::render_reasoning(
                frame,
                main_chunks[1],
                &self.reasoning_steps.make_contiguous(),
                &self.theme,
            );
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Option<AppEvent> {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            return Some(AppEvent::Quit);
        }

        if self.command.active {
            return self.handle_command_key(key);
        }

        match self.mode {
            Mode::Normal => self.handle_normal(key),
            Mode::Insert => self.handle_insert(key),
            Mode::Command => self.handle_insert(key),
        }
    }

    fn handle_normal(&mut self, key: KeyEvent) -> Option<AppEvent> {
        match key.code {
            KeyCode::Char('q') => return Some(AppEvent::Quit),
            KeyCode::Char('i') | KeyCode::Char('a') | KeyCode::Char('o') => {
                self.mode = Mode::Insert;
            }
            KeyCode::Char('/') | KeyCode::Char(':') => {
                self.mode = Mode::Command;
                self.command.active = true;
                self.command.input.clear();
                self.command.cursor = 0;
                self.command.input.push(match key.code {
                    KeyCode::Char('/') => '/',
                    _ => ':',
                });
                self.command.cursor = 1;
            }
            KeyCode::Char('j') | KeyCode::Down => self.scroll_down(),
            KeyCode::Char('k') | KeyCode::Up => self.scroll_up(),
            KeyCode::Char('G') => self.scroll_offset = 0,
            KeyCode::Char('g') => {
                // Would need double-g detection; simplified
            }
            _ => {}
        }
        None
    }

    fn handle_insert(&mut self, key: KeyEvent) -> Option<AppEvent> {
        if key.code == KeyCode::Esc {
            self.mode = Mode::Normal;
            return None;
        }

        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('a') => self.cursor_home(),
                KeyCode::Char('e') => self.cursor_end(),
                KeyCode::Char('b') => self.cursor_left(),
                KeyCode::Char('f') => self.cursor_right(),
                KeyCode::Char('k') => self.kill_to_end(),
                KeyCode::Char('u') => self.kill_to_start(),
                KeyCode::Char('w') => self.kill_word(),
                _ => {}
            }
            return None;
        }

        match key.code {
            KeyCode::Enter => {
                let text = self.submit();
                if !text.is_empty() {
                    return Some(AppEvent::Submit(text));
                }
            }
            KeyCode::Backspace => self.backspace(),
            KeyCode::Delete => self.delete_forward(),
            KeyCode::Left => self.cursor_left(),
            KeyCode::Right => self.cursor_right(),
            KeyCode::Home => self.cursor_home(),
            KeyCode::End => self.cursor_end(),
            KeyCode::Char(c) => self.insert(c),
            _ => {}
        }
        None
    }

    fn handle_command_key(&mut self, key: KeyEvent) -> Option<AppEvent> {
        match key.code {
            KeyCode::Esc => {
                self.command.active = false;
                self.mode = Mode::Normal;
                None
            }
            KeyCode::Enter => {
                let cmd = self.command.input.clone();
                self.command.active = false;
                self.command.input.clear();
                self.command.cursor = 0;
                self.mode = Mode::Normal;
                Some(AppEvent::Command(cmd))
            }
            KeyCode::Backspace => {
                if self.command.cursor > 0 {
                    let prev = self.command.input[..self.command.cursor]
                        .char_indices()
                        .last()
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    self.command.input.remove(prev);
                    self.command.cursor = prev;
                }
                None
            }
            KeyCode::Left => {
                if self.command.cursor > 0 {
                    self.command.cursor = self.command.input[..self.command.cursor]
                        .char_indices()
                        .last()
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                }
                None
            }
            KeyCode::Right => {
                if self.command.cursor < self.command.input.len() {
                    self.command.cursor = self.command.input[self.command.cursor..]
                        .char_indices()
                        .nth(1)
                        .map(|(i, _)| self.command.cursor + i)
                        .unwrap_or(self.command.input.len());
                }
                None
            }
            KeyCode::Char(c) => {
                self.command.input.insert(self.command.cursor, c);
                self.command.cursor += c.len_utf8();
                None
            }
            _ => None,
        }
    }

    pub fn submit(&mut self) -> String {
        let text = self.input.clone();
        if !text.is_empty() {
            self.scrollback.push(ScrollEntry::User(text.clone()));
            self.input.clear();
            self.cursor = 0;
            self.scroll_to_bottom();
        }
        text
    }

    pub fn add_assistant(&mut self, text: impl Into<String>) {
        self.scrollback.push(ScrollEntry::Assistant(text.into()));
        self.scroll_to_bottom();
    }

    pub fn add_system(&mut self, text: impl Into<String>) {
        self.scrollback.push(ScrollEntry::System(text.into()));
        self.scroll_to_bottom();
    }

    pub fn add_reasoning(&mut self, text: impl Into<String>) {
        self.scrollback.push(ScrollEntry::Reasoning(text.into()));
        self.scroll_to_bottom();
    }

    pub fn add_error(&mut self, text: impl Into<String>) {
        self.scrollback.push(ScrollEntry::Error(text.into()));
        self.scroll_to_bottom();
    }

    pub fn update_last_assistant(&mut self, text: impl Into<String>) {
        if let Some(last) = self.scrollback.last_mut() {
            if let ScrollEntry::Assistant(ref mut existing) = last {
                *existing = text.into();
            }
        }
    }

    pub fn scroll_up(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_add(1);
    }

    pub fn scroll_down(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_sub(1);
    }

    pub fn scroll_to_bottom(&mut self) {
        self.scroll_offset = 0;
    }

    // Input editing
    fn insert(&mut self, c: char) {
        self.input.insert(self.cursor, c);
        self.cursor += c.len_utf8();
    }

    fn backspace(&mut self) {
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

    fn delete_forward(&mut self) {
        if self.cursor < self.input.len() {
            let next = self.input[self.cursor..]
                .char_indices()
                .nth(1)
                .map(|(i, _)| self.cursor + i)
                .unwrap_or(self.input.len());
            self.input.drain(self.cursor..next);
        }
    }

    fn cursor_left(&mut self) {
        if self.cursor > 0 {
            self.cursor = self.input[..self.cursor]
                .char_indices()
                .last()
                .map(|(i, _)| i)
                .unwrap_or(0);
        }
    }

    fn cursor_right(&mut self) {
        if self.cursor < self.input.len() {
            let next = self.input[self.cursor..]
                .char_indices()
                .nth(1)
                .map(|(i, _)| self.cursor + i)
                .unwrap_or(self.input.len());
            self.cursor = next;
        }
    }

    fn cursor_home(&mut self) {
        self.cursor = 0;
    }

    fn cursor_end(&mut self) {
        self.cursor = self.input.len();
    }

    fn kill_to_end(&mut self) {
        self.input.truncate(self.cursor);
    }

    fn kill_to_start(&mut self) {
        self.input.drain(..self.cursor);
        self.cursor = 0;
    }

    fn kill_word(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let prefix = &self.input[..self.cursor];
        let new_cursor = prefix
            .trim_end_matches(|c: char| c.is_ascii_punctuation())
            .rfind(|c: char| c.is_ascii_punctuation() || c.is_whitespace())
            .map(|i| i + 1)
            .unwrap_or(0);
        self.input.drain(new_cursor..self.cursor);
        self.cursor = new_cursor;
    }
}

#[derive(Debug, Clone)]
pub enum AppEvent {
    Quit,
    Submit(String),
    Command(String),
}
