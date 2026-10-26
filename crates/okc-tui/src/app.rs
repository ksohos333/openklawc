//! Enhanced TUI app with opencode-inspired features:
//!   - Slash command system
//!   - Input history (up/down)
//!   - Session/model picker overlays
//!   - Help overlay
//!   - Search integration

use std::collections::VecDeque;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::Frame;

use crate::chat::{ChatMode, ChatPickerFocus, ChatPickerState, DeleteArm};
use crate::theme::Theme;
use crate::widgets::{
    chat_picker, chat_widget, confirm, input_widget, overlay, reasoning_panel, status_bar,
};
use okc_core::{AutonomyLevel, ReasoningStep};

#[derive(Debug, Clone)]
pub enum ScrollEntry {
    User(String),
    Assistant(String),
    System(String),
    Reasoning(String),
    ToolCall {
        name: String,
        args: String,
    },
    ToolResult {
        name: String,
        success: bool,
        result: String,
    },
    Error(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Normal,
    Insert,
    Command,
}

/// Active overlay (modal) state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Overlay {
    None,
    Sessions,
    Models,
    Help,
    Search,
    /// The opencode-style chat picker.
    ChatPicker,
}

/// The app state.
pub struct App {
    pub input: String,
    pub cursor: usize,
    pub scrollback: Vec<ScrollEntry>,
    pub scroll_offset: usize,
    pub mode: Mode,
    pub theme: Theme,
    pub reasoning_steps: VecDeque<ReasoningStep>,
    pub show_reasoning: bool,
    pub model: String,
    pub autonomy: AutonomyLevel,
    pub mcp_count: usize,

    // ─── Input history ────────────────────────────────────────────────────
    pub input_history: VecDeque<String>,
    pub history_index: Option<usize>,
    pub saved_input: String,
    /// Set after a lone `g` in normal mode, waiting for the second `g`.
    pub pending_g: bool,

    // ─── Overlay state ────────────────────────────────────────────────────
    pub overlay: Overlay,
    pub overlay_items: Vec<overlay::PickerItem>,
    /// The unfiltered list the overlay filter is applied to.
    pub overlay_source: Vec<overlay::PickerItem>,
    pub overlay_selected: usize,
    pub overlay_filter: String,

    // ─── Session state ────────────────────────────────────────────────────
    pub current_session_id: Option<String>,
    pub token_count: usize,
    pub iteration_count: usize,
    /// Verified / unverified claim counts for the current session, fed by
    /// the anticonf audit trail. `None` until a session starts.
    pub verify_badge: Option<(usize, usize)>,

    // ─── Chat picker ───────────────────────────────────────────────────────
    pub chat_picker: ChatPickerState,

    // ─── Tool approval ─────────────────────────────────────────────────────
    /// Set while a tool call is waiting for the user's yes/no.
    pub pending_confirm: Option<PendingConfirm>,
}

/// A tool call the agent wants to run, awaiting approval.
#[derive(Debug, Clone)]
pub struct PendingConfirm {
    pub name: String,
    pub arguments: String,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    pub fn new() -> Self {
        let mut app = Self {
            input: String::new(),
            cursor: 0,
            scrollback: vec![
                ScrollEntry::System("openklawc — type a message and press Enter".to_string()),
                ScrollEntry::System("/help for commands  •  Esc for normal mode".to_string()),
            ],
            scroll_offset: 0,
            mode: Mode::Insert,
            theme: Theme::gold(),
            reasoning_steps: VecDeque::new(),
            show_reasoning: true,
            model: "ollama/qwen3".to_string(),
            autonomy: AutonomyLevel::ReadOnly,
            mcp_count: 0,
            input_history: VecDeque::with_capacity(100),
            history_index: None,
            saved_input: String::new(),
            pending_g: false,
            overlay: Overlay::None,
            overlay_items: Vec::new(),
            overlay_source: Vec::new(),
            overlay_selected: 0,
            overlay_filter: String::new(),
            current_session_id: None,
            token_count: 0,
            iteration_count: 0,
            verify_badge: None,
            chat_picker: ChatPickerState::new(),
            pending_confirm: None,
        };
        app.scroll_to_bottom();
        app
    }

    // ═════════════════════════════════════════════════════════════════════════
    // RENDERING
    // ═════════════════════════════════════════════════════════════════════════
    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        let main_area = if self.overlay != Overlay::None {
            // Dim the background when overlay is active
            self.render_main(frame, area);
            self.render_overlay(frame, area);
            return;
        } else {
            area
        };

        self.render_main(frame, main_area);
    }

    fn render_main(&mut self, frame: &mut Frame, area: Rect) {
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
            &self.input,
            self.cursor,
            &self.theme,
            prompt,
        );

        status_bar::render_status(
            frame,
            left_chunks[2],
            &self.model,
            &format!("{:?}", self.autonomy).to_lowercase(),
            self.mcp_count,
            self.token_count,
            self.verify_badge,
            &self.theme,
        );

        // A pending approval replaces the input box so it is impossible to miss.
        if let Some(ref pending) = self.pending_confirm {
            confirm::render_confirm(
                frame,
                left_chunks[1],
                &pending.name,
                &pending.arguments,
                &self.theme,
            );
        }

        if self.show_reasoning && main_chunks.len() > 1 {
            reasoning_panel::render_reasoning(
                frame,
                main_chunks[1],
                self.reasoning_steps.make_contiguous(),
                &self.theme,
            );
        }
    }

    fn render_overlay(&self, frame: &mut Frame, area: Rect) {
        match self.overlay {
            Overlay::Help => overlay::render_help(frame, area, &self.theme),
            Overlay::Sessions => overlay::render_picker(
                frame,
                area,
                "Sessions",
                &self.overlay_items,
                self.overlay_selected,
                Some(&self.overlay_filter),
                &self.theme,
            ),
            Overlay::Models => overlay::render_picker(
                frame,
                area,
                "Models",
                &self.overlay_items,
                self.overlay_selected,
                Some(&self.overlay_filter),
                &self.theme,
            ),
            Overlay::Search => overlay::render_picker(
                frame,
                area,
                "Search Results",
                &self.overlay_items,
                self.overlay_selected,
                Some(&self.overlay_filter),
                &self.theme,
            ),
            Overlay::ChatPicker => {
                chat_picker::render_chat_picker(frame, area, &self.chat_picker, &self.theme)
            }
            Overlay::None => {}
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // INPUT HANDLING
    // ═════════════════════════════════════════════════════════════════════════
    pub fn handle_key(&mut self, key: KeyEvent) -> Option<AppEvent> {
        // Global shortcuts
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            return Some(AppEvent::Quit);
        }

        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('l') {
            return Some(AppEvent::Redraw);
        }

        // A pending tool approval takes over the keyboard.
        if self.pending_confirm.is_some() {
            return self.handle_confirm_key(key);
        }

        // Tab toggles reasoning panel
        if key.code == KeyCode::Tab {
            self.show_reasoning = !self.show_reasoning;
            return None;
        }

        // Overlay mode takes priority
        if self.overlay == Overlay::ChatPicker {
            return self.handle_chat_picker_key(key);
        }

        if self.overlay != Overlay::None {
            return self.handle_overlay_key(key);
        }

        // Help shortcut
        if key.code == KeyCode::Char('?') && self.mode == Mode::Normal {
            self.open_help();
            return None;
        }

        match self.mode {
            Mode::Normal => self.handle_normal(key),
            Mode::Insert => self.handle_insert(key),
            Mode::Command => self.handle_command_input(key),
        }
    }

    /// y approves the pending tool call, n / Esc denies it.
    fn handle_confirm_key(&mut self, key: KeyEvent) -> Option<AppEvent> {
        match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') => {
                self.pending_confirm = None;
                Some(AppEvent::ConfirmTool(true))
            }
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                self.pending_confirm = None;
                Some(AppEvent::ConfirmTool(false))
            }
            _ => None,
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
                self.input.clear();
                self.cursor = 0;
                self.input.push(match key.code {
                    KeyCode::Char('/') => '/',
                    _ => ':',
                });
                self.cursor = 1;
            }
            KeyCode::Char('j') | KeyCode::Down => self.scroll_down(),
            KeyCode::Char('k') | KeyCode::Up => self.scroll_up(),
            KeyCode::Char('G') => {
                self.scroll_offset = 0;
                self.pending_g = false;
            }
            KeyCode::Char('g') => {
                if self.pending_g {
                    // gg -> jump to the bottom
                    self.scroll_to_bottom();
                    self.pending_g = false;
                } else {
                    self.pending_g = true;
                }
            }
            KeyCode::Char('c') => {
                self.open_chat_picker();
                return Some(AppEvent::OpenChatPicker);
            }
            _ => {
                self.pending_g = false;
            }
        }
        None
    }

    fn handle_insert(&mut self, key: KeyEvent) -> Option<AppEvent> {
        if key.code == KeyCode::Esc {
            self.mode = Mode::Normal;
            self.history_index = None;
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
                KeyCode::Char('l') => return Some(AppEvent::Redraw),
                _ => {}
            }
            return None;
        }

        match key.code {
            KeyCode::Enter => {
                let text = self.input.clone();
                if !text.is_empty() {
                    // Save to history
                    self.input_history.push_back(text.clone());
                    if self.input_history.len() > 100 {
                        self.input_history.pop_front();
                    }
                    self.history_index = None;

                    if text.starts_with('/') {
                        return self.handle_slash_command(&text);
                    }

                    self.scrollback.push(ScrollEntry::User(text.clone()));
                    self.input.clear();
                    self.cursor = 0;
                    self.scroll_to_bottom();
                    return Some(AppEvent::Submit(text));
                }
            }
            KeyCode::Backspace => self.backspace(),
            KeyCode::Delete => self.delete_forward(),
            KeyCode::Left => self.cursor_left(),
            KeyCode::Right => self.cursor_right(),
            KeyCode::Home => self.cursor_home(),
            KeyCode::End => self.cursor_end(),
            KeyCode::Up => self.history_prev(),
            KeyCode::Down => self.history_next(),
            KeyCode::Char(c) => self.insert(c),
            _ => {}
        }
        None
    }

    fn handle_command_input(&mut self, key: KeyEvent) -> Option<AppEvent> {
        if key.code == KeyCode::Esc {
            self.mode = Mode::Normal;
            self.input.clear();
            self.cursor = 0;
            return None;
        }

        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('a') => self.cursor_home(),
                KeyCode::Char('e') => self.cursor_end(),
                KeyCode::Char('k') => self.kill_to_end(),
                KeyCode::Char('u') => self.kill_to_start(),
                KeyCode::Char('w') => self.kill_word(),
                _ => {}
            }
            return None;
        }

        match key.code {
            KeyCode::Enter => {
                let cmd = self.input.clone();
                self.mode = Mode::Normal;
                self.input.clear();
                self.cursor = 0;
                if !cmd.is_empty() {
                    return self.handle_slash_command(&cmd);
                }
            }
            KeyCode::Backspace => self.backspace(),
            KeyCode::Left => self.cursor_left(),
            KeyCode::Right => self.cursor_right(),
            KeyCode::Home => self.cursor_home(),
            KeyCode::End => self.cursor_end(),
            KeyCode::Char(c) => self.insert(c),
            _ => {}
        }
        None
    }

    // ═════════════════════════════════════════════════════════════════════════
    // SLASH COMMAND SYSTEM
    // ═════════════════════════════════════════════════════════════════════════
    fn handle_slash_command(&mut self, text: &str) -> Option<AppEvent> {
        let parts: Vec<&str> = text.split_whitespace().collect();
        let cmd = parts.first().unwrap_or(&"");

        match *cmd {
            "/help" | "/h" => {
                self.open_help();
                None
            }
            "/clear" => {
                self.scrollback.clear();
                self.add_system("Scrollback cleared.");
                None
            }
            "/models" => {
                self.overlay = Overlay::Models;
                self.overlay_selected = 0;
                self.overlay_filter.clear();
                Some(AppEvent::ListModels)
            }
            "/sessions" => {
                self.overlay = Overlay::Sessions;
                self.overlay_selected = 0;
                self.overlay_filter.clear();
                Some(AppEvent::ListSessions)
            }
            "/chats" | "/picker" => {
                self.open_chat_picker();
                Some(AppEvent::OpenChatPicker)
            }
            "/autonomy" => {
                if let Some(level) = parts.get(1) {
                    let level = match *level {
                        "ask" => AutonomyLevel::Ask,
                        "read_only" => AutonomyLevel::ReadOnly,
                        "yolo" => AutonomyLevel::Yolo,
                        "god" => AutonomyLevel::God,
                        _ => {
                            self.add_error(format!(
                                "Unknown autonomy level: {}. Use ask/read_only/yolo/god",
                                level
                            ));
                            return None;
                        }
                    };
                    self.autonomy = level;
                    self.add_system(format!("Autonomy set to: {:?}", self.autonomy).to_lowercase());
                    Some(AppEvent::SetAutonomy(level))
                } else {
                    self.add_system(
                        format!("Current autonomy: {:?}", self.autonomy).to_lowercase(),
                    );
                    None
                }
            }
            "/search" => {
                let query = parts[1..].join(" ");
                if query.is_empty() {
                    self.add_error("Usage: /search <query>");
                    return None;
                }
                self.overlay = Overlay::Search;
                self.overlay_selected = 0;
                self.overlay_filter = query.clone();
                Some(AppEvent::SearchHistory(query))
            }
            "/reasoning" => {
                self.show_reasoning = !self.show_reasoning;
                self.add_system(format!(
                    "Reasoning panel: {}",
                    if self.show_reasoning { "on" } else { "off" }
                ));
                None
            }
            "/audit" | "/trail" => Some(AppEvent::AuditTrail),
            "/quit" | "/q" => Some(AppEvent::Quit),
            _ => {
                self.add_error(format!(
                    "Unknown command: {}. Type /help for available commands.",
                    cmd
                ));
                None
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // OVERLAY HANDLING
    // ═════════════════════════════════════════════════════════════════════════
    fn handle_overlay_key(&mut self, key: KeyEvent) -> Option<AppEvent> {
        match key.code {
            KeyCode::Esc => {
                self.close_overlay();
                None
            }
            KeyCode::Enter => {
                // Capture the overlay kind *before* closing: close_overlay()
                // resets `self.overlay` to None, which made the match below
                // unreachable and left every picker unselectable.
                let kind = self.overlay.clone();
                let id = self
                    .overlay_items
                    .get(self.overlay_selected)
                    .map(|item| item.id.clone());
                self.close_overlay();
                match (kind, id) {
                    (Overlay::Sessions, Some(id)) | (Overlay::Search, Some(id)) => {
                        return Some(AppEvent::ResumeSession(id))
                    }
                    (Overlay::Models, Some(id)) => return Some(AppEvent::SwitchModel(id)),
                    _ => {}
                }
                None
            }
            KeyCode::Up | KeyCode::Char('k') => {
                if self.overlay_selected > 0 {
                    self.overlay_selected -= 1;
                }
                None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.overlay_selected + 1 < self.overlay_items.len() {
                    self.overlay_selected += 1;
                }
                None
            }
            KeyCode::Char('/') => {
                // Start filtering
                self.overlay_filter.clear();
                self.apply_overlay_filter();
                None
            }
            KeyCode::Backspace => {
                if !self.overlay_filter.is_empty() {
                    self.overlay_filter.pop();
                    self.apply_overlay_filter();
                }
                None
            }
            KeyCode::Char(c) => {
                self.overlay_filter.push(c);
                self.overlay_selected = 0;
                self.apply_overlay_filter();
                None
            }
            _ => None,
        }
    }

    /// Keyboard handling for the chat picker overlay.
    ///
    /// The list and the search box have separate focus: with the list focused
    /// single-key shortcuts (n/b/p/d/s) act, with the search focused typed
    /// characters go into the query.
    fn handle_chat_picker_key(&mut self, key: KeyEvent) -> Option<AppEvent> {
        // Any action other than the delete key disarms a pending delete.
        if key.code != KeyCode::Char('d') {
            self.chat_picker.cancel_delete();
        }
        match key.code {
            KeyCode::Esc => {
                if self.chat_picker.focus == ChatPickerFocus::Search {
                    // First Esc leaves the search box, second closes the picker.
                    self.chat_picker.focus = ChatPickerFocus::List;
                } else {
                    self.close_overlay();
                }
                None
            }
            KeyCode::Left => {
                self.chat_picker.focus = ChatPickerFocus::List;
                None
            }
            KeyCode::Enter => {
                let id = self.chat_picker.selected_session().map(|s| s.id.clone());
                match id {
                    Some(id) => {
                        self.close_overlay();
                        Some(AppEvent::ResumeSession(id))
                    }
                    None => None,
                }
            }
            KeyCode::Up => {
                self.chat_picker.move_selection(-1);
                None
            }
            KeyCode::Down => {
                self.chat_picker.move_selection(1);
                None
            }
            KeyCode::PageUp => {
                self.chat_picker.move_selection(-10);
                None
            }
            KeyCode::PageDown => {
                self.chat_picker.move_selection(10);
                None
            }
            KeyCode::Home => {
                self.chat_picker.selected = 0;
                None
            }
            KeyCode::End => {
                self.chat_picker.selected = self.chat_picker.visible.len().saturating_sub(1);
                None
            }
            KeyCode::Backspace => {
                if self.chat_picker.focus == ChatPickerFocus::Search {
                    self.chat_picker.query.pop();
                    self.chat_picker.refilter();
                }
                None
            }
            KeyCode::Char(c) => self.handle_chat_picker_char(c),
            _ => None,
        }
    }

    fn handle_chat_picker_char(&mut self, c: char) -> Option<AppEvent> {
        if self.chat_picker.focus == ChatPickerFocus::Search {
            self.chat_picker.query.push(c);
            self.chat_picker.refilter();
            return None;
        }

        match c {
            'n' => {
                let mode = self.chat_picker.mode;
                Some(AppEvent::CreateSession(mode))
            }
            'b' => {
                self.chat_picker.mode = ChatMode::Build;
                Some(AppEvent::CreateSession(ChatMode::Build))
            }
            'p' => {
                self.chat_picker.mode = ChatMode::Plan;
                Some(AppEvent::CreateSession(ChatMode::Plan))
            }
            'S' => {
                self.chat_picker.mode = ChatMode::Secure;
                Some(AppEvent::CreateSession(ChatMode::Secure))
            }
            'd' => match self.chat_picker.arm_delete() {
                DeleteArm::Armed(title) => {
                    self.chat_picker.notice = Some(format!("Press d again to delete “{title}”"));
                    None
                }
                DeleteArm::Confirmed(_) => self
                    .chat_picker
                    .selected_session()
                    .map(|s| AppEvent::DeleteSession(s.id.clone())),
                DeleteArm::None => None,
            },
            's' | '/' => {
                self.chat_picker.focus = ChatPickerFocus::Search;
                None
            }
            'q' => Some(AppEvent::Quit),
            _ => None,
        }
    }

    fn close_overlay(&mut self) {
        self.overlay = Overlay::None;
        self.overlay_items.clear();
        self.overlay_source.clear();
        self.overlay_selected = 0;
        self.overlay_filter.clear();
    }

    /// Filter the overlay items by the current filter text.
    ///
    /// The items pushed in by the host are the unfiltered set; this keeps a
    /// copy of the original list so the filter can be re-applied in either
    /// direction without asking the host to re-query.
    fn apply_overlay_filter(&mut self) {
        if self.overlay_source.is_empty() {
            self.overlay_source = self.overlay_items.clone();
        }
        let needle = self.overlay_filter.trim().to_lowercase();
        if needle.is_empty() {
            self.overlay_items = self.overlay_source.clone();
        } else {
            self.overlay_items = self
                .overlay_source
                .iter()
                .filter(|item| {
                    item.title.to_lowercase().contains(&needle)
                        || item
                            .subtitle
                            .as_deref()
                            .map(|s| s.to_lowercase().contains(&needle))
                            .unwrap_or(false)
                })
                .cloned()
                .collect();
        }
        if self.overlay_selected >= self.overlay_items.len() {
            self.overlay_selected = 0;
        }
    }

    pub fn set_overlay_items(&mut self, items: Vec<overlay::PickerItem>) {
        self.overlay_source = items.clone();
        self.overlay_items = items;
        self.overlay_selected = 0;
    }

    fn open_help(&mut self) {
        self.overlay = Overlay::Help;
    }

    /// Open the chat picker overlay and reset its transient state.
    pub fn open_chat_picker(&mut self) {
        self.overlay = Overlay::ChatPicker;
        self.chat_picker.selected = 0;
        self.chat_picker.scroll = 0;
        self.chat_picker.focus = ChatPickerFocus::List;
        self.chat_picker.query.clear();
        self.chat_picker.notice = None;
        self.chat_picker.cancel_delete();
        self.chat_picker.refilter();
    }

    // ═════════════════════════════════════════════════════════════════════════
    // INPUT HISTORY
    // ═════════════════════════════════════════════════════════════════════════
    fn history_prev(&mut self) {
        if self.input_history.is_empty() {
            return;
        }
        if self.history_index.is_none() {
            self.saved_input = self.input.clone();
            self.history_index = Some(self.input_history.len() - 1);
        } else if let Some(idx) = self.history_index {
            if idx > 0 {
                self.history_index = Some(idx - 1);
            }
        }
        if let Some(idx) = self.history_index {
            self.input = self.input_history[idx].clone();
            self.cursor = self.input.len();
        }
    }

    fn history_next(&mut self) {
        if let Some(idx) = self.history_index {
            if idx + 1 < self.input_history.len() {
                self.history_index = Some(idx + 1);
                self.input = self.input_history[idx + 1].clone();
                self.cursor = self.input.len();
            } else {
                self.history_index = None;
                self.input = self.saved_input.clone();
                self.cursor = self.input.len();
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // SCROLLBACK
    // ═════════════════════════════════════════════════════════════════════════
    pub fn add_assistant(&mut self, text: impl Into<String>) {
        self.scrollback.push(ScrollEntry::Assistant(text.into()));
        self.scroll_to_bottom();
    }

    pub fn add_system(&mut self, text: impl Into<String>) {
        self.scrollback.push(ScrollEntry::System(text.into()));
        self.scroll_to_bottom();
    }

    pub fn add_error(&mut self, text: impl Into<String>) {
        self.scrollback.push(ScrollEntry::Error(text.into()));
        self.scroll_to_bottom();
    }

    pub fn add_reasoning(&mut self, text: impl Into<String>) {
        self.scrollback.push(ScrollEntry::Reasoning(text.into()));
        self.scroll_to_bottom();
    }

    pub fn update_last_assistant(&mut self, text: impl Into<String>) {
        if let Some(ScrollEntry::Assistant(existing)) = self.scrollback.last_mut() {
            *existing = text.into();
        }
    }

    /// Clear the visible transcript, keeping the two startup hint lines.
    pub fn clear_transcript(&mut self) {
        self.scrollback.clear();
        self.scroll_to_bottom();
    }

    /// Replace the visible transcript with a stored conversation.
    ///
    /// Takes `(role, content)` pairs so the TUI does not have to depend on the
    /// session store.
    pub fn set_transcript(&mut self, messages: &[(String, String)]) {
        self.scrollback.clear();
        for (role, content) in messages {
            if role == "system" {
                continue;
            }
            let entry = match role.as_str() {
                "assistant" => ScrollEntry::Assistant(content.clone()),
                "tool" => ScrollEntry::ToolResult {
                    name: "tool".to_string(),
                    success: true,
                    result: content.clone(),
                },
                _ => ScrollEntry::User(content.clone()),
            };
            self.scrollback.push(entry);
        }
        self.scroll_to_bottom();
    }

    fn scroll_up(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_add(1);
    }

    fn scroll_down(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_sub(1);
    }

    fn scroll_to_bottom(&mut self) {
        self.scroll_offset = 0;
    }

    // ═════════════════════════════════════════════════════════════════════════
    // INPUT EDITING
    // ═════════════════════════════════════════════════════════════════════════
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

// ═════════════════════════════════════════════════════════════════════════════
// APP EVENTS
// ═════════════════════════════════════════════════════════════════════════════
#[derive(Debug, Clone)]
pub enum AppEvent {
    Quit,
    Submit(String),
    Redraw,

    // Slash commands
    ListModels,
    ListSessions,
    SetAutonomy(AutonomyLevel),
    SwitchModel(String),
    ResumeSession(String),
    SearchHistory(String),

    // Chat picker
    /// Open the chat picker and ask the host to load the session list.
    OpenChatPicker,
    /// Start a new chat in the given working mode.
    CreateSession(ChatMode),
    /// Delete a stored chat by id.
    DeleteSession(String),
    /// Answer a pending tool approval: true allows the call.
    ConfirmTool(bool),
    /// Show the audit trail (events + reasoning steps) of the current
    /// session.
    AuditTrail,
}
