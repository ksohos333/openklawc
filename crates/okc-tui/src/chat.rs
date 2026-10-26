//! Chat picker data model — the opencode-style session browser.
//!
//! The picker shows every stored session in a two-pane layout: a list on the
//! left, the metadata of the highlighted session on the right.

/// Working mode a chat was started in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatMode {
    /// Default coding mode — tools may run freely within the autonomy level.
    #[default]
    Build,
    /// Planning mode — the agent plans before touching anything.
    Plan,
    /// Secure mode — read-only tools only.
    Secure,
}

impl ChatMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            ChatMode::Build => "build",
            ChatMode::Plan => "plan",
            ChatMode::Secure => "secure",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            ChatMode::Build => "Build",
            ChatMode::Plan => "Plan",
            ChatMode::Secure => "Secure",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "build" => Some(ChatMode::Build),
            "plan" => Some(ChatMode::Plan),
            "secure" => Some(ChatMode::Secure),
            _ => None,
        }
    }
}

/// One session as shown in the picker.
#[derive(Debug, Clone, Default)]
pub struct ChatSession {
    pub id: String,
    pub title: String,
    /// Working mode ("build", "plan", "secure"), or empty when unknown.
    pub agent_mode: String,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub cwd: Option<String>,
    /// Pre-formatted timestamp.
    pub updated_at: String,
    pub created_at: String,
    pub tokens_in: u64,
    pub tokens_out: u64,
    /// Cost in USD, when the model has a known price.
    pub cost_usd: Option<f64>,
    pub message_count: i64,
}

impl ChatSession {
    /// The dim second line under the title.
    pub fn meta_line(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if !self.updated_at.is_empty() {
            parts.push(self.updated_at.clone());
        }
        if !self.agent_mode.is_empty() {
            parts.push(self.agent_mode.clone());
        }
        if let Some(m) = &self.model {
            parts.push(m.clone());
        }
        if parts.is_empty() {
            "no metadata".to_string()
        } else {
            parts.join(" · ")
        }
    }
}

/// The outcome of pressing the delete key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeleteArm {
    /// The first press — the deletion is waiting for a second one.
    Armed(String),
    /// The second press — the deletion should happen now.
    Confirmed(String),
    /// Nothing is selected.
    None,
}

/// Which part of the picker has keyboard focus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatPickerFocus {
    /// The session list — single-key shortcuts are active.
    #[default]
    List,
    /// The search box — typed characters go into the query.
    Search,
}

/// State of the chat picker overlay.
#[derive(Debug, Default)]
pub struct ChatPickerState {
    /// Every known session.
    pub sessions: Vec<ChatSession>,
    /// Indices into `sessions` matching the current query.
    pub visible: Vec<usize>,
    /// Index into `visible`.
    pub selected: usize,
    pub query: String,
    pub focus: ChatPickerFocus,
    /// Index of the first rendered row.
    pub scroll: usize,
    /// Mode applied to the next new chat.
    pub mode: ChatMode,
    /// A one-line message shown in the status bar (delete confirmation,
    /// action feedback).
    pub notice: Option<String>,
    /// The chat a delete is waiting to confirm.
    pub pending_delete: Option<String>,
}

impl ChatPickerState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the session list and re-apply the filter.
    pub fn set_sessions(&mut self, sessions: Vec<ChatSession>) {
        self.sessions = sessions;
        self.refilter();
    }

    /// Recompute the visible rows for the current query.
    pub fn refilter(&mut self) {
        let q = self.query.trim().to_lowercase();
        self.visible = self
            .sessions
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                q.is_empty()
                    || s.title.to_lowercase().contains(&q)
                    || s.id.to_lowercase().contains(&q)
                    || s.agent_mode.to_lowercase().contains(&q)
                    || s.model
                        .as_deref()
                        .map(|m| m.to_lowercase().contains(&q))
                        .unwrap_or(false)
            })
            .map(|(i, _)| i)
            .collect();
        if self.selected >= self.visible.len() {
            self.selected = self.visible.len().saturating_sub(1);
        }
        self.scroll = 0;
    }

    /// The currently highlighted session.
    pub fn selected_session(&self) -> Option<&ChatSession> {
        self.visible
            .get(self.selected)
            .and_then(|i| self.sessions.get(*i))
    }

    /// Ask to delete the highlighted chat.
    ///
    /// The first call arms the deletion, the second one carries it out. Any
    /// other action disarms it.
    pub fn arm_delete(&mut self) -> DeleteArm {
        let Some(session) = self.selected_session() else {
            return DeleteArm::None;
        };
        let title = session.title.clone();
        if self.pending_delete.as_deref() == Some(title.as_str()) {
            self.pending_delete = None;
            DeleteArm::Confirmed(title)
        } else {
            self.pending_delete = Some(title.clone());
            DeleteArm::Armed(title)
        }
    }

    /// Cancel a pending delete.
    pub fn cancel_delete(&mut self) {
        self.pending_delete = None;
    }

    /// Move the highlight by `delta` rows, clamped to the visible range.
    pub fn move_selection(&mut self, delta: isize) {
        if self.visible.is_empty() {
            self.selected = 0;
            return;
        }
        let last = self.visible.len() as isize - 1;
        let next = (self.selected as isize + delta).clamp(0, last);
        self.selected = next as usize;
    }

    /// Number of session rows that fit in `rows` terminal lines.
    pub fn rows_that_fit(rows: u16) -> usize {
        (rows as usize) / 2
    }

    /// Scroll so the highlighted row stays inside the viewport.
    pub fn ensure_visible(&mut self, rows: usize) {
        if rows == 0 {
            self.scroll = 0;
            return;
        }
        if self.selected < self.scroll {
            self.scroll = self.selected;
        } else if self.selected >= self.scroll + rows {
            self.scroll = self.selected + 1 - rows;
        }
        let max_scroll = self.visible.len().saturating_sub(rows);
        if self.scroll > max_scroll {
            self.scroll = max_scroll;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<ChatSession> {
        vec![
            ChatSession {
                id: "a".into(),
                title: "Alpha build".into(),
                agent_mode: "build".into(),
                model: Some("MiniMax-M3".into()),
                ..Default::default()
            },
            ChatSession {
                id: "b".into(),
                title: "Beta plan".into(),
                agent_mode: "plan".into(),
                ..Default::default()
            },
            ChatSession {
                id: "c".into(),
                title: "Gamma".into(),
                ..Default::default()
            },
        ]
    }

    #[test]
    fn filter_narrows_by_title() {
        let mut st = ChatPickerState::new();
        st.set_sessions(sample());
        assert_eq!(st.visible.len(), 3);

        st.query = "beta".into();
        st.refilter();
        assert_eq!(st.visible.len(), 1);
        assert_eq!(st.selected_session().unwrap().id, "b");
    }

    #[test]
    fn filter_narrows_by_mode() {
        let mut st = ChatPickerState::new();
        st.set_sessions(sample());
        st.query = "plan".into();
        st.refilter();
        assert_eq!(st.visible.len(), 1);
        assert_eq!(st.selected_session().unwrap().id, "b");
    }

    #[test]
    fn selection_clamps_to_range() {
        let mut st = ChatPickerState::new();
        st.set_sessions(sample());
        st.move_selection(-5);
        assert_eq!(st.selected, 0);
        st.move_selection(99);
        assert_eq!(st.selected, 2);
    }

    #[test]
    fn empty_list_has_no_selection() {
        let mut st = ChatPickerState::new();
        st.set_sessions(Vec::new());
        assert!(st.selected_session().is_none());
        st.move_selection(1);
        assert_eq!(st.selected, 0);
    }

    #[test]
    fn scroll_tracks_selection() {
        let mut st = ChatPickerState::new();
        st.set_sessions(sample());
        st.move_selection(2);
        st.ensure_visible(1);
        assert_eq!(st.scroll, 2);
        st.move_selection(-2);
        st.ensure_visible(1);
        assert_eq!(st.scroll, 0);
    }
}
