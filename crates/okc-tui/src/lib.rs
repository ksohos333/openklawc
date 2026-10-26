//! openklawc-tui — ratatui-based terminal interface with opencode-inspired features.
//!
//! Features:
//!   - Slash command system (/help, /models, /sessions, /chats, /autonomy, /search, /clear)
//!   - Input history with up/down recall
//!   - Session picker overlay
//!   - Model picker overlay
//!   - opencode-style chat picker overlay
//!   - Help overlay
//!   - Reasoning panel with metadata display
//!   - Vim-style modal editing

pub mod app;
pub mod chat;
pub mod theme;
pub mod widgets;

pub use app::{App, AppEvent, Mode, Overlay, PendingConfirm, ScrollEntry};
pub use chat::{ChatMode, ChatPickerFocus, ChatPickerState, ChatSession, DeleteArm};
pub use theme::Theme;
pub use widgets::overlay::PickerItem;
