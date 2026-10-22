//! openklawc-tui — ratatui-based terminal interface.
//!
//! Layout:
//!   ┌─────────────────────────────────────────┐
//!   │  scrollback (chat history)              │
//!   │                                         │
//!   │                                         │
//!   ├─────────────────────────────────────────┤
//!   │  input line                             │
//!   ├─────────────────────────────────────────┤
//!   │  status bar (model, autonomy, MCPs)     │
//!   └─────────────────────────────────────────┘
//!
//! When reasoning is enabled, a right panel shows the agent's thought process.

pub mod app;
pub mod theme;
pub mod widgets;

pub use app::{App, AppEvent};
pub use theme::Theme;
