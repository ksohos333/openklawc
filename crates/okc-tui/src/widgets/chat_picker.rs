//! The opencode-style chat picker overlay.
//!
//! Layout (mirrors the opencode chat picker):
//!
//! ```text
//! ┌─ OpenCode Chat Picker ────────────────── ← Back  ESC ─┐
//! │ [New Chat (n)] [Secure (S)] [Build (b)] [Plan (p)]  │
//! │ [DELETE (d)]            [ Search sessions…       ] │
//! ├───────────────────────────────┬───────────────────────┤
//! │ ▸ Alpha build                 │ Title: Alpha build   │
//! │   2026-10-26 06:34 · build    │ ID: a                │
//! │                               │ Agent: build         │
//! │                               │ Model: MiniMax-M3    │
//! │                               │ Dir: /home/dev       │
//! │                               │ Updated: …           │
//! │                               │ Created: …           │
//! │                               │ Tokens: in 1,234 …   │
//! │                               │ Cost: $0.0123        │
//! ├───────────────────────────────┴───────────────────────┤
//! │ n New chat · b Build · p Plan · d Delete · esc Back  │
//! └───────────────────────────────────────────────────────┘
//! ```

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
    Frame,
};

use crate::chat::{ChatMode, ChatPickerFocus, ChatPickerState};
use crate::theme::Theme;

/// Render the chat picker over the whole screen.
pub fn render_chat_picker(frame: &mut Frame, area: Rect, state: &ChatPickerState, theme: &Theme) {
    let popup = centered_rect(95, 92, area);
    frame.render_widget(Clear, popup);

    let block = Block::default()
        .title(" OpenCode Chat Picker ")
        .title_bottom(" ← Back  ESC ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.accent));
    let inner = block.inner(popup);
    frame.render_widget(block, popup);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(3),
            Constraint::Length(1),
        ])
        .split(inner);

    render_toolbar(frame, rows[0], state, theme);
    render_body(frame, rows[1], state, theme);
    render_status(frame, rows[2], state, theme);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TOOLBAR
// ═══════════════════════════════════════════════════════════════════════════════

fn render_toolbar(frame: &mut Frame, area: Rect, state: &ChatPickerState, theme: &Theme) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(area);

    // ─── Action buttons ───────────────────────────────────────────────────
    let buttons = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1)])
        .split(cols[0]);

    let row1: Vec<Span> = vec![
        button_span(" New Chat (n) ", theme.ok, state.mode == ChatMode::Build),
        Span::raw(" "),
        button_span(" Secure (S) ", theme.warning, false),
        Span::raw(" "),
        button_span(" Build (b) ", theme.ok, false),
        Span::raw(" "),
        button_span(" Plan (p) ", theme.accent, false),
    ];
    frame.render_widget(Paragraph::new(Line::from(row1)), buttons[0]);

    let row2: Vec<Span> = vec![
        button_span(" DELETE (d) ", theme.error, false),
        Span::raw(" "),
        Span::styled(
            format!("mode: {}", state.mode.label()),
            Style::default().fg(theme.fg_dim),
        ),
    ];
    frame.render_widget(Paragraph::new(Line::from(row2)), buttons[1]);

    // ─── Search box ───────────────────────────────────────────────────────
    let search_focused = state.focus == ChatPickerFocus::Search;
    let search_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(if search_focused {
            theme.accent
        } else {
            theme.border
        }));
    let search_inner = search_block.inner(cols[1]);
    frame.render_widget(search_block, cols[1]);

    let (placeholder, value) = if state.query.is_empty() {
        (true, state.query.clone())
    } else {
        (false, state.query.clone())
    };
    let search_line = if placeholder {
        Line::from(vec![
            Span::styled("🔍 ", Style::default().fg(theme.fg_dim)),
            Span::styled("Search sessions…", Style::default().fg(theme.fg_dim)),
        ])
    } else {
        Line::from(vec![
            Span::styled("🔍 ", Style::default().fg(theme.accent)),
            Span::styled(&value, Style::default().fg(theme.accent)),
        ])
    };
    frame.render_widget(Paragraph::new(search_line), search_inner);

    if search_focused {
        let x = search_inner.x + 2 + value.chars().count() as u16;
        if x < search_inner.right() {
            frame.set_cursor_position((x, search_inner.y));
        }
    }
}

fn button_span(label: &str, color: ratatui::style::Color, active: bool) -> Span<'static> {
    let mut style = Style::default().fg(color);
    if active {
        style = style.add_modifier(Modifier::BOLD | Modifier::REVERSED);
    }
    Span::styled(label.to_string(), style)
}

// ═══════════════════════════════════════════════════════════════════════════════
// BODY — list + detail
// ═══════════════════════════════════════════════════════════════════════════════

fn render_body(frame: &mut Frame, area: Rect, state: &ChatPickerState, theme: &Theme) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(area);

    render_list(frame, cols[0], state, theme);
    render_detail(frame, cols[1], state, theme);
}

fn render_list(frame: &mut Frame, area: Rect, state: &ChatPickerState, theme: &Theme) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.border));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if state.visible.is_empty() {
        let msg = if state.sessions.is_empty() {
            "No chats yet — press n to start one."
        } else {
            "No chats match the search."
        };
        frame.render_widget(
            Paragraph::new(Line::styled(msg, Style::default().fg(theme.fg_dim))),
            inner,
        );
        return;
    }

    let rows = ChatPickerState::rows_that_fit(inner.height);
    let mut lines: Vec<Line> = Vec::new();
    let start = state.scroll.min(state.visible.len().saturating_sub(1));
    let end = (start + rows.max(1)).min(state.visible.len());

    for (row, session_idx) in state.visible[start..end].iter().enumerate() {
        let session = &state.sessions[*session_idx];
        let selected = start + row == state.selected;

        let title_style = if selected {
            Style::default()
                .fg(theme.bg)
                .bg(theme.accent)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(theme.fg)
        };
        let meta_style = if selected {
            Style::default().fg(theme.bg).bg(theme.accent)
        } else {
            Style::default().fg(theme.fg_dim)
        };

        let marker = if selected { "▸ " } else { "  " };
        lines.push(Line::from(vec![
            Span::styled(marker, title_style),
            Span::styled(session.title.clone(), title_style),
        ]));
        lines.push(Line::from(vec![
            Span::styled("    ", meta_style),
            Span::styled(session.meta_line(), meta_style),
        ]));
    }

    frame.render_widget(Paragraph::new(Text::from(lines)), inner);
}

fn render_detail(frame: &mut Frame, area: Rect, state: &ChatPickerState, theme: &Theme) {
    let block = Block::default()
        .title(" Details ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.border));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let Some(session) = state.selected_session() else {
        frame.render_widget(
            Paragraph::new(Line::styled(
                "No chat selected",
                Style::default().fg(theme.fg_dim),
            )),
            inner,
        );
        return;
    };

    let label = Style::default().fg(theme.fg_dim);
    let value = Style::default().fg(theme.fg);

    let mut lines: Vec<Line> = Vec::new();
    let mut field = |k: &str, v: String| {
        lines.push(Line::from(vec![
            Span::styled(format!("{}: ", k), label),
            Span::styled(v, value),
        ]));
    };

    field("Title", clip(&session.title, 24));
    field(
        "ID",
        if session.id.len() > 12 {
            session.id[..12].to_string()
        } else {
            session.id.clone()
        },
    );
    field(
        "Agent",
        if session.agent_mode.is_empty() {
            "default".to_string()
        } else {
            session.agent_mode.clone()
        },
    );
    field(
        "Model",
        session
            .model
            .clone()
            .unwrap_or_else(|| "unknown".to_string()),
    );
    field(
        "Dir",
        session
            .cwd
            .as_deref()
            .map(shorten_path)
            .unwrap_or_else(|| "unknown".to_string()),
    );
    field("Updated", session.updated_at.clone());
    field("Created", session.created_at.clone());
    field("Messages", session.message_count.to_string());
    lines.push(Line::from(vec![
        Span::styled("Tokens: ", label),
        Span::styled(
            format!("in {} / out {}", session.tokens_in, session.tokens_out),
            value,
        ),
    ]));
    lines.push(Line::from(vec![
        Span::styled("Cost: ", label),
        Span::styled(
            match session.cost_usd {
                Some(c) => format!("${:.4}", c),
                None => "n/a (no price configured)".to_string(),
            },
            value,
        ),
    ]));

    frame.render_widget(
        Paragraph::new(Text::from(lines)).wrap(Wrap { trim: true }),
        inner,
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// STATUS BAR
// ═══════════════════════════════════════════════════════════════════════════════

fn render_status(frame: &mut Frame, area: Rect, state: &ChatPickerState, theme: &Theme) {
    // A pending message takes over the status line.
    if let Some(ref notice) = state.notice {
        let spans = vec![
            Span::raw(" "),
            Span::styled(notice.clone(), Style::default().fg(theme.warning)),
        ];
        frame.render_widget(Paragraph::new(Line::from(spans)), area);
        return;
    }

    let spans = vec![
        Span::raw(" "),
        Span::styled("n", Style::default().fg(theme.accent)),
        Span::styled(" New chat · ", Style::default().fg(theme.fg_dim)),
        Span::styled("b", Style::default().fg(theme.accent)),
        Span::styled(" Build · ", Style::default().fg(theme.fg_dim)),
        Span::styled("p", Style::default().fg(theme.accent)),
        Span::styled(" Plan · ", Style::default().fg(theme.fg_dim)),
        Span::styled("d", Style::default().fg(theme.accent)),
        Span::styled(" Delete · ", Style::default().fg(theme.fg_dim)),
        Span::styled("s", Style::default().fg(theme.accent)),
        Span::styled(" Search · ", Style::default().fg(theme.fg_dim)),
        Span::styled("↑↓", Style::default().fg(theme.accent)),
        Span::styled(" Navigate · ", Style::default().fg(theme.fg_dim)),
        Span::styled("Enter", Style::default().fg(theme.accent)),
        Span::styled(" Resume · ", Style::default().fg(theme.fg_dim)),
        Span::styled("esc", Style::default().fg(theme.accent)),
        Span::styled(" Back ", Style::default().fg(theme.fg_dim)),
    ];
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

/// Clip a string to `max` characters, adding an ellipsis when it is cut.
fn clip(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// Shorten a path to its last two components so it fits the detail pane.
fn shorten_path(path: &str) -> String {
    let parts: Vec<&str> = path.rsplit('/').filter(|p| !p.is_empty()).collect();
    if parts.len() <= 2 {
        return path.to_string();
    }
    let tail = parts[..2]
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>()
        .join("/");
    if tail.len() + 2 <= 30 {
        format!("…/{tail}")
    } else {
        clip(&tail, 30)
    }
}

/// Helper: create a centered rect given percentage dimensions.
pub fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}
