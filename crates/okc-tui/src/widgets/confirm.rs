//! The tool-approval prompt shown when the agent wants to run a mutating tool.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use crate::theme::Theme;

/// Render the approval bar. Returns nothing; the caller owns the state.
pub fn render_confirm(
    frame: &mut Frame,
    area: Rect,
    tool_name: &str,
    arguments: &str,
    theme: &Theme,
) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.warning));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(14), Constraint::Min(10)])
        .split(inner);

    let prompt = Line::from(vec![
        Span::styled(
            " Approve? ",
            Style::default()
                .fg(theme.bg)
                .bg(theme.warning)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" y / n ", Style::default().fg(theme.fg_dim)),
    ]);
    frame.render_widget(Paragraph::new(prompt), cols[0]);

    let detail = Line::from(vec![
        Span::styled(tool_name, Style::default().fg(theme.accent)),
        Span::styled("  ", Style::default()),
        Span::styled(arguments, Style::default().fg(theme.fg_dim)),
    ]);
    frame.render_widget(Paragraph::new(detail), cols[1]);
}
