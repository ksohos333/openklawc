use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use crate::theme::Theme;

pub fn render_status(
    frame: &mut Frame,
    area: Rect,
    model: &str,
    autonomy: &str,
    mcp_count: usize,
    theme: &Theme,
) {
    let spans = vec![
        Span::styled("model: ", Style::default().fg(theme.fg_dim)),
        Span::styled(model, Style::default().fg(theme.accent)),
        Span::styled(" | ", Style::default().fg(theme.fg_dim)),
        Span::styled("autonomy: ", Style::default().fg(theme.fg_dim)),
        Span::styled(autonomy, Style::default().fg(theme.accent)),
        Span::styled(" | ", Style::default().fg(theme.fg_dim)),
        Span::styled("mcps: ", Style::default().fg(theme.fg_dim)),
        Span::styled(mcp_count.to_string(), Style::default().fg(theme.accent)),
    ];

    let block = Block::default()
        .borders(Borders::TOP)
        .border_style(Style::default().fg(theme.border));

    let paragraph = Paragraph::new(Line::from(spans)).block(block);
    frame.render_widget(paragraph, area);
}
