use ratatui::{
    layout::Rect,
    style::Style,
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
    token_count: usize,
    // Verified / unverified claim counts for the session. Rendered only
    // when at least one claim was noted.
    verified: Option<(usize, usize)>,
    theme: &Theme,
) {
    let mut spans = vec![
        Span::styled("model: ", Style::default().fg(theme.fg_dim)),
        Span::styled(model, Style::default().fg(theme.accent)),
        Span::styled(" | ", Style::default().fg(theme.fg_dim)),
        Span::styled("autonomy: ", Style::default().fg(theme.fg_dim)),
        Span::styled(autonomy, Style::default().fg(theme.accent)),
        Span::styled(" | ", Style::default().fg(theme.fg_dim)),
        Span::styled("mcps: ", Style::default().fg(theme.fg_dim)),
        Span::styled(mcp_count.to_string(), Style::default().fg(theme.accent)),
    ];

    if let Some((ok, bad)) = verified {
        if ok + bad > 0 {
            spans.push(Span::styled(" | ", Style::default().fg(theme.fg_dim)));
            spans.push(Span::styled("claims: ", Style::default().fg(theme.fg_dim)));
            spans.push(Span::styled(ok.to_string(), Style::default().fg(theme.ok)));
            spans.push(Span::styled("/", Style::default().fg(theme.fg_dim)));
            spans.push(Span::styled(bad.to_string(), Style::default().fg(theme.error)));
        }
    }

    if token_count > 0 {
        spans.push(Span::styled(" | ", Style::default().fg(theme.fg_dim)));
        spans.push(Span::styled("tokens: ", Style::default().fg(theme.fg_dim)));
        spans.push(Span::styled(
            format!("{}", token_count),
            Style::default().fg(theme.ok),
        ));
    }

    let block = Block::default()
        .borders(Borders::TOP)
        .border_style(Style::default().fg(theme.border));

    let paragraph = Paragraph::new(Line::from(spans)).block(block);
    frame.render_widget(paragraph, area);
}
