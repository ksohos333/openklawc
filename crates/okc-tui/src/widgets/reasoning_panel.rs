use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use crate::theme::Theme;
use okc_core::ReasoningStep;

pub fn render_reasoning(
    frame: &mut Frame,
    area: Rect,
    steps: &[ReasoningStep],
    theme: &Theme,
) {
    let mut lines = Vec::new();
    for step in steps.iter().rev().take(area.height as usize) {
        let icon = match step.kind {
            okc_core::ReasoningKind::Plan => "[PLAN]",
            okc_core::ReasoningKind::Observe => "[OBS]",
            okc_core::ReasoningKind::Reflect => "[REFL]",
            okc_core::ReasoningKind::Decide => "[DEC]",
            okc_core::ReasoningKind::Execute => "[EXEC]",
            okc_core::ReasoningKind::Error => "[ERR]",
        };
        lines.push(Line::from(vec![
            Span::styled(icon, Style::default().fg(theme.accent_dim)),
            Span::styled(" ", Style::default()),
            Span::styled(
                step.content.chars().take(100).collect::<String>(),
                Style::default().fg(theme.fg_dim),
            ),
        ]));
    }
    lines.reverse();

    let block = Block::default()
        .title(" reasoning ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.border));

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, area);
}
