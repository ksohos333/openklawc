use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use crate::theme::Theme;
use okc_core::ReasoningStep;

pub fn render_reasoning(frame: &mut Frame, area: Rect, steps: &[ReasoningStep], theme: &Theme) {
    let mut lines = Vec::new();
    for step in steps.iter().rev().take(area.height as usize) {
        let icon = match step.kind {
            okc_core::ReasoningKind::Plan => "[PLAN]",
            okc_core::ReasoningKind::Observe => "[OBS]",
            okc_core::ReasoningKind::Reflect => "[REFL]",
            okc_core::ReasoningKind::Decide => "[DEC]",
            okc_core::ReasoningKind::Execute => "[EXEC]",
            okc_core::ReasoningKind::Error => "[ERR]",
            okc_core::ReasoningKind::Analyze => "[ANLZ]",
            okc_core::ReasoningKind::Evaluate => "[EVAL]",
            okc_core::ReasoningKind::Backtrack => "[BKTR]",
        };

        let icon_style = if step.decision.is_backtrack {
            Style::default()
                .fg(theme.error)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(theme.accent_dim)
        };

        lines.push(Line::from(vec![
            Span::styled(icon, icon_style),
            Span::styled(" ", Style::default()),
            Span::styled(
                step.content.chars().take(80).collect::<String>(),
                Style::default().fg(theme.fg_dim),
            ),
        ]));

        // Show decision metadata if present
        if step.decision.confidence.is_some()
            || step.decision.chosen_approach.is_some()
            || step.decision.is_backtrack
        {
            let mut meta_parts = Vec::new();
            if let Some(ref c) = step.decision.confidence {
                let color = match c.as_str() {
                    "high" => theme.ok,
                    "medium" => theme.accent_dim,
                    "low" => theme.error,
                    _ => theme.fg_dim,
                };
                meta_parts.push(Span::styled(
                    format!("conf:{}", c),
                    Style::default().fg(color),
                ));
            }
            if let Some(ref a) = step.decision.chosen_approach {
                meta_parts.push(Span::styled(
                    format!(" approach:{}", a.chars().take(30).collect::<String>()),
                    Style::default().fg(theme.fg_dim),
                ));
            }
            if step.decision.is_backtrack {
                meta_parts.push(Span::styled(
                    " BACKTRACK",
                    Style::default()
                        .fg(theme.error)
                        .add_modifier(Modifier::BOLD),
                ));
            }
            if let Some(ref t) = step.decision.key_tradeoff {
                meta_parts.push(Span::styled(
                    format!(" tradeoff:{}", t.chars().take(25).collect::<String>()),
                    Style::default().fg(theme.fg_dim),
                ));
            }
            if !meta_parts.is_empty() {
                let mut meta_line = vec![Span::styled(
                    "        → ",
                    Style::default().fg(theme.border),
                )];
                for (i, part) in meta_parts.iter().enumerate() {
                    if i > 0 {
                        meta_line.push(Span::styled(" | ", Style::default().fg(theme.border)));
                    }
                    meta_line.push(part.clone());
                }
                lines.push(Line::from(meta_line));
            }
        }
    }
    lines.reverse();

    let block = Block::default()
        .title(" reasoning ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.border));

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, area);
}
