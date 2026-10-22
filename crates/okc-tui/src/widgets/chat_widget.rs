use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use crate::theme::Theme;
use crate::app::{ScrollEntry, Mode};

pub fn render_chat(
    frame: &mut Frame,
    area: Rect,
    scrollback: &[ScrollEntry],
    scroll_offset: usize,
    theme: &Theme,
    mode: Mode,
) {
    let mut lines = Vec::new();

    for entry in scrollback.iter().rev().skip(scroll_offset).take(area.height as usize * 2) {
        match entry {
            ScrollEntry::User(text) => {
                lines.push(Line::from(vec![
                    Span::styled("> ", Style::default().fg(theme.accent).add_modifier(Modifier::BOLD)),
                    Span::styled(text.clone(), Style::default().fg(theme.fg)),
                ]));
            }
            ScrollEntry::Assistant(text) => {
                lines.push(Line::from(Span::styled(text.clone(), Style::default().fg(theme.fg))));
            }
            ScrollEntry::System(text) => {
                lines.push(Line::from(Span::styled(
                    text.clone(),
                    Style::default().fg(theme.fg_dim).add_modifier(Modifier::ITALIC),
                )));
            }
            ScrollEntry::Reasoning(text) => {
                lines.push(Line::from(Span::styled(
                    format!("[thinking] {}", text),
                    Style::default().fg(theme.accent_dim).add_modifier(Modifier::ITALIC),
                )));
            }
            ScrollEntry::ToolCall { name, args } => {
                lines.push(Line::from(Span::styled(
                    format!("[[tool: {}]] {}", name, args),
                    Style::default().fg(theme.warning),
                )));
            }
            ScrollEntry::ToolResult { name, success, result } => {
                let color = if *success { theme.ok } else { theme.error };
                lines.push(Line::from(Span::styled(
                    format!("[[result: {}]] {}", name, result),
                    Style::default().fg(color),
                )));
            }
            ScrollEntry::Error(text) => {
                lines.push(Line::from(Span::styled(
                    format!("ERROR: {}", text),
                    Style::default().fg(theme.error).add_modifier(Modifier::BOLD),
                )));
            }
        }
        lines.push(Line::from(""));
    }

    lines.reverse();

    let block = Block::default()
        .title(format!(" openklawc — {}", match mode {
            Mode::Normal => "NORMAL",
            Mode::Insert => "INSERT",
            Mode::Command => "COMMAND",
        }))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.border));

    let paragraph = Paragraph::new(Text::from(lines))
        .block(block)
        .wrap(Wrap { trim: true });

    frame.render_widget(paragraph, area);
}
