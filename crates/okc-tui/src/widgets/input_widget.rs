use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use crate::theme::Theme;

pub fn render_input(
    frame: &mut Frame,
    area: Rect,
    input: &str,
    cursor: usize,
    theme: &Theme,
    prompt: &str,
) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.accent));

    let line = Line::from(vec![
        Span::styled(prompt, Style::default().fg(theme.accent).add_modifier(Modifier::BOLD)),
        Span::styled(input.to_string(), Style::default().fg(theme.fg)),
    ]);

    let paragraph = Paragraph::new(line).block(block);
    frame.render_widget(paragraph, area);

    // Position cursor
    let cursor_x = area.x + prompt.len() as u16 + unicode_width::UnicodeWidthStr::width(&input[..cursor.min(input.len())]) as u16 + 1;
    let cursor_y = area.y + 1;
    frame.set_cursor_position((cursor_x, cursor_y));
}
