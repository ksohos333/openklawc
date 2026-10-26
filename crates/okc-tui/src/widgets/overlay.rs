//! Modal overlays: session picker, model picker, help, etc.

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Margin, Rect},
    style::{Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
    Frame,
};

use crate::theme::Theme;

/// A list item in a picker overlay.
///
/// The highlighted row is tracked by index (`App::overlay_selected`), not by
/// a flag on the item.
#[derive(Debug, Clone)]
pub struct PickerItem {
    pub id: String,
    pub title: String,
    pub subtitle: Option<String>,
}

/// Render a centered modal picker overlay.
pub fn render_picker(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    items: &[PickerItem],
    selected_idx: usize,
    filter: Option<&str>,
    theme: &Theme,
) {
    // Centered popup
    let popup_area = centered_rect(60, 70, area);
    frame.render_widget(Clear, popup_area);

    let block = Block::default()
        .title(format!(" {} ", title))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.accent));

    let inner = popup_area.inner(Margin::new(2, 1));

    let mut lines = Vec::new();

    // Filter hint
    if let Some(f) = filter {
        lines.push(Line::from(vec![
            Span::styled("Filter: ", Style::default().fg(theme.fg_dim)),
            Span::styled(
                f,
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        lines.push(Line::from(""));
    }

    // Line offset of every item, so the viewport can scroll to the
    // highlighted one.
    let mut item_offsets: Vec<usize> = Vec::with_capacity(items.len());

    for (i, item) in items.iter().enumerate() {
        item_offsets.push(lines.len());
        let is_selected = i == selected_idx;
        let icon = if is_selected { "▸ " } else { "  " };
        let color = if is_selected { theme.accent } else { theme.fg };
        let style = if is_selected {
            Style::default().fg(color).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(color)
        };

        lines.push(Line::from(vec![
            Span::styled(icon, style),
            Span::styled(&item.title, style),
        ]));

        if let Some(sub) = &item.subtitle {
            let sub_color = if is_selected {
                theme.accent_dim
            } else {
                theme.fg_dim
            };
            lines.push(Line::from(vec![
                Span::styled("    ", Style::default()),
                Span::styled(sub.as_str(), Style::default().fg(sub_color)),
            ]));
        }
    }

    if items.is_empty() {
        lines.push(Line::styled("No items", Style::default().fg(theme.fg_dim)));
    }

    // Scroll so the highlighted item stays visible.
    let scroll = if item_offsets.is_empty() {
        0
    } else {
        let selected = selected_idx.min(item_offsets.len() - 1);
        let start = item_offsets[selected];
        let height = inner.height as usize;
        if start < height {
            0
        } else {
            start.saturating_sub(height.saturating_sub(2))
        }
    };

    let paragraph = Paragraph::new(Text::from(lines))
        .block(block)
        .wrap(Wrap { trim: true })
        .scroll((scroll as u16, 0));

    frame.render_widget(paragraph, popup_area);

    // Scrollbar hint
    let hint = Line::styled(
        "↑/↓ navigate • Enter select • Esc cancel • / filter",
        Style::default().fg(theme.fg_dim),
    );
    let hint_area = Rect {
        y: popup_area.bottom().saturating_sub(2),
        height: 1,
        ..popup_area
    };
    frame.render_widget(Paragraph::new(hint).alignment(Alignment::Center), hint_area);
}

/// Render a help overlay with key bindings.
pub fn render_help(frame: &mut Frame, area: Rect, theme: &Theme) {
    let popup_area = centered_rect(75, 80, area);
    frame.render_widget(Clear, popup_area);

    let block = Block::default()
        .title(" Key Bindings ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.accent));

    let help_text = "Global
  Ctrl+C              Quit
  Tab                 Toggle reasoning panel
  /                   Start command

Normal Mode (vim)
  i / a               Enter insert mode
  j / ↓               Scroll down
  k / ↑               Scroll up
  G                   Jump to top
  g g                 Jump to bottom
  :                   Command line
  ?                   This help

Insert Mode
  Enter               Submit message
  Esc                 Return to normal mode
  Ctrl+A / Home       Start of line
  Ctrl+E / End        End of line
  Ctrl+K              Kill to end
  Ctrl+U              Kill to start
  Ctrl+W              Kill word
  ↑ / ↓               Recall input history

Commands
  /help               Show this help
  /clear              Clear scrollback
  /models             Switch model
  /sessions           Browse sessions
  /autonomy <level>   Set autonomy (ask/read_only/yolo/god)
  /search <query>     Search conversation history
  /reasoning          Toggle reasoning panel
"
    .to_string();

    let paragraph = Paragraph::new(help_text)
        .block(block)
        .wrap(Wrap { trim: false });

    frame.render_widget(paragraph, popup_area);
}

/// Helper: create a centered rect given percentage dimensions.
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
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
