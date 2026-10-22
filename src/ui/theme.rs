// ui/theme.rs — gold-accent theme tokens (see design doc §3).
//
// One accent color, restrained. The "bling" lives in the single highlight.

use ratatui::style::Color;

pub struct Theme {
    pub bg: Color,
    pub fg: Color,
    pub fg_dim: Color,
    pub accent: Color,
    pub accent_dim: Color,
    pub error: Color,
    pub ok: Color,
}

impl Theme {
    pub fn gold() -> Self {
        Self {
            bg: Color::Rgb(24, 24, 24),         // #181818
            fg: Color::Rgb(200, 200, 200),       // #C8C8C8
            fg_dim: Color::Rgb(110, 110, 110),   // #6E6E6E
            accent: Color::Rgb(255, 215, 0),     // #FFD700
            accent_dim: Color::Rgb(140, 110, 0), // #8C6E00
            error: Color::Rgb(220, 80, 80),      // #DC5050
            ok: Color::Rgb(120, 200, 120),       // #78C878
        }
    }
}
