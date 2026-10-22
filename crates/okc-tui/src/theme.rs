use ratatui::style::Color;

#[derive(Debug, Clone, Copy)]
pub struct Theme {
    pub bg: Color,
    pub fg: Color,
    pub fg_dim: Color,
    pub accent: Color,
    pub accent_dim: Color,
    pub error: Color,
    pub ok: Color,
    pub warning: Color,
    pub border: Color,
}

impl Theme {
    pub fn gold() -> Self {
        Self {
            bg: Color::Rgb(24, 24, 24),
            fg: Color::Rgb(200, 200, 200),
            fg_dim: Color::Rgb(110, 110, 110),
            accent: Color::Rgb(255, 215, 0),
            accent_dim: Color::Rgb(140, 110, 0),
            error: Color::Rgb(220, 80, 80),
            ok: Color::Rgb(120, 200, 120),
            warning: Color::Rgb(220, 180, 80),
            border: Color::Rgb(60, 60, 60),
        }
    }

    pub fn cyberpunk() -> Self {
        Self {
            bg: Color::Rgb(10, 10, 20),
            fg: Color::Rgb(220, 220, 255),
            fg_dim: Color::Rgb(100, 100, 140),
            accent: Color::Rgb(0, 255, 200),
            accent_dim: Color::Rgb(0, 140, 120),
            error: Color::Rgb(255, 60, 100),
            ok: Color::Rgb(60, 255, 120),
            warning: Color::Rgb(255, 200, 60),
            border: Color::Rgb(40, 40, 80),
        }
    }
}
