//! openklawc-session — session persistence.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use okc_core::{ChatMessage, Session};
use serde::{Deserialize, Serialize};

pub struct SessionStore {
    dir: PathBuf,
}

impl SessionStore {
    pub fn new() -> Result<Self> {
        let dir = dirs::data_dir()
            .ok_or_else(|| anyhow::anyhow!("no data dir"))?
            .join("openklawc")
            .join("sessions");
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    pub fn save(&self, session: &Session) -> Result<()> {
        let path = self.dir.join(format!("{}.json", session.id));
        let json = serde_json::to_string_pretty(session)?;
        std::fs::write(&path, json)?;
        Ok(())
    }

    pub fn load(&self, id: &str) -> Result<Session> {
        let path = self.dir.join(format!("{}.json", id));
        let json = std::fs::read_to_string(&path)?;
        let session: Session = serde_json::from_str(&json)?;
        Ok(session)
    }

    pub fn list(&self) -> Result<Vec<SessionMeta>> {
        let mut sessions = Vec::new();
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            if entry.path().extension().and_then(|e| e.to_str()) == Some("json") {
                if let Ok(text) = std::fs::read_to_string(entry.path()) {
                    if let Ok(session) = serde_json::from_str::<Session>(&text) {
                        sessions.push(SessionMeta {
                            id: session.id.to_string(),
                            name: session.name,
                            updated_at: session.updated_at,
                            message_count: session.messages.len(),
                        });
                    }
                }
            }
        }
        sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(sessions)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    pub id: String,
    pub name: String,
    pub updated_at: DateTime<Utc>,
    pub message_count: usize,
}
