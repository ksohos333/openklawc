//! openklawc-session — session persistence with SQLite.
//!
//! Stores sessions, messages, reasoning steps, and metadata
//! in a local SQLite database for fast querying and recovery.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use okc_core::{ChatMessage, Role};
use rusqlite::{params, Connection};

pub struct SessionStore {
    conn: Connection,
    path: std::path::PathBuf,
}

/// Current schema version. Bump this AND add a migration arm in
/// `init_schema` whenever the schema changes — `CREATE TABLE IF NOT EXISTS`
/// alone is a silent no-op on existing databases.
const SCHEMA_VERSION: i64 = 1;

impl SessionStore {
    /// Open the session database at the default location.
    pub fn new() -> Result<Self> {
        Self::open(&Self::default_path())
    }

    /// Default database path: `<data dir>/openklawc/sessions.db`.
    pub fn default_path() -> std::path::PathBuf {
        dirs::data_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("openklawc")
            .join("sessions.db")
    }

    /// Open (or create) a session database at an explicit path. Creates and
    /// migrates the schema as needed; a pre-existing database is backed up
    /// beside the original before any migration touches it.
    pub fn open(path: &std::path::Path) -> Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let conn = Connection::open(path)
            .with_context(|| format!("opening session db at {}", path.display()))?;
        // The audit subscriber writes on its own connection; a busy timeout
        // keeps concurrent writers from failing with SQLITE_BUSY.
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        let store = Self {
            conn,
            path: path.to_path_buf(),
        };
        store.init_schema()?;
        Ok(store)
    }

    fn init_schema(&self) -> Result<()> {
        let version: i64 = self
            .conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .context("reading schema version")?;

        if version == 0 {
            // Version 0 means "fresh database" OR "pre-versioning legacy
            // database". A legacy db already has tables — back it up before
            // this migration adds the audit tables.
            let legacy: bool = self.conn.query_row(
                "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='sessions'",
                [],
                |r| r.get(0),
            )?;
            if legacy {
                let backup = self.path.with_extension("db.bak-pre-v1");
                std::fs::copy(&self.path, &backup).with_context(|| {
                    format!(
                        "backing up {} before migrating to v{}",
                        self.path.display(),
                        SCHEMA_VERSION
                    )
                })?;
                tracing::info!(
                    "backed up pre-v1 session db to {}",
                    backup.display()
                );
            }
        } else if version < SCHEMA_VERSION {
            // Future migrations go here — same backup-first rule.
            let backup = self
                .path
                .with_extension(format!("db.bak-pre-v{}", SCHEMA_VERSION));
            std::fs::copy(&self.path, &backup).with_context(|| {
                format!(
                    "backing up {} before migrating to v{}",
                    self.path.display(),
                    SCHEMA_VERSION
                )
            })?;
        }

        self.conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                model       TEXT,
                provider    TEXT,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                cwd         TEXT,
                config_json TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                session_id  TEXT NOT NULL,
                message_id  INTEGER NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT,
                tool_calls_json TEXT,
                tool_results_json TEXT,
                timestamp   TEXT NOT NULL,
                PRIMARY KEY (session_id, message_id)
            ) WITHOUT ROWID;

            CREATE TABLE IF NOT EXISTS reasoning_steps (
                session_id  TEXT NOT NULL,
                step_id     INTEGER NOT NULL,
                kind        TEXT NOT NULL,
                content     TEXT,
                decision_json TEXT,
                timestamp   TEXT NOT NULL,
                PRIMARY KEY (session_id, step_id)
            ) WITHOUT ROWID;

            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_reasoning_session ON reasoning_steps(session_id);
            CREATE TABLE IF NOT EXISTS search_index (
                rowid      INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message_id INTEGER NOT NULL,
                role       TEXT,
                content    TEXT
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(
                content, session_id UNINDEXED,
                content='search_index', content_rowid='rowid'
            );

            CREATE TRIGGER IF NOT EXISTS search_insert AFTER INSERT ON search_index BEGIN
                INSERT INTO search(rowid, content, session_id)
                VALUES (new.rowid, new.content, new.session_id);
            END;

            CREATE TRIGGER IF NOT EXISTS search_delete AFTER DELETE ON search_index BEGIN
                INSERT INTO search(search, rowid, content, session_id)
                VALUES ('delete', old.rowid, old.content, old.session_id);
            END;

            CREATE TABLE IF NOT EXISTS audit_events (
                event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                ts           TEXT NOT NULL,
                session_id   TEXT,
                kind         TEXT NOT NULL,
                payload_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_events(session_id);
            "#,
        )?;

        if version != SCHEMA_VERSION {
            self.conn
                .pragma_update(None, "user_version", SCHEMA_VERSION)?;
        }
        Ok(())
    }

    /// Create a new session and return its id.
    pub fn create(
        &mut self,
        name: &str,
        model: Option<&str>,
        provider: Option<&str>,
        cwd: Option<&str>,
    ) -> Result<String> {
        self.create_with_config(name, model, provider, cwd, &SessionConfig::default())
    }

    /// Create a new session with agent-mode / token metadata and return its id.
    pub fn create_with_config(
        &mut self,
        name: &str,
        model: Option<&str>,
        provider: Option<&str>,
        cwd: Option<&str>,
        config: &SessionConfig,
    ) -> Result<String> {
        let id = uuid::Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();
        let config_json = serde_json::to_string(config)?;
        self.conn.execute(
            "INSERT INTO sessions (id, name, model, provider, created_at, updated_at, cwd, config_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?5, ?6, ?7)",
            params![id, name, model, provider, now, cwd, config_json],
        )?;
        Ok(id)
    }

    /// Replace the stored config blob for a session.
    pub fn set_config(&mut self, id: &str, config: &SessionConfig) -> Result<()> {
        let json = serde_json::to_string(config)?;
        self.conn.execute(
            "UPDATE sessions SET config_json = ?1, updated_at = ?2 WHERE id = ?3",
            params![json, Utc::now().to_rfc3339(), id],
        )?;
        Ok(())
    }

    /// Read the stored config blob for a session.
    pub fn get_config(&self, id: &str) -> Result<SessionConfig> {
        let mut stmt = self
            .conn
            .prepare("SELECT config_json FROM sessions WHERE id = ?1")?;
        let raw: Option<String> = stmt.query_row(params![id], |row| row.get(0))?;
        Ok(parse_config(raw))
    }

    /// Rename a session.
    pub fn rename(&mut self, id: &str, name: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE sessions SET name = ?1, updated_at = ?2 WHERE id = ?3",
            params![name, Utc::now().to_rfc3339(), id],
        )?;
        Ok(())
    }

    /// Save or update a session's metadata.
    pub fn update_meta(&mut self, id: &str, name: Option<&str>, model: Option<&str>) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        if let Some(n) = name {
            self.conn.execute(
                "UPDATE sessions SET name = ?1, updated_at = ?2 WHERE id = ?3",
                params![n, now, id],
            )?;
        }
        if let Some(m) = model {
            self.conn.execute(
                "UPDATE sessions SET model = ?1, updated_at = ?2 WHERE id = ?3",
                params![m, now, id],
            )?;
        }
        Ok(())
    }

    /// Append a message to a session.
    pub fn append_message(&mut self, session_id: &str, msg: &PersistentMessage) -> Result<()> {
        self.conn.execute(
            "INSERT INTO messages (session_id, message_id, role, content, tool_calls_json, tool_results_json, timestamp)
             VALUES (?1, (SELECT COALESCE(MAX(message_id), 0) + 1 FROM messages WHERE session_id = ?1), ?2, ?3, ?4, ?5, ?6)",
            params![
                session_id,
                msg.role,
                msg.content,
                msg.tool_calls_json.as_deref(),
                msg.tool_results_json.as_deref(),
                msg.timestamp.to_rfc3339(),
            ],
        )?;

        // Add to full-text search index
        let searchable = format!("{} {}", msg.role, msg.content);
        self.conn.execute(
            "INSERT INTO search_index (session_id, message_id, role, content) VALUES (?1, (SELECT MAX(message_id) FROM messages WHERE session_id = ?1), ?2, ?3)",
            params![session_id, msg.role, searchable],
        )?;

        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "UPDATE sessions SET updated_at = ?1 WHERE id = ?2",
            params![now, session_id],
        )?;

        Ok(())
    }

    /// Store the messages of a turn that are not in the database yet.
    ///
    /// `messages` is the whole in-memory conversation (the system prompt
    /// first), `stored` is how many non-system messages are already saved.
    /// The system prompt is regenerated on every turn and never stored, so the
    /// row count and the index into `messages` drift apart — the count is
    /// used to work out what is new.
    ///
    /// Returns the new number of stored messages.
    pub fn sync_messages(
        &mut self,
        session_id: &str,
        messages: &[ChatMessage],
        stored: usize,
    ) -> Result<usize> {
        let total = messages.iter().filter(|m| m.role != Role::System).count();
        let new_count = total.saturating_sub(stored);
        if new_count > 0 {
            let start = messages.len() - new_count;
            for msg in &messages[start..] {
                if msg.role == Role::System {
                    continue;
                }
                self.append_message(session_id, &to_persistent(msg))?;
            }
        }
        Ok(total)
    }

    /// Get all messages for a session in order.
    pub fn get_messages(&self, session_id: &str) -> Result<Vec<PersistentMessage>> {
        let mut stmt = self.conn.prepare(
            "SELECT role, content, tool_calls_json, tool_results_json, timestamp FROM messages
             WHERE session_id = ?1 ORDER BY message_id ASC",
        )?;

        let rows = stmt.query_map(params![session_id], |row| {
            let ts: String = row.get(4)?;
            Ok(PersistentMessage {
                role: row.get(0)?,
                content: row.get(1)?,
                tool_calls_json: row.get(2)?,
                tool_results_json: row.get(3)?,
                timestamp: DateTime::parse_from_rfc3339(&ts)
                    .map(|d| d.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
            })
        })?;

        let mut msgs = Vec::new();
        for row in rows {
            msgs.push(row?);
        }
        Ok(msgs)
    }

    /// Search sessions by keyword in message content.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let mut stmt = self.conn.prepare(
            "SELECT s.id, s.name, s.model, s.provider, s.created_at, s.updated_at, s.cwd,
                    COUNT(m.message_id) as msg_count, s.config_json
             FROM sessions s
             LEFT JOIN messages m ON s.id = m.session_id
             WHERE s.id IN (
                 SELECT DISTINCT session_id FROM search WHERE search MATCH ?1
             )
             GROUP BY s.id
             ORDER BY s.updated_at DESC
             LIMIT ?2",
        )?;

        let rows = stmt.query_map(params![query, limit as i64], |row| {
            Ok(SessionMeta {
                id: row.get(0)?,
                name: row.get(1)?,
                model: row.get(2)?,
                provider: row.get(3)?,
                created_at: parse_time(row.get(4)?),
                updated_at: parse_time(row.get(5)?),
                cwd: row.get(6)?,
                message_count: row.get(7)?,
                config: parse_config(row.get(8)?),
            })
        })?;

        let mut results = Vec::new();
        for row in rows {
            let meta = row?;
            results.push(SearchResult {
                meta,
                // We need to fetch the matching messages separately
                matching_content: None,
            });
        }
        Ok(results)
    }

    /// List recent sessions.
    pub fn list(&self, limit: usize) -> Result<Vec<SessionMeta>> {
        let mut stmt = self.conn.prepare(
            "SELECT s.id, s.name, s.model, s.provider, s.created_at, s.updated_at, s.cwd,
                    COUNT(m.message_id) as msg_count, s.config_json
             FROM sessions s
             LEFT JOIN messages m ON s.id = m.session_id
             GROUP BY s.id
             ORDER BY s.updated_at DESC
             LIMIT ?1",
        )?;

        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok(SessionMeta {
                id: row.get(0)?,
                name: row.get(1)?,
                model: row.get(2)?,
                provider: row.get(3)?,
                created_at: parse_time(row.get(4)?),
                updated_at: parse_time(row.get(5)?),
                cwd: row.get(6)?,
                message_count: row.get(7)?,
                config: parse_config(row.get(8)?),
            })
        })?;

        let mut sessions = Vec::new();
        for row in rows {
            sessions.push(row?);
        }
        Ok(sessions)
    }

    /// Delete a session and all its messages, reasoning steps and audit
    /// events.
    pub fn delete(&mut self, id: &str) -> Result<()> {
        self.conn
            .execute("DELETE FROM messages WHERE session_id = ?1", params![id])?;
        self.conn.execute(
            "DELETE FROM reasoning_steps WHERE session_id = ?1",
            params![id],
        )?;
        self.conn.execute(
            "DELETE FROM audit_events WHERE session_id = ?1",
            params![id],
        )?;
        self.conn.execute(
            "DELETE FROM search_index WHERE session_id = ?1",
            params![id],
        )?;
        self.conn
            .execute("DELETE FROM sessions WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// Append an audit event — one who/what/when row from the event bus.
    pub fn append_audit(
        &self,
        ts: &str,
        session_id: Option<&str>,
        kind: &str,
        payload: &serde_json::Value,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO audit_events (ts, session_id, kind, payload_json) VALUES (?1, ?2, ?3, ?4)",
            params![ts, session_id, kind, payload.to_string()],
        )?;
        Ok(())
    }

    /// Recent audit events for a session, newest first.
    pub fn audit_for_session(&self, session_id: &str, limit: usize) -> Result<Vec<AuditEvent>> {
        let mut stmt = self.conn.prepare(
            "SELECT ts, kind, payload_json FROM audit_events \
             WHERE session_id = ?1 ORDER BY event_id DESC LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![session_id, limit], |r| {
            Ok(AuditEvent {
                ts: r.get(0)?,
                kind: r.get(1)?,
                payload: serde_json::from_str(&r.get::<_, String>(2)?)
                    .unwrap_or(serde_json::Value::Null),
            })
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    /// Persist one reasoning step — the decision audit trail.
    pub fn append_reasoning_step(
        &self,
        session_id: &str,
        step: &okc_core::ReasoningStep,
    ) -> Result<()> {
        let next: i64 = self.conn.query_row(
            "SELECT COALESCE(MAX(step_id), 0) + 1 FROM reasoning_steps WHERE session_id = ?1",
            params![session_id],
            |r| r.get(0),
        )?;
        let decision = if step.decision.is_default() {
            None
        } else {
            serde_json::to_string(&step.decision).ok()
        };
        self.conn.execute(
            "INSERT INTO reasoning_steps (session_id, step_id, kind, content, decision_json, timestamp) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                session_id,
                next,
                format!("{:?}", step.kind),
                step.content,
                decision,
                step.timestamp.to_rfc3339()
            ],
        )?;
        Ok(())
    }

    /// Reasoning steps for a session, oldest first.
    pub fn reasoning_for_session(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<ReasoningRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT kind, content, decision_json, timestamp FROM reasoning_steps \
             WHERE session_id = ?1 ORDER BY step_id DESC LIMIT ?2",
        )?;
        let mut rows: Vec<ReasoningRow> = stmt
            .query_map(params![session_id, limit], |r| {
                Ok(ReasoningRow {
                    kind: r.get(0)?,
                    content: r.get(1)?,
                    decision_json: r.get(2)?,
                    timestamp: r.get(3)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        rows.reverse();
        Ok(rows)
    }

    /// Export all sessions to JSON (for backup / migration).
    pub fn export(&self) -> Result<String> {
        let sessions = self.list(10000)?;
        #[derive(serde::Serialize)]
        struct Export {
            sessions: Vec<ExportedSession>,
        }
        #[derive(serde::Serialize)]
        struct ExportedSession {
            #[serde(flatten)]
            meta: SessionMeta,
            messages: Vec<PersistentMessage>,
        }

        let mut exported = Vec::with_capacity(sessions.len());
        for meta in sessions {
            let messages = self.get_messages(&meta.id)?;
            exported.push(ExportedSession { meta, messages });
        }

        Ok(serde_json::to_string_pretty(&Export {
            sessions: exported,
        })?)
    }
}

/// Convert an in-memory chat message into its stored form.
fn to_persistent(msg: &ChatMessage) -> PersistentMessage {
    let role = match msg.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };
    // Tool turns carry their payload in `tool_results`, not `content`.
    let content = if !msg.content.is_empty() {
        msg.content.clone()
    } else {
        msg.tool_results
            .as_ref()
            .map(|r| serde_json::to_string(r).unwrap_or_default())
            .unwrap_or_default()
    };
    PersistentMessage {
        role: role.to_string(),
        content,
        tool_calls_json: msg
            .tool_calls
            .as_ref()
            .map(|c| serde_json::to_string(c).unwrap_or_default()),
        tool_results_json: msg
            .tool_results
            .as_ref()
            .map(|r| serde_json::to_string(r).unwrap_or_default()),
        timestamp: Utc::now(),
    }
}

/// Decode the `config_json` blob. A malformed blob degrades to defaults
/// rather than failing the whole listing.
fn parse_config(raw: Option<String>) -> SessionConfig {
    raw.and_then(|r| serde_json::from_str(&r).ok())
        .unwrap_or_default()
}

fn parse_time(s: Option<String>) -> DateTime<Utc> {
    s.and_then(|t| DateTime::parse_from_rfc3339(&t).ok())
        .map(|d| d.with_timezone(&Utc))
        .unwrap_or_else(Utc::now)
}

/// A message stored in the database.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PersistentMessage {
    pub role: String,
    pub content: String,
    pub tool_calls_json: Option<String>,
    pub tool_results_json: Option<String>,
    pub timestamp: DateTime<Utc>,
}

/// Per-session extras stored in the `sessions.config_json` column.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct SessionConfig {
    /// Working mode the session was started in ("build", "plan", "secure").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_mode: Option<String>,
    /// Human-friendly title, when it differs from the session name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Tokens sent to the model across the session.
    #[serde(default)]
    pub tokens_in: u64,
    /// Tokens produced by the model across the session.
    #[serde(default)]
    pub tokens_out: u64,
    /// Accumulated cost in USD, when a price is known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
}

/// Session metadata for listing.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionMeta {
    pub id: String,
    pub name: String,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub cwd: Option<String>,
    pub message_count: i64,
    /// Agent mode / token counters / cost, decoded from `config_json`.
    pub config: SessionConfig,
}

/// One search result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub meta: SessionMeta,
    pub matching_content: Option<String>,
}

/// One stored audit event (who/what/when, from the event bus).
#[derive(Debug, Clone, serde::Serialize)]
pub struct AuditEvent {
    pub ts: String,
    pub kind: String,
    pub payload: serde_json::Value,
}

/// One stored reasoning step (the decision audit trail).
#[derive(Debug, Clone, serde::Serialize)]
pub struct ReasoningRow {
    pub kind: String,
    pub content: String,
    pub decision_json: Option<String>,
    pub timestamp: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn in_memory_store() -> SessionStore {
        let conn = Connection::open_in_memory().unwrap();
        let store = SessionStore {
            conn,
            path: std::path::PathBuf::from(":memory:"),
        };
        store.init_schema().unwrap();
        store
    }

    /// Unique temp dir for file-backed tests.
    fn temp_db(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "okc-session-test-{}-{}",
            std::process::id(),
            name
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("sessions.db")
    }

    #[test]
    fn test_create_and_list() {
        let mut store = in_memory_store();
        store
            .create(
                "test session",
                Some("MiniMax-M3"),
                Some("minimax"),
                Some("/home/dev"),
            )
            .unwrap();

        let sessions = store.list(10).unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].name, "test session");
        assert_eq!(sessions[0].model, Some("MiniMax-M3".to_string()));
        assert_eq!(sessions[0].message_count, 0);
    }

    #[test]
    fn test_append_message() {
        let mut store = in_memory_store();
        let id = store.create("msg test", None, None, None).unwrap();

        store
            .append_message(
                &id,
                &PersistentMessage {
                    role: "user".to_string(),
                    content: "Hello world".to_string(),
                    tool_calls_json: None,
                    tool_results_json: None,
                    timestamp: Utc::now(),
                },
            )
            .unwrap();

        let msgs = store.get_messages(&id).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content, "Hello world");
    }

    #[test]
    fn test_config_round_trip() {
        let mut store = in_memory_store();
        let config = SessionConfig {
            agent_mode: Some("build".into()),
            title: Some("My chat".into()),
            tokens_in: 100,
            tokens_out: 250,
            cost_usd: Some(0.0123),
        };
        let id = store
            .create_with_config("chat", Some("MiniMax-M3"), Some("minimax"), None, &config)
            .unwrap();

        let loaded = store.get_config(&id).unwrap();
        assert_eq!(loaded.agent_mode.as_deref(), Some("build"));
        assert_eq!(loaded.title.as_deref(), Some("My chat"));
        assert_eq!(loaded.tokens_in, 100);
        assert_eq!(loaded.tokens_out, 250);
        assert_eq!(loaded.cost_usd, Some(0.0123));

        // The config must also survive the listing query.
        let sessions = store.list(10).unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].config.agent_mode.as_deref(), Some("build"));
        assert_eq!(sessions[0].config.tokens_out, 250);
    }

    #[test]
    fn test_set_config_updates_existing_row() {
        let mut store = in_memory_store();
        let id = store.create("chat", None, None, None).unwrap();
        assert!(store.get_config(&id).unwrap().agent_mode.is_none());

        let mut config = store.get_config(&id).unwrap();
        config.agent_mode = Some("plan".into());
        config.tokens_in = 42;
        store.set_config(&id, &config).unwrap();

        let loaded = store.get_config(&id).unwrap();
        assert_eq!(loaded.agent_mode.as_deref(), Some("plan"));
        assert_eq!(loaded.tokens_in, 42);
    }

    #[test]
    fn test_malformed_config_degrades_to_default() {
        let mut store = in_memory_store();
        let id = store.create("chat", None, None, None).unwrap();
        store
            .conn
            .execute(
                "UPDATE sessions SET config_json = 'not json' WHERE id = ?1",
                params![id],
            )
            .unwrap();
        let loaded = store.get_config(&id).unwrap();
        assert_eq!(loaded.tokens_in, 0);
        assert!(loaded.agent_mode.is_none());
    }

    #[test]
    fn test_rename() {
        let mut store = in_memory_store();
        let id = store.create("old name", None, None, None).unwrap();
        store.rename(&id, "new name").unwrap();
        let sessions = store.list(10).unwrap();
        assert_eq!(sessions[0].name, "new name");
    }

    #[test]
    fn test_delete_removes_session() {
        let mut store = in_memory_store();
        let id = store.create("doomed", None, None, None).unwrap();
        store
            .append_message(
                &id,
                &PersistentMessage {
                    role: "user".into(),
                    content: "hello".into(),
                    tool_calls_json: None,
                    tool_results_json: None,
                    timestamp: Utc::now(),
                },
            )
            .unwrap();
        store.delete(&id).unwrap();
        assert!(store.list(10).unwrap().is_empty());
        assert!(store.get_messages(&id).unwrap().is_empty());
    }

    #[test]
    fn test_sync_messages_handles_the_system_prompt() {
        let mut store = in_memory_store();
        let id = store.create("chat", None, None, None).unwrap();

        // A turn always starts with a regenerated system prompt.
        let messages = vec![
            ChatMessage::system("you are openklawc"),
            ChatMessage::user("hello"),
            ChatMessage::assistant("hi there"),
        ];

        let stored = store.sync_messages(&id, &messages, 0).unwrap();
        assert_eq!(stored, 2);
        let rows = store.get_messages(&id).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].role, "user");
        assert_eq!(rows[1].role, "assistant");

        // Syncing again must not duplicate anything.
        let stored = store.sync_messages(&id, &messages, stored).unwrap();
        assert_eq!(stored, 2);
        assert_eq!(store.get_messages(&id).unwrap().len(), 2);
    }

    #[test]
    fn test_sync_messages_after_a_resume() {
        let mut store = in_memory_store();
        let id = store.create("chat", None, None, None).unwrap();

        // First turn.
        let first = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("q1"),
            ChatMessage::assistant("a1"),
        ];
        let stored = store.sync_messages(&id, &first, 0).unwrap();
        assert_eq!(stored, 2);

        // Resume: the history comes back without a system prompt.
        let resumed: Vec<ChatMessage> = store
            .get_messages(&id)
            .unwrap()
            .iter()
            .map(|m| ChatMessage::user(m.content.clone()))
            .collect();
        assert_eq!(resumed.len(), 2);

        // The next turn puts the system prompt back at the front.
        let second = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("q1"),
            ChatMessage::assistant("a1"),
            ChatMessage::user("q2"),
            ChatMessage::assistant("a2"),
        ];
        let stored = store.sync_messages(&id, &second, stored).unwrap();
        assert_eq!(stored, 4);

        let rows = store.get_messages(&id).unwrap();
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[2].content, "q2");
        assert_eq!(rows[3].content, "a2");
    }

    #[test]
    fn test_sync_messages_stores_tool_payloads() {
        let mut store = in_memory_store();
        let id = store.create("chat", None, None, None).unwrap();

        let tool_msg = ChatMessage::tool_results(vec![okc_core::NativeToolResult {
            call_id: "c1".into(),
            content: "the output".into(),
            is_error: false,
        }]);
        let messages = vec![ChatMessage::system("sys"), tool_msg];

        store.sync_messages(&id, &messages, 0).unwrap();
        let rows = store.get_messages(&id).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].role, "tool");
        assert!(rows[0].content.contains("the output"));
    }

    #[test]
    fn test_search() {
        let mut store = in_memory_store();
        let id = store.create("searchable", None, None, None).unwrap();

        store
            .append_message(
                &id,
                &PersistentMessage {
                    role: "user".to_string(),
                    content: "How do I implement a binary search tree?".to_string(),
                    tool_calls_json: None,
                    tool_results_json: None,
                    timestamp: Utc::now(),
                },
            )
            .unwrap();

        let results = store.search("binary search tree", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].meta.name, "searchable");
    }

    #[test]
    fn open_at_a_path_sets_schema_version() {
        let path = temp_db("version");
        {
            let store = SessionStore::open(&path).unwrap();
            let v: i64 = store
                .conn
                .query_row("PRAGMA user_version", [], |r| r.get(0))
                .unwrap();
            assert_eq!(v, SCHEMA_VERSION);
        }
        // Reopening an already-migrated db must be a no-op (no backup file).
        let _ = SessionStore::open(&path).unwrap();
        assert!(!path.with_extension("db.bak-pre-v1").exists());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn legacy_database_is_backed_up_before_migration() {
        let path = temp_db("legacy");

        // Hand-build a legacy (pre-versioning) database with data.
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE sessions (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, model TEXT,
                    provider TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    cwd TEXT, config_json TEXT
                );
                INSERT INTO sessions VALUES ('s1', 'legacy chat', NULL, NULL,
                    '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', NULL, NULL);",
            )
            .unwrap();
        }

        let store = SessionStore::open(&path).unwrap();
        let backup = path.with_extension("db.bak-pre-v1");
        assert!(backup.exists(), "legacy db must be backed up");

        // Data survived, schema version stamped.
        assert_eq!(store.list(10).unwrap().len(), 1);
        let v: i64 = store
            .conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(v, SCHEMA_VERSION);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&backup);
    }

    #[test]
    fn audit_events_round_trip() {
        let store = in_memory_store();
        let ts = "2026-10-26T12:00:00Z";
        store
            .append_audit(ts, Some("s1"), "ToolResult", &serde_json::json!({"name": "read_file"}))
            .unwrap();
        store
            .append_audit(ts, Some("s1"), "FileMutation", &serde_json::json!({"path": "a.rs"}))
            .unwrap();
        store
            .append_audit(ts, None, "AutonomyChanged", &serde_json::json!({"level": "Yolo"}))
            .unwrap();

        let events = store.audit_for_session("s1", 10).unwrap();
        assert_eq!(events.len(), 2, "session-less events excluded: {:?}", events);
        // Newest first.
        assert_eq!(events[0].kind, "FileMutation");
        assert_eq!(events[0].payload["path"], "a.rs");
    }

    #[test]
    fn reasoning_steps_round_trip_with_decisions() {
        let store = in_memory_store();
        let step = okc_core::ReasoningStep {
            timestamp: Utc::now(),
            kind: okc_core::ReasoningKind::Decide,
            content: "I will run cargo test".into(),
            tool_call: None,
            tool_result: None,
            decision: okc_core::DecisionMetadata {
                confidence: Some("high".into()),
                chosen_approach: Some("test-first".into()),
                alternatives: Some(vec!["read-code-first".into()]),
                key_tradeoff: Some("speed".into()),
                is_backtrack: false,
                iteration: 2,
            },
        };
        store.append_reasoning_step("s1", &step).unwrap();
        store
            .append_reasoning_step(
                "s1",
                &okc_core::ReasoningStep {
                    timestamp: Utc::now(),
                    kind: okc_core::ReasoningKind::Plan,
                    content: "plain step".into(),
                    tool_call: None,
                    tool_result: None,
                    decision: okc_core::DecisionMetadata::default(),
                },
            )
            .unwrap();

        let rows = store.reasoning_for_session("s1", 10).unwrap();
        assert_eq!(rows.len(), 2);
        // Oldest first.
        assert_eq!(rows[0].kind, "Decide");
        assert_eq!(rows[1].kind, "Plan");
        let decision: serde_json::Value =
            serde_json::from_str(&rows[0].decision_json.clone().unwrap()).unwrap();
        assert_eq!(decision["confidence"], "high");
        assert_eq!(decision["iteration"], 2);
        // The default-decision step stores no blob.
        assert!(rows[1].decision_json.is_none());
    }

    #[test]
    fn delete_cascades_to_audit_and_reasoning() {
        let store = in_memory_store();
        let step = okc_core::ReasoningStep {
            timestamp: Utc::now(),
            kind: okc_core::ReasoningKind::Plan,
            content: "x".into(),
            tool_call: None,
            tool_result: None,
            decision: okc_core::DecisionMetadata::default(),
        };
        store.append_audit("t", Some("s1"), "Log", &serde_json::json!({})).unwrap();
        store.append_reasoning_step("s1", &step).unwrap();

        let mut store = store;
        // 's1' has no sessions row; delete() is by-session-id and cleans
        // the dependents regardless.
        store.delete("s1").unwrap();
        assert!(store.audit_for_session("s1", 10).unwrap().is_empty());
        assert!(store.reasoning_for_session("s1", 10).unwrap().is_empty());
    }
}
