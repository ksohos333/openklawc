//! Rendering and interaction tests for the chat picker overlay.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{backend::TestBackend, Terminal};

use okc_tui::app::{App, AppEvent, Overlay, PendingConfirm};
use okc_tui::chat::{ChatMode, ChatPickerFocus, ChatSession};

fn session(id: &str, title: &str) -> ChatSession {
    ChatSession {
        id: id.into(),
        title: title.into(),
        agent_mode: "build".into(),
        model: Some("MiniMax-M3".into()),
        provider: Some("minimax".into()),
        cwd: Some("/home/dev/project".into()),
        updated_at: "2026-10-26 06:34".into(),
        created_at: "2026-10-25 09:00".into(),
        tokens_in: 199_802,
        tokens_out: 80_722,
        cost_usd: Some(1.972),
        message_count: 42,
    }
}

fn app_with_sessions() -> App {
    let mut app = App::new();
    app.open_chat_picker();
    app.chat_picker.set_sessions(vec![
        session("a", "Alpha build"),
        session("b", "Beta plan"),
        session("c", "Gamma"),
    ]);
    app
}

fn render_to_string(app: &mut App, width: u16, height: u16) -> String {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|frame| app.render(frame, frame.area()))
        .unwrap();
    let buffer = terminal.backend().buffer().clone();
    buffer
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>()
}

#[test]
fn picker_renders_title_and_rows() {
    let mut app = app_with_sessions();
    let text = render_to_string(&mut app, 120, 40);
    assert!(text.contains("OpenCode Chat Picker"), "title missing");
    assert!(text.contains("Alpha build"), "first session missing");
    assert!(text.contains("Beta plan"), "second session missing");
    assert!(text.contains("New Chat"), "new-chat button missing");
    assert!(text.contains("DELETE"), "delete button missing");
    assert!(text.contains("Search sessions"), "search box missing");
}

#[test]
fn picker_renders_detail_pane() {
    let mut app = app_with_sessions();
    let text = render_to_string(&mut app, 140, 40);
    assert!(text.contains("Title:"), "detail label missing");
    assert!(text.contains("MiniMax-M3"), "model missing");
    assert!(text.contains("…/dev/project"), "dir missing");
    assert!(text.contains("199802"), "token count missing");
    assert!(text.contains("1.9720"), "cost missing");
}

#[test]
fn picker_search_filters_rows() {
    let mut app = app_with_sessions();
    // 's' focuses the search box.
    let event = app.handle_key(KeyEvent::new(KeyCode::Char('s'), KeyModifiers::NONE));
    assert!(event.is_none());
    assert_eq!(app.chat_picker.focus, ChatPickerFocus::Search);

    for ch in "beta".chars() {
        app.handle_key(KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE));
    }
    assert_eq!(app.chat_picker.visible.len(), 1);
    assert_eq!(app.chat_picker.selected_session().unwrap().id, "b");

    let text = render_to_string(&mut app, 120, 40);
    assert!(text.contains("Beta plan"));
    assert!(!text.contains("Gamma"), "non-matching row still shown");
}

#[test]
fn picker_navigates_and_resumes() {
    let mut app = app_with_sessions();
    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
    assert_eq!(app.chat_picker.selected_session().unwrap().id, "b");

    match app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)) {
        Some(AppEvent::ResumeSession(id)) => assert_eq!(id, "b"),
        other => panic!("expected ResumeSession, got {:?}", other),
    }
    assert_eq!(app.overlay, Overlay::None);
}

#[test]
fn delete_needs_two_presses() {
    let mut app = app_with_sessions();

    // First press only arms the deletion.
    assert!(app
        .handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::NONE))
        .is_none());
    assert_eq!(
        app.chat_picker.pending_delete.as_deref(),
        Some("Alpha build")
    );
    assert!(app
        .chat_picker
        .notice
        .as_deref()
        .unwrap_or("")
        .contains("Press d again"));

    // The second press carries it out.
    match app.handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::NONE)) {
        Some(AppEvent::DeleteSession(id)) => assert_eq!(id, "a"),
        other => panic!("expected DeleteSession, got {:?}", other),
    }
    assert!(app.chat_picker.pending_delete.is_none());
}

#[test]
fn another_key_cancels_a_pending_delete() {
    let mut app = app_with_sessions();
    app.handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::NONE));
    assert!(app.chat_picker.pending_delete.is_some());

    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
    assert!(app.chat_picker.pending_delete.is_none());
}

#[test]
fn picker_shortcuts_emit_events() {
    let mut app = app_with_sessions();

    match app.handle_key(KeyEvent::new(KeyCode::Char('b'), KeyModifiers::NONE)) {
        Some(AppEvent::CreateSession(mode)) => assert_eq!(mode, ChatMode::Build),
        other => panic!("expected CreateSession, got {:?}", other),
    }

    match app.handle_key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE)) {
        Some(AppEvent::CreateSession(mode)) => assert_eq!(mode, ChatMode::Plan),
        other => panic!("expected CreateSession, got {:?}", other),
    }

    match app.handle_key(KeyEvent::new(KeyCode::Char('S'), KeyModifiers::NONE)) {
        Some(AppEvent::CreateSession(mode)) => assert_eq!(mode, ChatMode::Secure),
        other => panic!("expected CreateSession, got {:?}", other),
    }

    // Two presses: the first arms, the second deletes.
    app.handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::NONE));
    match app.handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::NONE)) {
        Some(AppEvent::DeleteSession(id)) => assert_eq!(id, "a"),
        other => panic!("expected DeleteSession, got {:?}", other),
    }
}

#[test]
fn escape_leaves_search_before_closing() {
    let mut app = app_with_sessions();
    app.handle_key(KeyEvent::new(KeyCode::Char('s'), KeyModifiers::NONE));
    assert_eq!(app.chat_picker.focus, ChatPickerFocus::Search);

    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
    assert_eq!(app.chat_picker.focus, ChatPickerFocus::List);
    assert_eq!(app.overlay, Overlay::ChatPicker);

    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
    assert_eq!(app.overlay, Overlay::None);
}

#[test]
fn empty_picker_shows_a_hint() {
    let mut app = App::new();
    app.open_chat_picker();
    app.chat_picker.set_sessions(Vec::new());
    let text = render_to_string(&mut app, 100, 30);
    assert!(text.contains("No chats yet"));
}

#[test]
fn pending_approval_swallows_the_keyboard() {
    let mut app = App::new();
    app.pending_confirm = Some(PendingConfirm {
        name: "write_file".into(),
        arguments: "{}".into(),
    });

    // A normal key must not reach the input.
    assert!(app
        .handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE))
        .is_none());
    assert!(app.input.is_empty());

    match app.handle_key(KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE)) {
        Some(AppEvent::ConfirmTool(true)) => {}
        other => panic!("expected ConfirmTool(true), got {:?}", other),
    }
    assert!(app.pending_confirm.is_none());
}

#[test]
fn declining_an_approval_emits_false() {
    let mut app = App::new();
    app.pending_confirm = Some(PendingConfirm {
        name: "shell".into(),
        arguments: "{}".into(),
    });
    match app.handle_key(KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE)) {
        Some(AppEvent::ConfirmTool(false)) => {}
        other => panic!("expected ConfirmTool(false), got {:?}", other),
    }
}

#[test]
fn picker_state_is_reset_when_reopened() {
    let mut app = app_with_sessions();
    app.handle_key(KeyEvent::new(KeyCode::Char('s'), KeyModifiers::NONE));
    app.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE));
    assert_eq!(app.chat_picker.query, "x");

    app.open_chat_picker();
    assert_eq!(app.chat_picker.query, "");
    assert_eq!(app.chat_picker.focus, ChatPickerFocus::List);
    assert_eq!(app.chat_picker.visible.len(), 3);
}
