//! JavaScript snippets evaluated in the page.

/// A helper that resolves a selector to an element.
///
/// The selector may be a snapshot ref (`e3`), a CSS selector, or plain text
/// matched against the visible labels of interactive elements.
const RESOLVE: &str = r#"
const __resolve = (sel) => {
  if (!sel) return null;
  const norm = String(sel).trim();
  if (/^e\d+$/.test(norm)) {
    const byRef = document.querySelector('[data-openklawc-ref="' + norm + '"]');
    if (byRef) return byRef;
  }
  try {
    const byCss = document.querySelector(norm);
    if (byCss) return byCss;
  } catch (e) { /* not a valid CSS selector */ }
  const lower = norm.toLowerCase();
  const nodes = Array.from(document.querySelectorAll(
    'a,button,input,textarea,select,summary,[role=button],[role=link],[onclick]'
  ));
  for (const node of nodes) {
    const hay = (
      (node.innerText || '') + ' ' +
      (node.value || '') + ' ' +
      (node.placeholder || '') + ' ' +
      (node.getAttribute('aria-label') || '') + ' ' +
      (node.getAttribute('title') || '') + ' ' +
      (node.getAttribute('name') || '')
    ).toLowerCase();
    if (hay.includes(lower)) return node;
  }
  return null;
};
"#;

/// Build an expression that resolves `selector` and describes the element.
pub fn describe(selector: &str) -> String {
    let sel = serde_json::to_string(selector).unwrap_or_else(|_| "\"\"".into());
    format!(
        r#"(() => {{
{resolve}
  const el = __resolve({sel});
  if (!el) return {{ ok: false, error: 'element not found: ' + {sel} }};
  el.scrollIntoView({{ block: 'center', inline: 'center' }});
  const rect = el.getBoundingClientRect();
  return {{
    ok: true,
    tag: el.tagName.toLowerCase(),
    text: ((el.innerText || '') + ' ' + (el.value || '') + '').trim().replace(/\s+/g, ' ').slice(0, 200),
    x: rect.left + rect.width / 2,
    y: rect.top + rect.height / 2,
    width: rect.width,
    height: rect.height
  }};
}})()"#,
        resolve = RESOLVE,
        sel = sel
    )
}

/// Build an expression that focuses the element and clears its value.
pub fn focus(selector: &str) -> String {
    let sel = serde_json::to_string(selector).unwrap_or_else(|_| "\"\"".into());
    format!(
        r#"(() => {{
{resolve}
  const el = __resolve({sel});
  if (!el) return {{ ok: false, error: 'element not found: ' + {sel} }};
  el.focus();
  if ('value' in el) el.value = '';
  return {{ ok: true, tag: el.tagName.toLowerCase() }};
}})()"#,
        resolve = RESOLVE,
        sel = sel
    )
}

/// Build an expression that selects an option in a `<select>`.
pub fn select_option(selector: &str, value: &str) -> String {
    let sel = serde_json::to_string(selector).unwrap_or_else(|_| "\"\"".into());
    let val = serde_json::to_string(value).unwrap_or_else(|_| "\"\"".into());
    format!(
        r#"(() => {{
{resolve}
  const el = __resolve({sel});
  if (!el) return {{ ok: false, error: 'element not found: ' + {sel} }};
  const options = Array.from(el.options || []).map(o => ({{ value: o.value, text: o.text }}));
  const wanted = {val};
  const match = options.find(o => o.value === wanted || o.text === wanted || o.text.toLowerCase() === wanted.toLowerCase());
  if (!match) return {{ ok: false, error: 'option not found: ' + wanted, options }};
  el.value = match.value;
  el.dispatchEvent(new Event('input', {{ bubbles: true }}));
  el.dispatchEvent(new Event('change', {{ bubbles: true }}));
  return {{ ok: true, value: match.value, text: match.text }};
}})()"#,
        resolve = RESOLVE,
        sel = sel,
        val = val
    )
}

/// Build an expression that produces a text snapshot of the page.
pub fn snapshot() -> String {
    r#"(() => {
  const out = [];
  let counter = 0;
  const interesting = ['a', 'button', 'input', 'select', 'textarea', 'summary', 'label', 'option'];
  const walk = (element, depth) => {
    if (depth > 15 || out.length > 400) return;
    for (const child of element.children) {
      const tag = child.tagName.toLowerCase();
      if (tag === 'script' || tag === 'style' || tag === 'noscript' || tag === 'template') continue;
      const role = child.getAttribute('role');
      const interactive = interesting.includes(tag) || role === 'button' || role === 'link' ||
        child.hasAttribute('onclick') || child.getAttribute('tabindex') !== null;
      const ownText = Array.from(child.childNodes)
        .filter(node => node.nodeType === 3)
        .map(node => node.textContent)
        .join(' ')
        .replace(/\s+/g, ' ')
        .trim();
      const text = ((child.innerText || '') || ownText).trim().replace(/\s+/g, ' ');
      if (interactive) {
        counter += 1;
        const ref = 'e' + counter;
        child.setAttribute('data-openklawc-ref', ref);
        const attrs = ['aria-label', 'placeholder', 'value', 'type', 'name', 'href', 'id', 'alt', 'title', 'role']
          .map(name => child.getAttribute(name))
          .filter(Boolean)
          .map(value => value.length > 60 ? value.slice(0, 60) + '…' : value);
        out.push('  '.repeat(depth) + '[' + ref + '] <' + tag + '> ' + text.slice(0, 120) +
          (attrs.length ? ' {' + attrs.join(', ') + '}' : ''));
      } else if (text && child.children.length === 0) {
        out.push('  '.repeat(depth) + '<text> ' + text.slice(0, 120));
      }
      walk(child, depth + 1);
    }
  };
  if (document.body) walk(document.body, 0);
  return {
    title: document.title,
    url: location.href,
    outline: out.join('\n')
  };
})()"#
    .to_string()
}

/// Build an expression that searches the page for text.
pub fn find_text(text: &str) -> String {
    let needle = serde_json::to_string(text).unwrap_or_else(|_| "\"\"".into());
    format!(
        r#"(() => {{
  const needle = {needle}.toLowerCase();
  const hits = [];
  const walker = document.createTreeWalker(document.body || document, NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode())) {{
    const value = (node.textContent || '').trim();
    if (!value) continue;
    const index = value.toLowerCase().indexOf(needle);
    if (index === -1) continue;
    const element = node.parentElement;
    if (!element) continue;
    const rect = element.getBoundingClientRect();
    hits.push({{
      tag: element.tagName.toLowerCase(),
      text: value.slice(0, 200),
      offset: index,
      x: rect.left + rect.width / 2,
      y: rect.top + rect.height / 2
    }});
    if (hits.length >= 25) break;
  }}
  return {{ count: hits.length, hits }};
}})()"#
    )
}

/// Build an expression that waits for text to appear on the page.
pub fn wait_for_text(text: &str) -> String {
    let needle = serde_json::to_string(text).unwrap_or_else(|_| "\"\"".into());
    format!(
        r#"(() => {{
  const needle = {needle};
  const body = (document.body || {{}}).innerText || '';
  if (body.includes(needle)) return {{ found: true }};
  for (const element of document.querySelectorAll('*')) {{
    const value = (element.innerText || '').trim();
    if (value === needle || (element.children.length === 0 && value.includes(needle))) {{
      return {{ found: true }};
    }}
  }}
  return {{ found: false }};
}})()"#
    )
}
