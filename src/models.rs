// models.rs — the model registry.
//
// Mirrors the `provider` section of your `~/.config/opencode/opencode.jsonc`
// (verified 2026-10-16). Twenty-one models across eight providers. This is
// the list opencode sees. When openklawc grows a real LLM client (Phase 5+),
// this is the set of `provider.model_id` strings the user can pick via `/model`.
//
// To auto-load from the opencode config instead of hardcoding, replace
// `Models::default()` with code that reads `~/.config/opencode/opencode.jsonc`
// and walks the `provider` map. Note: opencode config is JSONC (JSON with
// comments + trailing commas), so you'll need a JSONC parser, not plain
// `serde_json`.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Model {
    pub provider: &'static str,
    pub id: &'static str,
    pub display_name: &'static str,
    pub reasoning: bool,
    pub vision: bool,
    pub context: Option<&'static str>,
}

impl Model {
    pub fn qualified_id(&self) -> String {
        format!("{}/{}", self.provider, self.id)
    }
}

#[derive(Debug, Clone)]
pub struct Registry {
    by_id: HashMap<String, Model>,
    order: Vec<String>,
    default: String,
}

impl Registry {
    pub fn default_registry() -> Self {
        let raw: &[Model] = &[
            // local — this box
            m("local", "Qwen3-14B-Q4_K_M", "Qwen3 14B Q4_K_M (local, Zone 0)",
               false, false, None),

            // macstudio — Ollama on the mac studio
            m("macstudio", "huihui_ai/Qwen3.8-abliterated:27b-q8_0_L",
               "Qwen3.8 Abliterated 27B (macstudio)", false, false, None),
            m("macstudio", "huihui_ai/qwen3.5-abliterated:122b",
               "Qwen3.5 Abliterated 122B (macstudio heavy)", false, false, None),
            m("macstudio", "huihui_ai/gemma-4-abliterated:31b",
               "Gemma 4 Abliterated 31B (macstudio)", false, false, None),
            m("macstudio", "huihui-qwen3-vl-235b-abliterated:latest",
               "Qwen3-VL 235B Abliterated (macstudio vision)", false, true, None),

            // ollama — this box
            m("ollama", "dmtx/qwen3.5-9b-abliterated:latest",
               "Qwen3.5-9B Abliterated (local)", false, false, None),
            m("ollama", "qwen2.5vl:3b",
               "Qwen2.5-VL 3B (local vision)", false, true, None),

            // zai — Z.AI GLM coding
            m("zai", "glm-5.3", "GLM 5.3", false, false, None),

            // minimax — M Plan Go
            m("minimax", "MiniMax-M3", "MiniMax M3 (Go plan, 1M ctx)",
               true, false, Some("1M")),
            m("minimax", "MiniMax-M2.7", "MiniMax M2.7 (Go plan)",
               false, false, None),

            // devpass
            m("devpass", "claude-sonnet-4", "Claude Sonnet 4",
               false, false, None),
            m("devpass", "gpt-5", "GPT-5", false, false, None),

            // kimi — Moonshot
            m("kimi", "kimi-k2.7-code", "Kimi K2.7 Code", true, false, None),
            m("kimi", "kimi-k2.6", "Kimi K2.6", true, false, None),

        // typesafe.ai — Jev (System One) — typed, structured decisions, not chat
        m("typesafe", "jev", "Jev (System One, structured)", false, false, None),

        // chutes — TEE-backed
        m("chutes", "moonshotai/Kimi-K3-TEE", "Kimi K3 (TEE)", false, false, None),
            m("chutes", "deepseek-ai/DeepSeek-V4-Flash-0731-TEE",
               "DeepSeek V4 Flash (TEE, 1M ctx)", false, false, Some("1M")),
            m("chutes", "zai-org/GLM-5.2-TEE", "GLM 5.2 (TEE, 1M ctx)",
               false, false, Some("1M")),
            m("chutes", "Qwen/Qwen3.5-397B-A17B-TEE", "Qwen3.5 397B (TEE)",
               false, false, None),
            m("chutes", "Qwen/Qwen3-235B-A22B-Thinking-2507-TEE",
               "Qwen3 235B Thinking (TEE)", true, false, None),
            m("chutes", "Qwen/Qwen3.8-27B-TEE", "Qwen3.8 27B (TEE, vision)",
               false, true, None),
        ];

        let mut by_id: HashMap<String, Model> = HashMap::new();
        let mut order: Vec<String> = Vec::new();
        for m in raw {
            order.push(m.qualified_id());
            by_id.insert(m.qualified_id(), m.clone());
        }
        let default = "macstudio/huihui_ai/Qwen3.8-abliterated:27b-q8_0_L".to_string();
        Self { by_id, order, default }
    }

    pub fn all(&self) -> &[Model] {
        // SAFETY-ish: the by_id HashMap's iteration order is unspecified, so
        // we keep the `order` Vec in sync for stable display.
        // Returning a Vec slice would require owned storage; for v0.1 the
        // caller uses `iter()` over `self.order` keyed lookups instead.
        &[]
    }

    pub fn iter(&self) -> impl Iterator<Item = &Model> {
        self.order.iter().filter_map(|qid| self.by_id.get(qid))
    }

    pub fn get(&self, qualified: &str) -> Option<&Model> {
        self.by_id.get(qualified)
    }

    pub fn default_id(&self) -> &str {
        &self.default
    }
}

const fn m(
    provider: &'static str,
    id: &'static str,
    display_name: &'static str,
    reasoning: bool,
    vision: bool,
    context: Option<&'static str>,
) -> Model {
    Model { provider, id, display_name, reasoning, vision, context }
}
