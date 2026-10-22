use crate::llm::config::{ModelConfig, OpencodeConfig, ProviderConfig};
use anyhow::{anyhow, Result};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    OpenAICompatible,
    Anthropic,
    Unknown,
}

/// Qualified model id: "provider/model_name".
#[derive(Debug, Clone)]
pub struct QualifiedModel {
    pub provider: String,
    pub model_id: String,
    pub display_name: String,
    pub base_url: String,
    pub api_key: String,
    pub reasoning: bool,
    pub vision: bool,
    pub kind: ProviderKind,
}

pub struct ProviderRegistry {
    models: HashMap<String, QualifiedModel>,
    default: String,
}

impl ProviderRegistry {
    pub fn from_config(cfg: &OpencodeConfig) -> Self {
        let mut models = HashMap::new();
        for (prov_name, prov) in &cfg.provider {
            let kind = if prov.npm.contains("anthropic") {
                ProviderKind::Anthropic
            } else if prov.npm.contains("openai") {
                ProviderKind::OpenAICompatible
            } else {
                ProviderKind::Unknown
            };
            for (model_id, model_cfg) in &prov.models {
                let qid = format!("{}/{}", prov_name, model_id);
                models.insert(
                    qid.clone(),
                    QualifiedModel {
                        provider: prov_name.clone(),
                        model_id: model_id.clone(),
                        display_name: model_cfg.name.clone(),
                        base_url: prov.options.base_url.clone(),
                        api_key: prov.options.api_key.clone(),
                        reasoning: model_cfg.reasoning,
                        vision: model_cfg.vision,
                        kind,
                    },
                );
            }
        }
        Self {
            default: cfg.default_model_id(),
            models,
        }
    }

    pub fn get(&self, qid: &str) -> Option<&QualifiedModel> {
        self.models.get(qid)
    }

    pub fn default_model(&self) -> Option<&QualifiedModel> {
        self.get(&self.default)
    }

    pub fn list(&self) -> Vec<(&String, &QualifiedModel)> {
        self.models.iter().collect()
    }

    pub fn default_id(&self) -> &str {
        &self.default
    }
}
