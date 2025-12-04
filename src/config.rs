use anyhow::Result;
use std::{env, fmt, path::PathBuf};

/// 路由模式
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum RoutingMode {
    /// 仅转换模式（默认，兼容现有行为）
    #[default]
    Transform,
    /// Anthropic 透传模式
    Passthrough,
    /// 自动路由模式
    Auto,
    /// 完整网关模式
    Gateway,
}

impl fmt::Display for RoutingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RoutingMode::Transform => write!(f, "Transform"),
            RoutingMode::Passthrough => write!(f, "Passthrough"),
            RoutingMode::Auto => write!(f, "Auto"),
            RoutingMode::Gateway => write!(f, "Gateway"),
        }
    }
}

impl RoutingMode {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "passthrough" | "anthropic" => RoutingMode::Passthrough,
            "auto" => RoutingMode::Auto,
            "gateway" => RoutingMode::Gateway,
            _ => RoutingMode::Transform,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub port: u16,

    // 路由配置
    pub routing_mode: RoutingMode,

    // Anthropic 后端配置
    pub anthropic_base_url: Option<String>,
    pub anthropic_api_key: Option<String>,

    // OpenAI 后端配置
    pub openai_base_url: Option<String>,
    pub openai_api_key: Option<String>,

    // 转换后端配置（兼容现有）
    pub base_url: Option<String>,
    pub api_key: Option<String>,

    // 模型路由配置
    pub reasoning_model: Option<String>,
    pub completion_model: Option<String>,

    // 日志配置
    pub debug: bool,
    pub verbose: bool,
    pub log_raw_json: bool,
}

impl Config {
    fn load_dotenv(custom_path: Option<PathBuf>) -> Option<PathBuf> {
        if let Some(path) = custom_path {
            if path.exists() {
                if let Ok(_) = dotenvy::from_path(&path) {
                    return Some(path);
                }
            }
            eprintln!("⚠️  WARNING: Custom config file not found: {}", path.display());
        }

        if let Ok(path) = dotenvy::dotenv() {
            return Some(path);
        }

        if let Some(home) = env::var("HOME").ok() {
            let home_config = PathBuf::from(home).join(".anthropic-proxy.env");
            if home_config.exists() {
                if let Ok(_) = dotenvy::from_path(&home_config) {
                    return Some(home_config);
                }
            }
        }

        let etc_config = PathBuf::from("/etc/anthropic-proxy/.env");
        if etc_config.exists() {
            if let Ok(_) = dotenvy::from_path(&etc_config) {
                return Some(etc_config);
            }
        }

        None
    }

    pub fn from_env() -> Result<Self> {
        Self::from_env_with_path(None)
    }

    pub fn from_env_with_path(custom_path: Option<PathBuf>) -> Result<Self> {
        if let Some(path) = Self::load_dotenv(custom_path) {
            eprintln!("📄 Loaded config from: {}", path.display());
        } else {
            eprintln!("ℹ️  No .env file found, using environment variables only");
        }

        let port = env::var("PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(3000);

        // 路由模式
        let routing_mode = env::var("ROUTING_MODE")
            .map(|s| RoutingMode::from_str(&s))
            .unwrap_or_default();

        // Anthropic 后端配置
        let anthropic_base_url = env::var("ANTHROPIC_BASE_URL").ok();
        let anthropic_api_key = env::var("ANTHROPIC_API_KEY").ok();

        // OpenAI 后端配置
        let openai_base_url = env::var("OPENAI_BASE_URL").ok();
        let openai_api_key = env::var("OPENAI_API_KEY").ok();

        // 转换后端配置（兼容现有）
        let base_url = env::var("UPSTREAM_BASE_URL")
            .or_else(|_| env::var("ANTHROPIC_PROXY_BASE_URL"))
            .ok();

        let api_key = env::var("UPSTREAM_API_KEY")
            .or_else(|_| env::var("OPENROUTER_API_KEY"))
            .ok()
            .filter(|k| !k.is_empty());

        // 验证配置
        match routing_mode {
            RoutingMode::Transform => {
                if base_url.is_none() {
                    return Err(anyhow::anyhow!(
                        "UPSTREAM_BASE_URL is required in Transform mode.\n\
                        Set it to your OpenAI-compatible endpoint.\n\
                        Examples:\n\
                          - OpenRouter: https://openrouter.ai/api\n\
                          - OpenAI: https://api.openai.com\n\
                          - Local: http://localhost:11434"
                    ));
                }
            }
            RoutingMode::Passthrough => {
                if anthropic_base_url.is_none() || anthropic_api_key.is_none() {
                    return Err(anyhow::anyhow!(
                        "ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY are required in Passthrough mode.\n\
                        Example:\n\
                          ANTHROPIC_BASE_URL=https://api.anthropic.com\n\
                          ANTHROPIC_API_KEY=sk-ant-xxxxx"
                    ));
                }
            }
            RoutingMode::Auto | RoutingMode::Gateway => {
                // Auto/Gateway 模式至少需要配置一个后端
                let has_anthropic = anthropic_base_url.is_some() && anthropic_api_key.is_some();
                let has_openai = openai_base_url.is_some() && openai_api_key.is_some();
                let has_upstream = base_url.is_some();

                if !has_anthropic && !has_openai && !has_upstream {
                    return Err(anyhow::anyhow!(
                        "At least one backend must be configured in {} mode.\n\
                        Configure one or more of:\n\
                          - Anthropic: ANTHROPIC_BASE_URL + ANTHROPIC_API_KEY\n\
                          - OpenAI: OPENAI_BASE_URL + OPENAI_API_KEY\n\
                          - Upstream: UPSTREAM_BASE_URL + UPSTREAM_API_KEY",
                        routing_mode
                    ));
                }
            }
        }

        let reasoning_model = env::var("REASONING_MODEL").ok();
        let completion_model = env::var("COMPLETION_MODEL").ok();

        let debug = env::var("DEBUG")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        let verbose = env::var("VERBOSE")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        let log_raw_json = env::var("LOG_RAW_JSON")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        // 警告检查
        if let Some(ref url) = base_url {
            if url.ends_with("/v1") {
                eprintln!("⚠️  WARNING: UPSTREAM_BASE_URL ends with '/v1'");
                eprintln!("   This will result in URLs like: {}/v1/chat/completions", url);
                eprintln!("   Consider removing '/v1' from UPSTREAM_BASE_URL");
            }
        }

        Ok(Config {
            port,
            routing_mode,
            anthropic_base_url,
            anthropic_api_key,
            openai_base_url,
            openai_api_key,
            base_url,
            api_key,
            reasoning_model,
            completion_model,
            debug,
            verbose,
            log_raw_json,
        })
    }

    pub fn chat_completions_url(&self) -> String {
        if let Some(ref url) = self.base_url {
            format!("{}/v1/chat/completions", url.trim_end_matches('/'))
        } else {
            String::new()
        }
    }

    pub fn anthropic_messages_url(&self) -> String {
        if let Some(ref url) = self.anthropic_base_url {
            format!("{}/v1/messages", url.trim_end_matches('/'))
        } else {
            String::new()
        }
    }

    pub fn openai_chat_completions_url(&self) -> String {
        if let Some(ref url) = self.openai_base_url {
            format!("{}/v1/chat/completions", url.trim_end_matches('/'))
        } else {
            String::new()
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_routing_mode_from_str_transform() {
        assert_eq!(RoutingMode::from_str("transform"), RoutingMode::Transform);
        assert_eq!(RoutingMode::from_str("TRANSFORM"), RoutingMode::Transform);
    }

    #[test]
    fn test_routing_mode_from_str_passthrough() {
        assert_eq!(RoutingMode::from_str("passthrough"), RoutingMode::Passthrough);
        assert_eq!(RoutingMode::from_str("anthropic"), RoutingMode::Passthrough);
    }

    #[test]
    fn test_routing_mode_from_str_auto() {
        assert_eq!(RoutingMode::from_str("auto"), RoutingMode::Auto);
        assert_eq!(RoutingMode::from_str("AUTO"), RoutingMode::Auto);
    }

    #[test]
    fn test_routing_mode_from_str_gateway() {
        assert_eq!(RoutingMode::from_str("gateway"), RoutingMode::Gateway);
        assert_eq!(RoutingMode::from_str("GATEWAY"), RoutingMode::Gateway);
    }

    #[test]
    fn test_routing_mode_from_str_default() {
        assert_eq!(RoutingMode::from_str("unknown"), RoutingMode::Transform);
        assert_eq!(RoutingMode::from_str(""), RoutingMode::Transform);
    }

    #[test]
    fn test_routing_mode_display() {
        assert_eq!(format!("{}", RoutingMode::Transform), "Transform");
        assert_eq!(format!("{}", RoutingMode::Passthrough), "Passthrough");
        assert_eq!(format!("{}", RoutingMode::Auto), "Auto");
        assert_eq!(format!("{}", RoutingMode::Gateway), "Gateway");
    }

    #[test]
    fn test_chat_completions_url() {
        let config = Config {
            port: 3000,
            routing_mode: RoutingMode::Transform,
            anthropic_base_url: None,
            anthropic_api_key: None,
            openai_base_url: None,
            openai_api_key: None,
            base_url: Some("https://api.example.com".to_string()),
            api_key: None,
            reasoning_model: None,
            completion_model: None,
            debug: false,
            verbose: false,
            log_raw_json: false,
        };

        assert_eq!(config.chat_completions_url(), "https://api.example.com/v1/chat/completions");
    }

    #[test]
    fn test_chat_completions_url_with_trailing_slash() {
        let config = Config {
            port: 3000,
            routing_mode: RoutingMode::Transform,
            anthropic_base_url: None,
            anthropic_api_key: None,
            openai_base_url: None,
            openai_api_key: None,
            base_url: Some("https://api.example.com/".to_string()),
            api_key: None,
            reasoning_model: None,
            completion_model: None,
            debug: false,
            verbose: false,
            log_raw_json: false,
        };

        assert_eq!(config.chat_completions_url(), "https://api.example.com/v1/chat/completions");
    }

    #[test]
    fn test_anthropic_messages_url() {
        let config = Config {
            port: 3000,
            routing_mode: RoutingMode::Passthrough,
            anthropic_base_url: Some("https://api.anthropic.com".to_string()),
            anthropic_api_key: Some("test".to_string()),
            openai_base_url: None,
            openai_api_key: None,
            base_url: None,
            api_key: None,
            reasoning_model: None,
            completion_model: None,
            debug: false,
            verbose: false,
            log_raw_json: false,
        };

        assert_eq!(config.anthropic_messages_url(), "https://api.anthropic.com/v1/messages");
    }

    #[test]
    fn test_openai_chat_completions_url() {
        let config = Config {
            port: 3000,
            routing_mode: RoutingMode::Auto,
            anthropic_base_url: None,
            anthropic_api_key: None,
            openai_base_url: Some("https://api.openai.com".to_string()),
            openai_api_key: Some("test".to_string()),
            base_url: None,
            api_key: None,
            reasoning_model: None,
            completion_model: None,
            debug: false,
            verbose: false,
            log_raw_json: false,
        };

        assert_eq!(config.openai_chat_completions_url(), "https://api.openai.com/v1/chat/completions");
    }
}
