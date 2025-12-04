//! 路由决策模块
//!
//! 根据请求格式、模型名称和配置决定如何路由请求

use crate::config::{Config, RoutingMode};
use crate::error::ProxyError;

/// 目标后端
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Backend {
    /// Anthropic 官方 API
    Anthropic,
    /// OpenAI 官方 API
    OpenAI,
    /// 通用上游（用于转换模式）
    Upstream,
}

/// 请求格式
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RequestFormat {
    /// Anthropic Messages API 格式
    Anthropic,
    /// OpenAI Chat Completions API 格式
    OpenAI,
}

/// 转换方向
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransformDirection {
    /// Anthropic → OpenAI
    AnthropicToOpenAI,
    /// OpenAI → Anthropic
    OpenAIToAnthropic,
}

/// 路由决策结果
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// 目标后端
    pub backend: Backend,
    /// 是否需要转换
    pub needs_transform: bool,
    /// 转换方向
    pub transform_direction: Option<TransformDirection>,
}

impl RoutingDecision {
    /// 根据请求格式、模型名称和配置决定路由
    pub fn decide(
        request_format: RequestFormat,
        model: &str,
        config: &Config,
    ) -> Result<Self, ProxyError> {
        match config.routing_mode {
            RoutingMode::Transform => Self::decide_transform_mode(request_format, config),
            RoutingMode::Passthrough => Self::decide_passthrough_mode(request_format, config),
            RoutingMode::Auto | RoutingMode::Gateway => {
                Self::decide_auto_mode(request_format, model, config)
            }
        }
    }

    /// Transform 模式：仅支持 Anthropic 请求，转换为 OpenAI 格式发送到上游
    fn decide_transform_mode(
        request_format: RequestFormat,
        config: &Config,
    ) -> Result<Self, ProxyError> {
        match request_format {
            RequestFormat::Anthropic => {
                if config.base_url.is_none() {
                    return Err(ProxyError::Config(
                        "UPSTREAM_BASE_URL is required in Transform mode".into(),
                    ));
                }
                Ok(Self {
                    backend: Backend::Upstream,
                    needs_transform: true,
                    transform_direction: Some(TransformDirection::AnthropicToOpenAI),
                })
            }
            RequestFormat::OpenAI => Err(ProxyError::Transform(
                "OpenAI endpoint is not supported in Transform mode. \
                Please use /v1/messages or change ROUTING_MODE to 'auto' or 'gateway'."
                    .into(),
            )),
        }
    }

    /// Passthrough 模式：仅支持 Anthropic 请求，直接透传到 Anthropic API
    fn decide_passthrough_mode(
        request_format: RequestFormat,
        config: &Config,
    ) -> Result<Self, ProxyError> {
        match request_format {
            RequestFormat::Anthropic => {
                if config.anthropic_base_url.is_none() || config.anthropic_api_key.is_none() {
                    return Err(ProxyError::Config(
                        "ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY are required in Passthrough mode"
                            .into(),
                    ));
                }
                Ok(Self {
                    backend: Backend::Anthropic,
                    needs_transform: false,
                    transform_direction: None,
                })
            }
            RequestFormat::OpenAI => Err(ProxyError::Transform(
                "OpenAI endpoint is not supported in Passthrough mode. \
                Please use /v1/messages or change ROUTING_MODE to 'auto' or 'gateway'."
                    .into(),
            )),
        }
    }

    /// Auto/Gateway 模式：根据模型名称自动路由
    fn decide_auto_mode(
        request_format: RequestFormat,
        model: &str,
        config: &Config,
    ) -> Result<Self, ProxyError> {
        let target_backend = Self::infer_backend_from_model(model);

        match (request_format, target_backend) {
            // Anthropic 请求 → Anthropic 后端（透传）
            (RequestFormat::Anthropic, Backend::Anthropic) => {
                if config.anthropic_base_url.is_none() || config.anthropic_api_key.is_none() {
                    return Err(ProxyError::Config(
                        "ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY are required for Claude models"
                            .into(),
                    ));
                }
                Ok(Self {
                    backend: Backend::Anthropic,
                    needs_transform: false,
                    transform_direction: None,
                })
            }

            // OpenAI 请求 → OpenAI 后端（透传）
            (RequestFormat::OpenAI, Backend::OpenAI) => {
                if config.openai_base_url.is_none() || config.openai_api_key.is_none() {
                    return Err(ProxyError::Config(
                        "OPENAI_BASE_URL and OPENAI_API_KEY are required for OpenAI models".into(),
                    ));
                }
                Ok(Self {
                    backend: Backend::OpenAI,
                    needs_transform: false,
                    transform_direction: None,
                })
            }

            // Anthropic 请求 → OpenAI 后端（需要 A→O 转换）
            (RequestFormat::Anthropic, Backend::OpenAI) => {
                // 优先使用 OpenAI 后端，否则使用通用上游
                let backend = if config.openai_base_url.is_some() && config.openai_api_key.is_some()
                {
                    Backend::OpenAI
                } else if config.base_url.is_some() {
                    Backend::Upstream
                } else {
                    return Err(ProxyError::Config(
                        "No OpenAI-compatible backend configured. \
                        Set OPENAI_BASE_URL + OPENAI_API_KEY or UPSTREAM_BASE_URL."
                            .into(),
                    ));
                };

                Ok(Self {
                    backend,
                    needs_transform: true,
                    transform_direction: Some(TransformDirection::AnthropicToOpenAI),
                })
            }

            // OpenAI 请求 → Anthropic 后端（需要 O→A 转换）
            (RequestFormat::OpenAI, Backend::Anthropic) => {
                if config.anthropic_base_url.is_none() || config.anthropic_api_key.is_none() {
                    return Err(ProxyError::Config(
                        "ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY are required for Claude models"
                            .into(),
                    ));
                }
                Ok(Self {
                    backend: Backend::Anthropic,
                    needs_transform: true,
                    transform_direction: Some(TransformDirection::OpenAIToAnthropic),
                })
            }

            _ => Err(ProxyError::Internal("Invalid routing combination".into())),
        }
    }

    /// 根据模型名称推断目标后端
    fn infer_backend_from_model(model: &str) -> Backend {
        let model_lower = model.to_lowercase();

        // Anthropic 模型模式
        if model_lower.starts_with("claude")
            || model_lower.contains("anthropic/")
            || model_lower.contains("anthropic-")
        {
            return Backend::Anthropic;
        }

        // OpenAI 模型模式
        if model_lower.starts_with("gpt")
            || model_lower.starts_with("o1")
            || model_lower.starts_with("o3")
            || model_lower.contains("openai/")
            || model_lower.starts_with("text-")
            || model_lower.starts_with("davinci")
            || model_lower.starts_with("curie")
            || model_lower.starts_with("babbage")
            || model_lower.starts_with("ada")
        {
            return Backend::OpenAI;
        }

        // 默认使用 OpenAI（兼容性考虑）
        Backend::OpenAI
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_transform_config() -> Config {
        Config {
            port: 3000,
            routing_mode: RoutingMode::Transform,
            anthropic_base_url: None,
            anthropic_api_key: None,
            openai_base_url: None,
            openai_api_key: None,
            base_url: Some("https://api.example.com".to_string()),
            api_key: Some("test-key".to_string()),
            reasoning_model: None,
            completion_model: None,
            debug: false,
            verbose: false,
            log_raw_json: false,
        }
    }

    fn create_passthrough_config() -> Config {
        Config {
            port: 3000,
            routing_mode: RoutingMode::Passthrough,
            anthropic_base_url: Some("https://api.anthropic.com".to_string()),
            anthropic_api_key: Some("test-key".to_string()),
            openai_base_url: None,
            openai_api_key: None,
            base_url: None,
            api_key: None,
            reasoning_model: None,
            completion_model: None,
            debug: false,
            verbose: false,
            log_raw_json: false,
        }
    }

    fn create_auto_config() -> Config {
        Config {
            port: 3000,
            routing_mode: RoutingMode::Auto,
            anthropic_base_url: Some("https://api.anthropic.com".to_string()),
            anthropic_api_key: Some("test-key".to_string()),
            openai_base_url: Some("https://api.openai.com".to_string()),
            openai_api_key: Some("test-key".to_string()),
            base_url: None,
            api_key: None,
            reasoning_model: None,
            completion_model: None,
            debug: false,
            verbose: false,
            log_raw_json: false,
        }
    }

    #[test]
    fn test_infer_backend_claude() {
        assert_eq!(
            RoutingDecision::infer_backend_from_model("claude-3-5-sonnet-20241022"),
            Backend::Anthropic
        );
        assert_eq!(
            RoutingDecision::infer_backend_from_model("anthropic/claude-3-opus"),
            Backend::Anthropic
        );
    }

    #[test]
    fn test_infer_backend_openai() {
        assert_eq!(
            RoutingDecision::infer_backend_from_model("gpt-4"),
            Backend::OpenAI
        );
        assert_eq!(
            RoutingDecision::infer_backend_from_model("o1-preview"),
            Backend::OpenAI
        );
        assert_eq!(
            RoutingDecision::infer_backend_from_model("openai/gpt-4-turbo"),
            Backend::OpenAI
        );
    }

    #[test]
    fn test_infer_backend_default() {
        assert_eq!(
            RoutingDecision::infer_backend_from_model("unknown-model"),
            Backend::OpenAI
        );
    }

    #[test]
    fn test_transform_mode_anthropic_request() {
        let config = create_transform_config();
        let decision = RoutingDecision::decide(RequestFormat::Anthropic, "claude-3", &config).unwrap();
        
        assert_eq!(decision.backend, Backend::Upstream);
        assert!(decision.needs_transform);
        assert_eq!(decision.transform_direction, Some(TransformDirection::AnthropicToOpenAI));
    }

    #[test]
    fn test_transform_mode_openai_request_fails() {
        let config = create_transform_config();
        let result = RoutingDecision::decide(RequestFormat::OpenAI, "gpt-4", &config);
        
        assert!(result.is_err());
    }

    #[test]
    fn test_passthrough_mode_anthropic_request() {
        let config = create_passthrough_config();
        let decision = RoutingDecision::decide(RequestFormat::Anthropic, "claude-3", &config).unwrap();
        
        assert_eq!(decision.backend, Backend::Anthropic);
        assert!(!decision.needs_transform);
        assert_eq!(decision.transform_direction, None);
    }

    #[test]
    fn test_passthrough_mode_openai_request_fails() {
        let config = create_passthrough_config();
        let result = RoutingDecision::decide(RequestFormat::OpenAI, "gpt-4", &config);
        
        assert!(result.is_err());
    }

    #[test]
    fn test_auto_mode_anthropic_to_anthropic() {
        let config = create_auto_config();
        let decision = RoutingDecision::decide(RequestFormat::Anthropic, "claude-3", &config).unwrap();
        
        assert_eq!(decision.backend, Backend::Anthropic);
        assert!(!decision.needs_transform);
    }

    #[test]
    fn test_auto_mode_openai_to_openai() {
        let config = create_auto_config();
        let decision = RoutingDecision::decide(RequestFormat::OpenAI, "gpt-4", &config).unwrap();
        
        assert_eq!(decision.backend, Backend::OpenAI);
        assert!(!decision.needs_transform);
    }

    #[test]
    fn test_auto_mode_anthropic_to_openai() {
        let config = create_auto_config();
        let decision = RoutingDecision::decide(RequestFormat::Anthropic, "gpt-4", &config).unwrap();
        
        assert_eq!(decision.backend, Backend::OpenAI);
        assert!(decision.needs_transform);
        assert_eq!(decision.transform_direction, Some(TransformDirection::AnthropicToOpenAI));
    }

    #[test]
    fn test_auto_mode_openai_to_anthropic() {
        let config = create_auto_config();
        let decision = RoutingDecision::decide(RequestFormat::OpenAI, "claude-3", &config).unwrap();
        
        assert_eq!(decision.backend, Backend::Anthropic);
        assert!(decision.needs_transform);
        assert_eq!(decision.transform_direction, Some(TransformDirection::OpenAIToAnthropic));
    }

    #[test]
    fn test_infer_backend_o3_model() {
        assert_eq!(
            RoutingDecision::infer_backend_from_model("o3-mini"),
            Backend::OpenAI
        );
    }

    #[test]
    fn test_infer_backend_text_model() {
        assert_eq!(
            RoutingDecision::infer_backend_from_model("text-davinci-003"),
            Backend::OpenAI
        );
    }

    #[test]
    fn test_infer_backend_davinci() {
        assert_eq!(
            RoutingDecision::infer_backend_from_model("davinci"),
            Backend::OpenAI
        );
    }
}
