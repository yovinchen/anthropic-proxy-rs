//! OpenAI API 端点处理器 (/v1/chat/completions)

use crate::backends::{self, Backend};
use crate::config::Config;
use crate::error::{ProxyError, ProxyResult};
use crate::models::openai;
use crate::router::{RequestFormat, RoutingDecision};
use crate::transform;
use axum::{response::Response, Extension};
use reqwest::Client;
use std::sync::Arc;

/// OpenAI API 端点处理器
pub async fn openai_handler(
    Extension(config): Extension<Arc<Config>>,
    Extension(client): Extension<Client>,
    body: axum::body::Bytes,
) -> ProxyResult<Response> {
    // 解析请求
    let raw_json: serde_json::Value = serde_json::from_slice(&body).map_err(|e| {
        tracing::error!("Failed to parse request as JSON: {}", e);
        ProxyError::Transform(format!("Invalid JSON: {}", e))
    })?;

    if config.debug && config.log_raw_json {
        tracing::debug!(
            "Raw OpenAI request JSON: {}",
            serde_json::to_string_pretty(&raw_json).unwrap_or_default()
        );
    }

    let req: openai::OpenAIRequest = serde_json::from_value(raw_json.clone()).map_err(|e| {
        tracing::error!("Failed to deserialize OpenAI request: {}", e);
        ProxyError::Transform(format!("Failed to deserialize: {}", e))
    })?;

    let is_streaming = req.stream.unwrap_or(false);

    tracing::debug!("Received OpenAI request for model: {}", req.model);
    tracing::debug!("Streaming: {}", is_streaming);

    // 路由决策
    let decision = RoutingDecision::decide(RequestFormat::OpenAI, &req.model, &config)?;

    tracing::debug!(
        "Routing decision: backend={:?}, needs_transform={}, direction={:?}",
        decision.backend,
        decision.needs_transform,
        decision.transform_direction
    );

    if config.verbose {
        tracing::trace!(
            "Incoming OpenAI request: {}",
            serde_json::to_string_pretty(&req).unwrap_or_default()
        );
    }

    match (decision.backend, decision.needs_transform) {
        // 透传到 OpenAI
        (Backend::OpenAI, false) => {
            backends::openai::forward_request(config, client, req, is_streaming).await
        }
        // 转换后发送到 Anthropic
        (Backend::Anthropic, true) => {
            let anthropic_req = transform::openai_to_anthropic_request(req, &config)?;

            if config.verbose {
                tracing::trace!(
                    "Transformed Anthropic request: {}",
                    serde_json::to_string_pretty(&anthropic_req).unwrap_or_default()
                );
            }

            if is_streaming {
                backends::anthropic::handle_transformed_streaming(config, client, anthropic_req).await
            } else {
                backends::anthropic::handle_transformed_non_streaming(config, client, anthropic_req).await
            }
        }
        _ => Err(ProxyError::Internal("Invalid routing decision".into())),
    }
}
