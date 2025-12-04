//! Anthropic API 端点处理器 (/v1/messages)

use crate::backends::{self, Backend};
use crate::config::Config;
use crate::error::{ProxyError, ProxyResult};
use crate::models::anthropic;
use crate::router::{RequestFormat, RoutingDecision};
use crate::transform;
use axum::{response::Response, Extension};
use reqwest::Client;
use std::sync::Arc;

/// Anthropic API 端点处理器
pub async fn anthropic_handler(
    Extension(config): Extension<Arc<Config>>,
    Extension(client): Extension<Client>,
    body: axum::body::Bytes,
) -> ProxyResult<Response> {
    // 解析请求为 JSON Value（保留原始结构）
    let raw_json: serde_json::Value = serde_json::from_slice(&body).map_err(|e| {
        tracing::error!("Failed to parse request as JSON: {}", e);
        tracing::debug!("Raw request body: {}", String::from_utf8_lossy(&body));
        ProxyError::Transform(format!("Invalid JSON: {}", e))
    })?;

    if config.debug && config.log_raw_json {
        tracing::debug!(
            "Raw request JSON: {}",
            serde_json::to_string_pretty(&raw_json).unwrap_or_default()
        );
    }

    // 提取必要字段用于路由决策
    let model = raw_json
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let is_streaming = raw_json
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    tracing::debug!("Received Anthropic request for model: {}", model);
    tracing::debug!("Streaming: {}", is_streaming);

    // 路由决策
    let decision = RoutingDecision::decide(RequestFormat::Anthropic, model, &config)?;

    tracing::debug!(
        "Routing decision: backend={:?}, needs_transform={}, direction={:?}",
        decision.backend,
        decision.needs_transform,
        decision.transform_direction
    );

    if config.verbose {
        tracing::trace!(
            "Incoming Anthropic request: {}",
            serde_json::to_string_pretty(&raw_json).unwrap_or_default()
        );
    }

    match (decision.backend, decision.needs_transform) {
        // 完全透传到 Anthropic（不解析结构体，直接转发原始 body）
        (Backend::Anthropic, false) => {
            backends::anthropic::forward_raw_request(config, client, body, is_streaming).await
        }
        // 需要转换，先解析为结构体
        (Backend::OpenAI | Backend::Upstream, true) => {
            let req: anthropic::AnthropicRequest =
                serde_json::from_value(raw_json.clone()).map_err(|e| {
                    tracing::error!("Failed to deserialize request: {}", e);
                    ProxyError::Transform(format!("Failed to deserialize: {}", e))
                })?;

            let openai_req = transform::anthropic_to_openai(req, &config)?;

            if config.verbose {
                tracing::trace!(
                    "Transformed OpenAI request: {}",
                    serde_json::to_string_pretty(&openai_req).unwrap_or_default()
                );
            }

            if is_streaming {
                backends::upstream::handle_streaming(config, client, openai_req, decision.backend).await
            } else {
                backends::upstream::handle_non_streaming(config, client, openai_req, decision.backend).await
            }
        }
        _ => Err(ProxyError::Internal("Invalid routing decision".into())),
    }
}
