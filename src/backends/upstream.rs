//! 通用上游后端
//!
//! 处理 Anthropic → OpenAI 转换后的请求

use crate::config::Config;
use crate::error::{ProxyError, ProxyResult};
use crate::models::openai as models;
use crate::router::Backend;
use crate::streaming::openai_to_anthropic::create_stream;
use crate::transform;
use axum::{
    body::Body,
    http::{HeaderMap, HeaderValue},
    response::{IntoResponse, Response},
    Json,
};
use reqwest::Client;
use std::sync::Arc;
use std::time::Duration;

/// 处理非流式请求 (A→O)
pub async fn handle_non_streaming(
    config: Arc<Config>,
    client: Client,
    openai_req: models::OpenAIRequest,
    backend: Backend,
) -> ProxyResult<Response> {
    let (url, api_key) = get_backend_config(&config, backend)?;

    tracing::debug!("Sending non-streaming request to {}", url);

    let mut req_builder = client
        .post(&url)
        .json(&openai_req)
        .timeout(Duration::from_secs(300));

    if let Some(key) = &api_key {
        req_builder = req_builder.header("Authorization", format!("Bearer {}", key));
    }

    let response = req_builder.send().await?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
        tracing::error!("Upstream error ({}): {}", status, error_text);
        return Err(ProxyError::Upstream(format!(
            "Upstream returned {}: {}",
            status, error_text
        )));
    }

    let openai_resp: models::OpenAIResponse = response.json().await?;

    if config.verbose {
        tracing::trace!(
            "Received OpenAI response: {}",
            serde_json::to_string_pretty(&openai_resp).unwrap_or_default()
        );
    }

    let anthropic_resp = transform::openai_to_anthropic(openai_resp)?;

    if config.verbose {
        tracing::trace!(
            "Transformed Anthropic response: {}",
            serde_json::to_string_pretty(&anthropic_resp).unwrap_or_default()
        );
    }

    Ok(Json(anthropic_resp).into_response())
}

/// 处理流式请求 (A→O)
pub async fn handle_streaming(
    config: Arc<Config>,
    client: Client,
    openai_req: models::OpenAIRequest,
    backend: Backend,
) -> ProxyResult<Response> {
    let (url, api_key) = get_backend_config(&config, backend)?;

    tracing::debug!("Sending streaming request to {}", url);

    let mut req_builder = client
        .post(&url)
        .json(&openai_req)
        .timeout(Duration::from_secs(300));

    if let Some(key) = &api_key {
        req_builder = req_builder.header("Authorization", format!("Bearer {}", key));
    }

    let response = req_builder.send().await?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
        tracing::error!("Upstream error ({}) from {}: {}", status, url, error_text);
        return Err(ProxyError::Upstream(format!(
            "Upstream returned {} from {}: {}",
            status, url, error_text
        )));
    }

    let stream = response.bytes_stream();
    let sse_stream = create_stream(stream);

    let mut headers = HeaderMap::new();
    headers.insert(
        "Content-Type",
        HeaderValue::from_static("text/event-stream"),
    );
    headers.insert("Cache-Control", HeaderValue::from_static("no-cache"));
    headers.insert("Connection", HeaderValue::from_static("keep-alive"));

    Ok((headers, Body::from_stream(sse_stream)).into_response())
}

/// 获取后端配置
fn get_backend_config(config: &Config, backend: Backend) -> ProxyResult<(String, Option<String>)> {
    match backend {
        Backend::OpenAI => Ok((
            config.openai_chat_completions_url(),
            config.openai_api_key.clone(),
        )),
        Backend::Upstream => Ok((
            config.chat_completions_url(),
            config.api_key.clone(),
        )),
        _ => Err(ProxyError::Internal("Invalid backend for A→O".into())),
    }
}
