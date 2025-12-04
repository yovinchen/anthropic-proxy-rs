//! Anthropic 后端
//!
//! 处理与 Anthropic API 的通信

use crate::config::Config;
use crate::error::{ProxyError, ProxyResult};
use crate::models::anthropic as models;
use crate::streaming::anthropic_to_openai::create_stream;
use crate::transform;
use axum::{
    body::Body,
    http::{HeaderMap, HeaderValue},
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use futures::StreamExt;
use reqwest::Client;
use std::sync::Arc;
use std::time::Duration;

/// 完全透传原始请求到 Anthropic API（不解析/重新序列化）
pub async fn forward_raw_request(
    config: Arc<Config>,
    client: Client,
    body: Bytes,
    is_streaming: bool,
) -> ProxyResult<Response> {
    let url = config.anthropic_messages_url();
    let api_key = config
        .anthropic_api_key
        .as_ref()
        .ok_or_else(|| ProxyError::Config("ANTHROPIC_API_KEY not configured".into()))?;

    tracing::debug!("Forwarding raw request to Anthropic: {}", url);

    // 直接发送原始 body，不做任何解析
    let req_builder = client
        .post(&url)
        .body(body)
        .header("Content-Type", "application/json")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .timeout(Duration::from_secs(300));

    let response = req_builder.send().await?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        tracing::error!("Anthropic API error ({}): {}", status, error_text);
        return Err(ProxyError::Upstream(format!(
            "Anthropic API returned {}: {}",
            status, error_text
        )));
    }

    if is_streaming {
        let stream = response.bytes_stream();
        let mut headers = HeaderMap::new();
        headers.insert(
            "Content-Type",
            HeaderValue::from_static("text/event-stream"),
        );
        headers.insert("Cache-Control", HeaderValue::from_static("no-cache"));
        headers.insert("Connection", HeaderValue::from_static("keep-alive"));

        // 直接透传流
        let passthrough_stream = stream.map(|result| {
            result.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
        });

        Ok((headers, Body::from_stream(passthrough_stream)).into_response())
    } else {
        let body = response.bytes().await?;
        Ok(Response::builder()
            .header("Content-Type", "application/json")
            .body(Body::from(body))
            .unwrap())
    }
}

/// 透传请求到 Anthropic API（解析后重新序列化，用于需要修改的场景）
#[allow(dead_code)]
pub async fn forward_request(
    config: Arc<Config>,
    client: Client,
    req: models::AnthropicRequest,
    is_streaming: bool,
) -> ProxyResult<Response> {
    let url = config.anthropic_messages_url();
    let api_key = config
        .anthropic_api_key
        .as_ref()
        .ok_or_else(|| ProxyError::Config("ANTHROPIC_API_KEY not configured".into()))?;

    tracing::debug!("Forwarding to Anthropic: {}", url);

    let req_builder = client
        .post(&url)
        .json(&req)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .timeout(Duration::from_secs(300));

    let response = req_builder.send().await?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        tracing::error!("Anthropic API error ({}): {}", status, error_text);
        return Err(ProxyError::Upstream(format!(
            "Anthropic API returned {}: {}",
            status, error_text
        )));
    }

    if is_streaming {
        let stream = response.bytes_stream();
        let mut headers = HeaderMap::new();
        headers.insert(
            "Content-Type",
            HeaderValue::from_static("text/event-stream"),
        );
        headers.insert("Cache-Control", HeaderValue::from_static("no-cache"));
        headers.insert("Connection", HeaderValue::from_static("keep-alive"));

        // 直接透传流
        let passthrough_stream = stream.map(|result| {
            result.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
        });

        Ok((headers, Body::from_stream(passthrough_stream)).into_response())
    } else {
        let body = response.bytes().await?;
        Ok(Response::builder()
            .header("Content-Type", "application/json")
            .body(Body::from(body))
            .unwrap())
    }
}

/// 处理转换后的非流式请求 (O→A)
pub async fn handle_transformed_non_streaming(
    config: Arc<Config>,
    client: Client,
    anthropic_req: models::AnthropicRequest,
) -> ProxyResult<Response> {
    let url = config.anthropic_messages_url();
    let api_key = config
        .anthropic_api_key
        .as_ref()
        .ok_or_else(|| ProxyError::Config("ANTHROPIC_API_KEY not configured".into()))?;

    tracing::debug!("Sending non-streaming request to Anthropic: {}", url);

    let req_builder = client
        .post(&url)
        .json(&anthropic_req)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .timeout(Duration::from_secs(300));

    let response = req_builder.send().await?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
        tracing::error!("Anthropic error ({}): {}", status, error_text);
        return Err(ProxyError::Upstream(format!(
            "Anthropic returned {}: {}",
            status, error_text
        )));
    }

    let anthropic_resp: models::AnthropicResponse = response.json().await?;

    if config.verbose {
        tracing::trace!(
            "Received Anthropic response: {}",
            serde_json::to_string_pretty(&anthropic_resp).unwrap_or_default()
        );
    }

    let openai_resp = transform::anthropic_to_openai_response(anthropic_resp)?;

    if config.verbose {
        tracing::trace!(
            "Transformed OpenAI response: {}",
            serde_json::to_string_pretty(&openai_resp).unwrap_or_default()
        );
    }

    Ok(Json(openai_resp).into_response())
}

/// 处理转换后的流式请求 (O→A)
pub async fn handle_transformed_streaming(
    config: Arc<Config>,
    client: Client,
    anthropic_req: models::AnthropicRequest,
) -> ProxyResult<Response> {
    let url = config.anthropic_messages_url();
    let api_key = config
        .anthropic_api_key
        .as_ref()
        .ok_or_else(|| ProxyError::Config("ANTHROPIC_API_KEY not configured".into()))?;

    tracing::debug!("Sending streaming request to Anthropic: {}", url);

    let req_builder = client
        .post(&url)
        .json(&anthropic_req)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .timeout(Duration::from_secs(300));

    let response = req_builder.send().await?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
        tracing::error!("Anthropic error ({}) from {}: {}", status, url, error_text);
        return Err(ProxyError::Upstream(format!(
            "Anthropic returned {} from {}: {}",
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
