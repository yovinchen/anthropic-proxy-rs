//! OpenAI 后端
//!
//! 处理与 OpenAI API 的通信

use crate::config::Config;
use crate::error::{ProxyError, ProxyResult};
use crate::models::openai as models;
use axum::{
    body::Body,
    http::{HeaderMap, HeaderValue},
    response::{IntoResponse, Response},
};
use futures::StreamExt;
use reqwest::Client;
use std::sync::Arc;
use std::time::Duration;

/// 透传请求到 OpenAI API
pub async fn forward_request(
    config: Arc<Config>,
    client: Client,
    req: models::OpenAIRequest,
    is_streaming: bool,
) -> ProxyResult<Response> {
    let url = config.openai_chat_completions_url();
    let api_key = config
        .openai_api_key
        .as_ref()
        .ok_or_else(|| ProxyError::Config("OPENAI_API_KEY not configured".into()))?;

    tracing::debug!("Forwarding to OpenAI: {}", url);

    let req_builder = client
        .post(&url)
        .json(&req)
        .header("Authorization", format!("Bearer {}", api_key))
        .timeout(Duration::from_secs(300));

    let response = req_builder.send().await?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        tracing::error!("OpenAI API error ({}): {}", status, error_text);
        return Err(ProxyError::Upstream(format!(
            "OpenAI API returned {}: {}",
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
