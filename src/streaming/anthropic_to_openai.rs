//! Anthropic 流 → OpenAI 流转换

use bytes::Bytes;
use futures::stream::Stream;
use futures::StreamExt;
use serde_json::json;

/// 创建 Anthropic → OpenAI 流转换器
pub fn create_stream(
    stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send {
    async_stream::stream! {
        let mut buffer = String::new();
        let mut message_id = String::new();
        let mut model = String::new();
        let mut current_content = String::new();
        let _current_tool_calls: Vec<serde_json::Value> = Vec::new();

        tokio::pin!(stream);

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    buffer.push_str(&text);

                    while let Some(pos) = buffer.find("\n\n") {
                        let line = buffer[..pos].to_string();
                        buffer = buffer[pos + 2..].to_string();

                        if line.trim().is_empty() {
                            continue;
                        }

                        for l in line.lines() {
                            if let Some(data) = l.strip_prefix("data: ") {
                                if let Ok(event) = serde_json::from_str::<serde_json::Value>(data) {
                                    let event_type = event.get("type").and_then(|t| t.as_str()).unwrap_or("");

                                    match event_type {
                                        "message_start" => {
                                            if let Some(msg) = event.get("message") {
                                                message_id = msg.get("id").and_then(|i| i.as_str()).unwrap_or("").to_string();
                                                model = msg.get("model").and_then(|m| m.as_str()).unwrap_or("").to_string();
                                            }
                                        }
                                        "content_block_delta" => {
                                            if let Some(delta) = event.get("delta") {
                                                let delta_type = delta.get("type").and_then(|t| t.as_str()).unwrap_or("");

                                                match delta_type {
                                                    "text_delta" => {
                                                        if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                                                            current_content.push_str(text);

                                                            let openai_chunk = json!({
                                                                "id": message_id,
                                                                "object": "chat.completion.chunk",
                                                                "created": std::time::SystemTime::now()
                                                                    .duration_since(std::time::UNIX_EPOCH)
                                                                    .unwrap()
                                                                    .as_secs(),
                                                                "model": model,
                                                                "choices": [{
                                                                    "index": 0,
                                                                    "delta": {
                                                                        "content": text
                                                                    },
                                                                    "finish_reason": serde_json::Value::Null
                                                                }]
                                                            });
                                                            let sse_data = format!("data: {}\n\n",
                                                                serde_json::to_string(&openai_chunk).unwrap_or_default());
                                                            yield Ok(Bytes::from(sse_data));
                                                        }
                                                    }
                                                    "input_json_delta" => {
                                                        if let Some(json_str) = delta.get("partial_json").and_then(|j| j.as_str()) {
                                                            // Tool call argument streaming
                                                            let openai_chunk = json!({
                                                                "id": message_id,
                                                                "object": "chat.completion.chunk",
                                                                "created": std::time::SystemTime::now()
                                                                    .duration_since(std::time::UNIX_EPOCH)
                                                                    .unwrap()
                                                                    .as_secs(),
                                                                "model": model,
                                                                "choices": [{
                                                                    "index": 0,
                                                                    "delta": {
                                                                        "tool_calls": [{
                                                                            "index": 0,
                                                                            "function": {
                                                                                "arguments": json_str
                                                                            }
                                                                        }]
                                                                    },
                                                                    "finish_reason": serde_json::Value::Null
                                                                }]
                                                            });
                                                            let sse_data = format!("data: {}\n\n",
                                                                serde_json::to_string(&openai_chunk).unwrap_or_default());
                                                            yield Ok(Bytes::from(sse_data));
                                                        }
                                                    }
                                                    _ => {}
                                                }
                                            }
                                        }
                                        "content_block_start" => {
                                            if let Some(block) = event.get("content_block") {
                                                let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
                                                if block_type == "tool_use" {
                                                    let tool_id = block.get("id").and_then(|i| i.as_str()).unwrap_or("");
                                                    let tool_name = block.get("name").and_then(|n| n.as_str()).unwrap_or("");

                                                    let openai_chunk = json!({
                                                        "id": message_id,
                                                        "object": "chat.completion.chunk",
                                                        "created": std::time::SystemTime::now()
                                                            .duration_since(std::time::UNIX_EPOCH)
                                                            .unwrap()
                                                            .as_secs(),
                                                        "model": model,
                                                        "choices": [{
                                                            "index": 0,
                                                            "delta": {
                                                                "tool_calls": [{
                                                                    "index": 0,
                                                                    "id": tool_id,
                                                                    "type": "function",
                                                                    "function": {
                                                                        "name": tool_name,
                                                                        "arguments": ""
                                                                    }
                                                                }]
                                                            },
                                                            "finish_reason": serde_json::Value::Null
                                                        }]
                                                    });
                                                    let sse_data = format!("data: {}\n\n",
                                                        serde_json::to_string(&openai_chunk).unwrap_or_default());
                                                    yield Ok(Bytes::from(sse_data));
                                                }
                                            }
                                        }
                                        "message_delta" => {
                                            if let Some(delta) = event.get("delta") {
                                                if let Some(stop_reason) = delta.get("stop_reason").and_then(|s| s.as_str()) {
                                                    let finish_reason = match stop_reason {
                                                        "end_turn" => "stop",
                                                        "tool_use" => "tool_calls",
                                                        "max_tokens" => "length",
                                                        _ => "stop",
                                                    };

                                                    let openai_chunk = json!({
                                                        "id": message_id,
                                                        "object": "chat.completion.chunk",
                                                        "created": std::time::SystemTime::now()
                                                            .duration_since(std::time::UNIX_EPOCH)
                                                            .unwrap()
                                                            .as_secs(),
                                                        "model": model,
                                                        "choices": [{
                                                            "index": 0,
                                                            "delta": {},
                                                            "finish_reason": finish_reason
                                                        }]
                                                    });
                                                    let sse_data = format!("data: {}\n\n",
                                                        serde_json::to_string(&openai_chunk).unwrap_or_default());
                                                    yield Ok(Bytes::from(sse_data));
                                                }
                                            }
                                        }
                                        "message_stop" => {
                                            yield Ok(Bytes::from("data: [DONE]\n\n"));
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Stream error: {}", e);
                    break;
                }
            }
        }
    }
}
