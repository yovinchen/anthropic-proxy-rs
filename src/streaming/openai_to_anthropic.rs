//! OpenAI 流 → Anthropic 流转换

use crate::models::openai;
use crate::transform::utils::map_stop_reason;
use bytes::Bytes;
use futures::stream::Stream;
use futures::StreamExt;
use serde_json::json;

/// 创建 OpenAI → Anthropic 流转换器
pub fn create_stream(
    stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send {
    async_stream::stream! {
        let mut buffer = String::new();
        let mut message_id = None;
        let mut current_model = None;
        let mut content_index = 0;
        let mut tool_call_id = None;
        let mut _tool_call_name = None;
        let mut tool_call_args = String::new();
        let mut has_sent_message_start = false;
        let mut current_block_type: Option<String> = None;

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
                                if data.trim() == "[DONE]" {
                                    let event = json!({"type": "message_stop"});
                                    let sse_data = format!("event: message_stop\ndata: {}\n\n",
                                        serde_json::to_string(&event).unwrap_or_default());
                                    yield Ok(Bytes::from(sse_data));
                                    continue;
                                }

                                if let Ok(chunk) = serde_json::from_str::<openai::StreamChunk>(data) {
                                    if message_id.is_none() {
                                        message_id = Some(chunk.id.clone());
                                    }
                                    if current_model.is_none() {
                                        current_model = Some(chunk.model.clone());
                                    }

                                    if let Some(choice) = chunk.choices.first() {
                                        // 发送 message_start
                                        if !has_sent_message_start {
                                            let event = json!({
                                                "type": "message_start",
                                                "message": {
                                                    "id": message_id.clone().unwrap_or_default(),
                                                    "type": "message",
                                                    "role": "assistant",
                                                    "model": current_model.clone().unwrap_or_default(),
                                                    "usage": {
                                                        "input_tokens": 0,
                                                        "output_tokens": 0
                                                    }
                                                }
                                            });
                                            let sse_data = format!("event: message_start\ndata: {}\n\n",
                                                serde_json::to_string(&event).unwrap_or_default());
                                            yield Ok(Bytes::from(sse_data));
                                            has_sent_message_start = true;
                                        }

                                        // 处理 reasoning/thinking
                                        if let Some(reasoning) = &choice.delta.reasoning {
                                            if current_block_type.is_none() {
                                                let event = json!({
                                                    "type": "content_block_start",
                                                    "index": content_index,
                                                    "content_block": {
                                                        "type": "thinking",
                                                        "thinking": ""
                                                    }
                                                });
                                                let sse_data = format!("event: content_block_start\ndata: {}\n\n",
                                                    serde_json::to_string(&event).unwrap_or_default());
                                                yield Ok(Bytes::from(sse_data));
                                                current_block_type = Some("thinking".to_string());
                                            }

                                            let event = json!({
                                                "type": "content_block_delta",
                                                "index": content_index,
                                                "delta": {
                                                    "type": "thinking_delta",
                                                    "thinking": reasoning
                                                }
                                            });
                                            let sse_data = format!("event: content_block_delta\ndata: {}\n\n",
                                                serde_json::to_string(&event).unwrap_or_default());
                                            yield Ok(Bytes::from(sse_data));
                                        }

                                        // 处理文本内容
                                        if let Some(content) = &choice.delta.content {
                                            if !content.is_empty() {
                                                if current_block_type.as_deref() != Some("text") {
                                                    if current_block_type.is_some() {
                                                        let event = json!({
                                                            "type": "content_block_stop",
                                                            "index": content_index
                                                        });
                                                        let sse_data = format!("event: content_block_stop\ndata: {}\n\n",
                                                            serde_json::to_string(&event).unwrap_or_default());
                                                        yield Ok(Bytes::from(sse_data));
                                                        content_index += 1;
                                                    }

                                                    let event = json!({
                                                        "type": "content_block_start",
                                                        "index": content_index,
                                                        "content_block": {
                                                            "type": "text",
                                                            "text": ""
                                                        }
                                                    });
                                                    let sse_data = format!("event: content_block_start\ndata: {}\n\n",
                                                        serde_json::to_string(&event).unwrap_or_default());
                                                    yield Ok(Bytes::from(sse_data));
                                                    current_block_type = Some("text".to_string());
                                                }

                                                let event = json!({
                                                    "type": "content_block_delta",
                                                    "index": content_index,
                                                    "delta": {
                                                        "type": "text_delta",
                                                        "text": content
                                                    }
                                                });
                                                let sse_data = format!("event: content_block_delta\ndata: {}\n\n",
                                                    serde_json::to_string(&event).unwrap_or_default());
                                                yield Ok(Bytes::from(sse_data));
                                            }
                                        }

                                        // 处理工具调用
                                        if let Some(tool_calls) = &choice.delta.tool_calls {
                                            for tool_call in tool_calls {
                                                if let Some(id) = &tool_call.id {
                                                    if current_block_type.is_some() {
                                                        let event = json!({
                                                            "type": "content_block_stop",
                                                            "index": content_index
                                                        });
                                                        let sse_data = format!("event: content_block_stop\ndata: {}\n\n",
                                                            serde_json::to_string(&event).unwrap_or_default());
                                                        yield Ok(Bytes::from(sse_data));
                                                        content_index += 1;
                                                    }

                                                    tool_call_id = Some(id.clone());
                                                    tool_call_args.clear();
                                                }

                                                if let Some(function) = &tool_call.function {
                                                    if let Some(name) = &function.name {
                                                        _tool_call_name = Some(name.clone());

                                                        let event = json!({
                                                            "type": "content_block_start",
                                                            "index": content_index,
                                                            "content_block": {
                                                                "type": "tool_use",
                                                                "id": tool_call_id.clone().unwrap_or_default(),
                                                                "name": name
                                                            }
                                                        });
                                                        let sse_data = format!("event: content_block_start\ndata: {}\n\n",
                                                            serde_json::to_string(&event).unwrap_or_default());
                                                        yield Ok(Bytes::from(sse_data));
                                                        current_block_type = Some("tool_use".to_string());
                                                    }

                                                    if let Some(args) = &function.arguments {
                                                        tool_call_args.push_str(args);

                                                        let event = json!({
                                                            "type": "content_block_delta",
                                                            "index": content_index,
                                                            "delta": {
                                                                "type": "input_json_delta",
                                                                "partial_json": args
                                                            }
                                                        });
                                                        let sse_data = format!("event: content_block_delta\ndata: {}\n\n",
                                                            serde_json::to_string(&event).unwrap_or_default());
                                                        yield Ok(Bytes::from(sse_data));
                                                    }
                                                }
                                            }
                                        }

                                        // 处理完成原因
                                        if let Some(finish_reason) = &choice.finish_reason {
                                            if current_block_type.is_some() {
                                                let event = json!({
                                                    "type": "content_block_stop",
                                                    "index": content_index
                                                });
                                                let sse_data = format!("event: content_block_stop\ndata: {}\n\n",
                                                    serde_json::to_string(&event).unwrap_or_default());
                                                yield Ok(Bytes::from(sse_data));
                                            }

                                            let stop_reason = map_stop_reason(Some(finish_reason));
                                            let event = json!({
                                                "type": "message_delta",
                                                "delta": {
                                                    "stop_reason": stop_reason,
                                                    "stop_sequence": serde_json::Value::Null
                                                },
                                                "usage": chunk.usage.as_ref().map(|u| json!({
                                                    "output_tokens": u.completion_tokens
                                                }))
                                            });
                                            let sse_data = format!("event: message_delta\ndata: {}\n\n",
                                                serde_json::to_string(&event).unwrap_or_default());
                                            yield Ok(Bytes::from(sse_data));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Stream error: {}", e);
                    let error_event = json!({
                        "type": "error",
                        "error": {
                            "type": "stream_error",
                            "message": format!("Stream error: {}", e)
                        }
                    });
                    let sse_data = format!("event: error\ndata: {}\n\n",
                        serde_json::to_string(&error_event).unwrap_or_default());
                    yield Ok(Bytes::from(sse_data));
                    break;
                }
            }
        }
    }
}
