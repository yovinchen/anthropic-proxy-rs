//! OpenAI 请求转换为 Anthropic 格式

use crate::config::Config;
use crate::error::ProxyResult;
use crate::models::{anthropic, openai};
use serde_json::{json, Value};

/// 将 OpenAI 请求转换为 Anthropic 格式
pub fn openai_to_anthropic_request(
    req: openai::OpenAIRequest,
    config: &Config,
) -> ProxyResult<anthropic::AnthropicRequest> {
    let mut messages = Vec::new();
    let mut system_prompt = None;

    for msg in req.messages {
        match msg.role.as_str() {
            "system" => {
                // 收集系统消息
                if let Some(content) = &msg.content {
                    let text = match content {
                        openai::MessageContent::Text(t) => t.clone(),
                        openai::MessageContent::Parts(parts) => {
                            parts
                                .iter()
                                .filter_map(|p| match p {
                                    openai::ContentPart::Text { text } => Some(text.clone()),
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join("\n")
                        }
                    };
                    system_prompt = Some(anthropic::SystemPrompt::Single(text));
                }
            }
            "user" | "assistant" => {
                let content = convert_openai_message_content(&msg)?;
                messages.push(anthropic::Message {
                    role: msg.role.clone(),
                    content,
                });
            }
            "tool" => {
                // 工具结果转换为 ToolResult 内容块
                if let (Some(content), Some(tool_call_id)) = (&msg.content, &msg.tool_call_id) {
                    let text = match content {
                        openai::MessageContent::Text(t) => t.clone(),
                        openai::MessageContent::Parts(parts) => {
                            parts
                                .iter()
                                .filter_map(|p| match p {
                                    openai::ContentPart::Text { text } => Some(text.clone()),
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join("\n")
                        }
                    };
                    messages.push(anthropic::Message {
                        role: "user".to_string(),
                        content: anthropic::MessageContent::Blocks(vec![
                            anthropic::ContentBlock::ToolResult {
                                tool_use_id: tool_call_id.clone(),
                                content: anthropic::ToolResultContent::Text(text),
                                is_error: None,
                            },
                        ]),
                    });
                }
            }
            _ => {
                tracing::warn!("Unknown message role: {}", msg.role);
            }
        }
    }

    // 转换工具定义
    let tools = req.tools.map(|tools| {
        tools
            .into_iter()
            .map(|t| anthropic::Tool {
                name: t.function.name,
                description: t.function.description,
                input_schema: t.function.parameters,
                tool_type: None,
            })
            .collect()
    });

    // 使用配置的模型或请求中的模型
    let model = config
        .completion_model
        .clone()
        .unwrap_or_else(|| req.model.clone());

    Ok(anthropic::AnthropicRequest {
        model,
        messages,
        max_tokens: req.max_tokens.unwrap_or(4096),
        system: system_prompt,
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: None,
        stop_sequences: req.stop,
        stream: req.stream,
        tools,
        metadata: None,
        extra: serde_json::Value::Null,
    })
}

/// 转换 OpenAI 消息内容为 Anthropic 格式
fn convert_openai_message_content(
    msg: &openai::Message,
) -> ProxyResult<anthropic::MessageContent> {
    let mut blocks = Vec::new();

    // 处理消息内容
    if let Some(content) = &msg.content {
        match content {
            openai::MessageContent::Text(text) => {
                if !text.is_empty() {
                    blocks.push(anthropic::ContentBlock::Text {
                        text: text.clone(),
                        cache_control: None,
                    });
                }
            }
            openai::MessageContent::Parts(parts) => {
                for part in parts {
                    match part {
                        openai::ContentPart::Text { text } => {
                            blocks.push(anthropic::ContentBlock::Text {
                                text: text.clone(),
                                cache_control: None,
                            });
                        }
                        openai::ContentPart::ImageUrl { image_url } => {
                            // 解析 data URL
                            if let Some((media_type, data)) = parse_data_url(&image_url.url) {
                                blocks.push(anthropic::ContentBlock::Image {
                                    source: anthropic::ImageSource {
                                        source_type: "base64".to_string(),
                                        media_type,
                                        data,
                                    },
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    // 处理工具调用（assistant 消息）
    if let Some(tool_calls) = &msg.tool_calls {
        for tool_call in tool_calls {
            let input: Value = serde_json::from_str(&tool_call.function.arguments)
                .unwrap_or_else(|_| json!({}));
            blocks.push(anthropic::ContentBlock::ToolUse {
                id: tool_call.id.clone(),
                name: tool_call.function.name.clone(),
                input,
            });
        }
    }

    // 如果只有一个文本块，返回简单文本
    if blocks.len() == 1 {
        if let anthropic::ContentBlock::Text { text, .. } = &blocks[0] {
            return Ok(anthropic::MessageContent::Text(text.clone()));
        }
    }

    if blocks.is_empty() {
        // 返回空文本
        Ok(anthropic::MessageContent::Text(String::new()))
    } else {
        Ok(anthropic::MessageContent::Blocks(blocks))
    }
}

/// 解析 data URL
fn parse_data_url(url: &str) -> Option<(String, String)> {
    if url.starts_with("data:") {
        let rest = &url[5..];
        if let Some(comma_pos) = rest.find(',') {
            let meta = &rest[..comma_pos];
            let data = &rest[comma_pos + 1..];

            // 提取 media type
            let media_type = if let Some(semi_pos) = meta.find(';') {
                meta[..semi_pos].to_string()
            } else {
                meta.to_string()
            };

            return Some((media_type, data.to_string()));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> Config {
        Config {
            port: 3000,
            routing_mode: crate::config::RoutingMode::Transform,
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

    #[test]
    fn test_basic_openai_to_anthropic() {
        let config = create_test_config();
        let req = openai::OpenAIRequest {
            model: "gpt-4".to_string(),
            messages: vec![openai::Message {
                role: "user".to_string(),
                content: Some(openai::MessageContent::Text("Hello".to_string())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            max_tokens: Some(100),
            temperature: None,
            top_p: None,
            stop: None,
            stream: None,
            tools: None,
            tool_choice: None,
            reasoning_effort: None,
        };

        let result = openai_to_anthropic_request(req, &config).unwrap();
        
        assert_eq!(result.model, "gpt-4");
        assert_eq!(result.messages.len(), 1);
        assert_eq!(result.messages[0].role, "user");
    }

    #[test]
    fn test_system_message_conversion() {
        let config = create_test_config();
        let req = openai::OpenAIRequest {
            model: "gpt-4".to_string(),
            messages: vec![
                openai::Message {
                    role: "system".to_string(),
                    content: Some(openai::MessageContent::Text("You are helpful".to_string())),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
                openai::Message {
                    role: "user".to_string(),
                    content: Some(openai::MessageContent::Text("Hello".to_string())),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
            ],
            max_tokens: Some(100),
            temperature: None,
            top_p: None,
            stop: None,
            stream: None,
            tools: None,
            tool_choice: None,
            reasoning_effort: None,
        };

        let result = openai_to_anthropic_request(req, &config).unwrap();
        
        assert!(result.system.is_some());
        assert_eq!(result.messages.len(), 1); // 只有 user 消息
    }

    #[test]
    fn test_parse_data_url() {
        let url = "data:image/png;base64,iVBORw0KGgo=";
        let result = parse_data_url(url);
        
        assert!(result.is_some());
        let (media_type, data) = result.unwrap();
        assert_eq!(media_type, "image/png");
        assert_eq!(data, "iVBORw0KGgo=");
    }

    #[test]
    fn test_parse_data_url_without_base64() {
        let url = "data:text/plain,Hello";
        let result = parse_data_url(url);
        
        assert!(result.is_some());
        let (media_type, data) = result.unwrap();
        assert_eq!(media_type, "text/plain");
        assert_eq!(data, "Hello");
    }
}
