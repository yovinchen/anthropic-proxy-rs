//! Anthropic 请求转换为 OpenAI 格式

use crate::config::Config;
use crate::error::{ProxyError, ProxyResult};
use crate::models::{anthropic, openai};
use crate::transform::utils::{clean_schema, parse_model_with_effort};

/// 将 Anthropic 请求转换为 OpenAI 格式
pub fn anthropic_to_openai(
    req: anthropic::AnthropicRequest,
    config: &Config,
) -> ProxyResult<openai::OpenAIRequest> {
    // 根据 thinking 参数决定模型
    let has_thinking = req
        .extra
        .get("thinking")
        .and_then(|v| v.as_object())
        .map(|o| o.get("type").and_then(|t| t.as_str()) == Some("enabled"))
        .unwrap_or(false);

    // 使用配置的模型或请求中的模型
    let raw_model = if has_thinking {
        config.reasoning_model.clone()
            .or_else(|| Some(req.model.clone()))
            .unwrap_or_else(|| req.model.clone())
    } else {
        config.completion_model.clone()
            .or_else(|| Some(req.model.clone()))
            .unwrap_or_else(|| req.model.clone())
    };

    // 解析模型名称和 effort 级别
    let (model, reasoning_effort) = parse_model_with_effort(&raw_model);

    if let Some(ref effort) = reasoning_effort {
        tracing::debug!("Using reasoning_effort: {} for model: {}", effort, model);
    }

    // 转换消息
    let mut openai_messages = Vec::new();

    // 添加系统消息
    if let Some(system) = req.system {
        match system {
            anthropic::SystemPrompt::Single(text) => {
                openai_messages.push(openai::Message {
                    role: "system".to_string(),
                    content: Some(openai::MessageContent::Text(text)),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                });
            }
            anthropic::SystemPrompt::Multiple(messages) => {
                for msg in messages {
                    openai_messages.push(openai::Message {
                        role: "system".to_string(),
                        content: Some(openai::MessageContent::Text(msg.text)),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    });
                }
            }
        }
    }

    // 转换用户/助手消息
    for msg in req.messages {
        let converted = convert_message(msg)?;
        openai_messages.extend(converted);
    }

    // 转换工具定义
    let tools = req.tools.and_then(|tools| {
        let filtered: Vec<_> = tools
            .into_iter()
            .filter(|t| t.tool_type.as_deref() != Some("BatchTool"))
            .collect();

        if filtered.is_empty() {
            None
        } else {
            Some(
                filtered
                    .into_iter()
                    .map(|t| openai::Tool {
                        tool_type: "function".to_string(),
                        function: openai::Function {
                            name: t.name,
                            description: t.description,
                            parameters: clean_schema(t.input_schema),
                        },
                    })
                    .collect(),
            )
        }
    });

    Ok(openai::OpenAIRequest {
        model,
        messages: openai_messages,
        max_tokens: Some(req.max_tokens.max(16)), // 某些提供商要求最少 16 tokens
        temperature: req.temperature,
        top_p: req.top_p,
        stop: req.stop_sequences,
        stream: req.stream,
        tools,
        tool_choice: None,
        reasoning_effort,
    })
}

/// 转换单条 Anthropic 消息为一条或多条 OpenAI 消息
fn convert_message(msg: anthropic::Message) -> ProxyResult<Vec<openai::Message>> {
    let mut result = Vec::new();

    match msg.content {
        anthropic::MessageContent::Text(text) => {
            result.push(openai::Message {
                role: msg.role,
                content: Some(openai::MessageContent::Text(text)),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        }
        anthropic::MessageContent::Blocks(blocks) => {
            let mut current_content_parts = Vec::new();
            let mut tool_calls = Vec::new();

            for block in blocks {
                match block {
                    anthropic::ContentBlock::Text { text, .. } => {
                        current_content_parts.push(openai::ContentPart::Text { text });
                    }
                    anthropic::ContentBlock::Image { source } => {
                        let data_url = format!(
                            "data:{};base64,{}",
                            source.media_type, source.data
                        );
                        current_content_parts.push(openai::ContentPart::ImageUrl {
                            image_url: openai::ImageUrl { url: data_url },
                        });
                    }
                    anthropic::ContentBlock::ToolUse { id, name, input } => {
                        tool_calls.push(openai::ToolCall {
                            id,
                            call_type: "function".to_string(),
                            function: openai::FunctionCall {
                                name,
                                arguments: serde_json::to_string(&input)
                                    .map_err(|e| ProxyError::Serialization(e))?,
                            },
                        });
                    }
                    anthropic::ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        ..
                    } => {
                        // 工具结果转换为独立的 "tool" 角色消息
                        result.push(openai::Message {
                            role: "tool".to_string(),
                            content: Some(openai::MessageContent::Text(content.to_string_content())),
                            tool_calls: None,
                            tool_call_id: Some(tool_use_id),
                            name: None,
                        });
                    }
                    anthropic::ContentBlock::Thinking { .. } => {
                        // 跳过 thinking 块
                    }
                }
            }

            // 添加包含内容和/或工具调用的消息
            if !current_content_parts.is_empty() || !tool_calls.is_empty() {
                let content = if current_content_parts.is_empty() {
                    None
                } else if current_content_parts.len() == 1 {
                    match &current_content_parts[0] {
                        openai::ContentPart::Text { text } => {
                            Some(openai::MessageContent::Text(text.clone()))
                        }
                        _ => Some(openai::MessageContent::Parts(current_content_parts)),
                    }
                } else {
                    Some(openai::MessageContent::Parts(current_content_parts))
                };

                result.push(openai::Message {
                    role: msg.role,
                    content,
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                    tool_call_id: None,
                    name: None,
                });
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn create_test_config() -> Config {
        Config {
            port: 3000,
            routing_mode: crate::config::RoutingMode::Transform,
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

    #[test]
    fn test_basic_text_conversion() {
        let config = create_test_config();
        let req = anthropic::AnthropicRequest {
            model: "claude-3-sonnet".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("Hello".to_string()),
            }],
            max_tokens: 100,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
            metadata: None,
            extra: json!({}),
        };

        let result = anthropic_to_openai(req, &config).unwrap();
        
        assert_eq!(result.model, "claude-3-sonnet");
        assert_eq!(result.messages.len(), 1);
        assert_eq!(result.messages[0].role, "user");
    }

    #[test]
    fn test_system_prompt_conversion() {
        let config = create_test_config();
        let req = anthropic::AnthropicRequest {
            model: "claude-3-sonnet".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("Hello".to_string()),
            }],
            max_tokens: 100,
            system: Some(anthropic::SystemPrompt::Single("You are helpful".to_string())),
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
            metadata: None,
            extra: json!({}),
        };

        let result = anthropic_to_openai(req, &config).unwrap();
        
        assert_eq!(result.messages.len(), 2);
        assert_eq!(result.messages[0].role, "system");
        assert_eq!(result.messages[1].role, "user");
    }

    #[test]
    fn test_tool_definition_conversion() {
        let config = create_test_config();
        let req = anthropic::AnthropicRequest {
            model: "claude-3-sonnet".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("Search for rust".to_string()),
            }],
            max_tokens: 100,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: Some(vec![anthropic::Tool {
                name: "search".to_string(),
                description: Some("Search the web".to_string()),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    }
                }),
                tool_type: None,
            }]),
            metadata: None,
            extra: json!({}),
        };

        let result = anthropic_to_openai(req, &config).unwrap();
        
        assert!(result.tools.is_some());
        let tools = result.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "search");
    }

    #[test]
    fn test_model_override_with_thinking() {
        let mut config = create_test_config();
        config.reasoning_model = Some("gpt-4-turbo".to_string());
        
        let req = anthropic::AnthropicRequest {
            model: "claude-3-sonnet".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("Think about this".to_string()),
            }],
            max_tokens: 100,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
            metadata: None,
            extra: json!({"thinking": {"type": "enabled"}}),
        };

        let result = anthropic_to_openai(req, &config).unwrap();
        
        assert_eq!(result.model, "gpt-4-turbo");
    }
}
