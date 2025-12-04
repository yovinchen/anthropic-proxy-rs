//! OpenAI 响应转换为 Anthropic 格式

use crate::error::{ProxyError, ProxyResult};
use crate::models::{anthropic, openai};
use serde_json::json;

/// 将 OpenAI 响应转换为 Anthropic 格式
pub fn openai_to_anthropic(
    resp: openai::OpenAIResponse,
) -> ProxyResult<anthropic::AnthropicResponse> {
    let choice = resp
        .choices
        .first()
        .ok_or_else(|| ProxyError::Transform("No choices in response".to_string()))?;

    let mut content = Vec::new();

    // 添加文本内容
    if let Some(text) = &choice.message.content {
        if !text.is_empty() {
            content.push(anthropic::ResponseContent::Text {
                content_type: "text".to_string(),
                text: text.clone(),
            });
        }
    }

    // 添加工具调用
    if let Some(tool_calls) = &choice.message.tool_calls {
        for tool_call in tool_calls {
            let input: serde_json::Value = serde_json::from_str(&tool_call.function.arguments)
                .unwrap_or_else(|_| json!({}));

            content.push(anthropic::ResponseContent::ToolUse {
                content_type: "tool_use".to_string(),
                id: tool_call.id.clone(),
                name: tool_call.function.name.clone(),
                input,
            });
        }
    }

    let stop_reason = choice
        .finish_reason
        .as_ref()
        .map(|r| match r.as_str() {
            "tool_calls" => "tool_use",
            "stop" => "end_turn",
            "length" => "max_tokens",
            _ => "end_turn",
        })
        .map(String::from);

    Ok(anthropic::AnthropicResponse {
        id: resp.id,
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: resp.model,
        stop_reason,
        stop_sequence: None,
        usage: anthropic::Usage {
            input_tokens: resp.usage.prompt_tokens,
            output_tokens: resp.usage.completion_tokens,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_response_conversion() {
        let resp = openai::OpenAIResponse {
            id: "chatcmpl-123".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "gpt-4".to_string(),
            choices: vec![openai::Choice {
                index: 0,
                message: openai::ChoiceMessage {
                    role: "assistant".to_string(),
                    content: Some("Hello!".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: openai::Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
            system_fingerprint: None,
        };

        let result = openai_to_anthropic(resp).unwrap();
        
        assert_eq!(result.id, "chatcmpl-123");
        assert_eq!(result.role, "assistant");
        assert_eq!(result.content.len(), 1);
        assert_eq!(result.stop_reason, Some("end_turn".to_string()));
        assert_eq!(result.usage.input_tokens, 10);
        assert_eq!(result.usage.output_tokens, 5);
    }

    #[test]
    fn test_tool_call_response_conversion() {
        let resp = openai::OpenAIResponse {
            id: "chatcmpl-123".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "gpt-4".to_string(),
            choices: vec![openai::Choice {
                index: 0,
                message: openai::ChoiceMessage {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: Some(vec![openai::ToolCall {
                        id: "call_123".to_string(),
                        call_type: "function".to_string(),
                        function: openai::FunctionCall {
                            name: "search".to_string(),
                            arguments: r#"{"query":"rust"}"#.to_string(),
                        },
                    }]),
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            usage: openai::Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
            system_fingerprint: None,
        };

        let result = openai_to_anthropic(resp).unwrap();
        
        assert_eq!(result.content.len(), 1);
        assert_eq!(result.stop_reason, Some("tool_use".to_string()));
        
        match &result.content[0] {
            anthropic::ResponseContent::ToolUse { name, id, .. } => {
                assert_eq!(name, "search");
                assert_eq!(id, "call_123");
            }
            _ => panic!("Expected ToolUse content"),
        }
    }

    #[test]
    fn test_stop_reason_mapping() {
        let test_cases = vec![
            ("stop", "end_turn"),
            ("tool_calls", "tool_use"),
            ("length", "max_tokens"),
            ("unknown", "end_turn"),
        ];

        for (openai_reason, expected_anthropic) in test_cases {
            let resp = openai::OpenAIResponse {
                id: "test".to_string(),
                object: "chat.completion".to_string(),
                created: 0,
                model: "gpt-4".to_string(),
                choices: vec![openai::Choice {
                    index: 0,
                    message: openai::ChoiceMessage {
                        role: "assistant".to_string(),
                        content: Some("test".to_string()),
                        tool_calls: None,
                    },
                    finish_reason: Some(openai_reason.to_string()),
                }],
                usage: openai::Usage {
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    total_tokens: 0,
                },
                system_fingerprint: None,
            };

            let result = openai_to_anthropic(resp).unwrap();
            assert_eq!(result.stop_reason, Some(expected_anthropic.to_string()));
        }
    }
}
