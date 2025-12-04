//! Anthropic 响应转换为 OpenAI 格式

use crate::error::ProxyResult;
use crate::models::{anthropic, openai};

/// 将 Anthropic 响应转换为 OpenAI 格式
pub fn anthropic_to_openai_response(
    resp: anthropic::AnthropicResponse,
) -> ProxyResult<openai::OpenAIResponse> {
    let mut content = None;
    let mut tool_calls = Vec::new();

    for block in resp.content {
        match block {
            anthropic::ResponseContent::Text { text, .. } => {
                content = Some(text);
            }
            anthropic::ResponseContent::ToolUse {
                id, name, input, ..
            } => {
                tool_calls.push(openai::ToolCall {
                    id,
                    call_type: "function".to_string(),
                    function: openai::FunctionCall {
                        name,
                        arguments: serde_json::to_string(&input).unwrap_or_else(|_| "{}".to_string()),
                    },
                });
            }
            anthropic::ResponseContent::Thinking { .. } => {
                // 思考内容不包含在 OpenAI 响应中
            }
        }
    }

    let finish_reason = resp.stop_reason.map(|r| match r.as_str() {
        "end_turn" => "stop".to_string(),
        "tool_use" => "tool_calls".to_string(),
        "max_tokens" => "length".to_string(),
        _ => "stop".to_string(),
    });

    Ok(openai::OpenAIResponse {
        id: resp.id,
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: resp.model,
        choices: vec![openai::Choice {
            index: 0,
            message: openai::ChoiceMessage {
                role: "assistant".to_string(),
                content,
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
            },
            finish_reason,
        }],
        usage: openai::Usage {
            prompt_tokens: resp.usage.input_tokens,
            completion_tokens: resp.usage.output_tokens,
            total_tokens: resp.usage.input_tokens + resp.usage.output_tokens,
        },
        system_fingerprint: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_basic_anthropic_to_openai_response() {
        let resp = anthropic::AnthropicResponse {
            id: "msg_123".to_string(),
            response_type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![anthropic::ResponseContent::Text {
                content_type: "text".to_string(),
                text: "Hello!".to_string(),
            }],
            model: "claude-3-sonnet".to_string(),
            stop_reason: Some("end_turn".to_string()),
            stop_sequence: None,
            usage: anthropic::Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
        };

        let result = anthropic_to_openai_response(resp).unwrap();
        
        assert_eq!(result.id, "msg_123");
        assert_eq!(result.object, "chat.completion");
        assert_eq!(result.model, "claude-3-sonnet");
        assert_eq!(result.choices.len(), 1);
        assert_eq!(result.choices[0].message.content, Some("Hello!".to_string()));
        assert_eq!(result.choices[0].finish_reason, Some("stop".to_string()));
        assert_eq!(result.usage.prompt_tokens, 10);
        assert_eq!(result.usage.completion_tokens, 5);
        assert_eq!(result.usage.total_tokens, 15);
    }

    #[test]
    fn test_tool_use_response_conversion() {
        let resp = anthropic::AnthropicResponse {
            id: "msg_123".to_string(),
            response_type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![anthropic::ResponseContent::ToolUse {
                content_type: "tool_use".to_string(),
                id: "call_123".to_string(),
                name: "search".to_string(),
                input: json!({"query": "rust"}),
            }],
            model: "claude-3-sonnet".to_string(),
            stop_reason: Some("tool_use".to_string()),
            stop_sequence: None,
            usage: anthropic::Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
        };

        let result = anthropic_to_openai_response(resp).unwrap();
        
        assert_eq!(result.choices[0].finish_reason, Some("tool_calls".to_string()));
        assert!(result.choices[0].message.tool_calls.is_some());
        
        let tool_calls = result.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_123");
        assert_eq!(tool_calls[0].function.name, "search");
    }

    #[test]
    fn test_stop_reason_mapping() {
        let test_cases = vec![
            ("end_turn", "stop"),
            ("tool_use", "tool_calls"),
            ("max_tokens", "length"),
            ("unknown", "stop"),
        ];

        for (anthropic_reason, expected_openai) in test_cases {
            let resp = anthropic::AnthropicResponse {
                id: "test".to_string(),
                response_type: "message".to_string(),
                role: "assistant".to_string(),
                content: vec![anthropic::ResponseContent::Text {
                    content_type: "text".to_string(),
                    text: "test".to_string(),
                }],
                model: "claude-3".to_string(),
                stop_reason: Some(anthropic_reason.to_string()),
                stop_sequence: None,
                usage: anthropic::Usage {
                    input_tokens: 0,
                    output_tokens: 0,
                },
            };

            let result = anthropic_to_openai_response(resp).unwrap();
            assert_eq!(result.choices[0].finish_reason, Some(expected_openai.to_string()));
        }
    }
}
