//! 转换工具函数

use serde_json::Value;

/// 有效的 reasoning effort 级别
pub const EFFORT_LEVELS: &[&str] = &["minimal", "low", "medium", "high"];

/// 解析模型名称并提取 effort 后缀
/// 例如: "gpt-5.1-codex-high" -> ("gpt-5.1-codex", Some("high"))
pub fn parse_model_with_effort(model: &str) -> (String, Option<String>) {
    for &effort in EFFORT_LEVELS {
        let suffix = format!("-{}", effort);
        if model.ends_with(&suffix) {
            let base_model = model[..model.len() - suffix.len()].to_string();
            return (base_model, Some(effort.to_string()));
        }
    }
    (model.to_string(), None)
}

/// 清理 JSON schema，移除不支持的格式
pub fn clean_schema(mut schema: Value) -> Value {
    if let Some(obj) = schema.as_object_mut() {
        // 移除 "format": "uri"
        if obj.get("format").and_then(|v| v.as_str()) == Some("uri") {
            obj.remove("format");
        }

        // 递归清理嵌套 schema
        if let Some(properties) = obj.get_mut("properties").and_then(|v| v.as_object_mut()) {
            for (_, value) in properties.iter_mut() {
                *value = clean_schema(value.clone());
            }
        }

        if let Some(items) = obj.get_mut("items") {
            *items = clean_schema(items.clone());
        }
    }

    schema
}

/// 映射 OpenAI finish_reason 到 Anthropic stop_reason
pub fn map_stop_reason(finish_reason: Option<&str>) -> Option<String> {
    finish_reason.map(|r| match r {
        "tool_calls" => "tool_use",
        "stop" => "end_turn",
        "length" => "max_tokens",
        _ => "end_turn",
    }.to_string())
}


/// 解析 data URL
pub fn parse_data_url(url: &str) -> Option<(String, String)> {
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

    #[test]
    fn test_parse_model_with_effort_high() {
        let (model, effort) = parse_model_with_effort("gpt-4-high");
        assert_eq!(model, "gpt-4");
        assert_eq!(effort, Some("high".to_string()));
    }

    #[test]
    fn test_parse_model_with_effort_medium() {
        let (model, effort) = parse_model_with_effort("claude-3-sonnet-medium");
        assert_eq!(model, "claude-3-sonnet");
        assert_eq!(effort, Some("medium".to_string()));
    }

    #[test]
    fn test_parse_model_without_effort() {
        let (model, effort) = parse_model_with_effort("gpt-4-turbo");
        assert_eq!(model, "gpt-4-turbo");
        assert_eq!(effort, None);
    }

    #[test]
    fn test_clean_schema_removes_uri_format() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "format": "uri"
                }
            }
        });

        let cleaned = clean_schema(schema);
        let url_prop = cleaned.get("properties").unwrap().get("url").unwrap();
        assert!(url_prop.get("format").is_none());
    }

    #[test]
    fn test_clean_schema_preserves_other_formats() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "format": "email"
                }
            }
        });

        let cleaned = clean_schema(schema);
        let email_prop = cleaned.get("properties").unwrap().get("email").unwrap();
        assert_eq!(email_prop.get("format").unwrap(), "email");
    }

    #[test]
    fn test_map_stop_reason_tool_calls() {
        assert_eq!(map_stop_reason(Some("tool_calls")), Some("tool_use".to_string()));
    }

    #[test]
    fn test_map_stop_reason_stop() {
        assert_eq!(map_stop_reason(Some("stop")), Some("end_turn".to_string()));
    }

    #[test]
    fn test_map_stop_reason_length() {
        assert_eq!(map_stop_reason(Some("length")), Some("max_tokens".to_string()));
    }

    #[test]
    fn test_map_stop_reason_none() {
        assert_eq!(map_stop_reason(None), None);
    }

    #[test]
    fn test_parse_data_url_png() {
        let url = "data:image/png;base64,iVBORw0KGgo=";
        let result = parse_data_url(url);
        assert!(result.is_some());
        let (media_type, data) = result.unwrap();
        assert_eq!(media_type, "image/png");
        assert_eq!(data, "iVBORw0KGgo=");
    }

    #[test]
    fn test_parse_data_url_jpeg() {
        let url = "data:image/jpeg;base64,/9j/4AAQ";
        let result = parse_data_url(url);
        assert!(result.is_some());
        let (media_type, data) = result.unwrap();
        assert_eq!(media_type, "image/jpeg");
        assert_eq!(data, "/9j/4AAQ");
    }

    #[test]
    fn test_parse_data_url_invalid() {
        let url = "https://example.com/image.png";
        let result = parse_data_url(url);
        assert!(result.is_none());
    }
}
