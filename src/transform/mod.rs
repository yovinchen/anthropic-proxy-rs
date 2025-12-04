//! 协议转换模块
//!
//! 负责 Anthropic 和 OpenAI API 格式之间的双向转换

pub mod request;
pub mod response;
pub mod utils;

// 重新导出常用类型
pub use request::anthropic_to_openai::anthropic_to_openai;
pub use request::openai_to_anthropic::openai_to_anthropic_request;
pub use response::anthropic_to_openai::anthropic_to_openai_response;
pub use response::openai_to_anthropic::openai_to_anthropic;
