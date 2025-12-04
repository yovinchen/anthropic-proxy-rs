//! 后端模块
//!
//! 负责与各种 LLM API 后端的通信

pub mod anthropic;
pub mod openai;
pub mod upstream;

// 重新导出 Backend 枚举
pub use crate::router::Backend;
