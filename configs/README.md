# 配置文件说明

本目录包含四种配置模式的示例文件。

## 配置文件列表

| 文件 | 模式 | 用途 |
|------|------|------|
| `transform-mode.env` | Transform | Anthropic → OpenAI 转换 |
| `passthrough-anthropic.env` | Passthrough | Anthropic 透传 |
| `passthrough-openai.env` | Auto | OpenAI 透传 |
| `auto-mode.env` | Auto/Gateway | 智能路由 + 双向转换 |

## 快速使用

### 1. Transform 模式 (使用 OpenRouter)

```bash
# 复制并编辑配置
cp configs/transform-mode.env .env
# 编辑 .env，填入你的 OpenRouter API Key

# 启动
anthropic-proxy

# 测试
curl http://localhost:3000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3-5-sonnet", "max_tokens": 100, "messages": [{"role": "user", "content": "Hi"}]}'
```

### 2. Passthrough 模式 (Anthropic 透传)

```bash
# 复制并编辑配置
cp configs/passthrough-anthropic.env .env
# 编辑 .env，填入你的 Anthropic API Key

# 启动
anthropic-proxy

# 测试
curl http://localhost:3000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3-5-sonnet-20241022", "max_tokens": 100, "messages": [{"role": "user", "content": "Hi"}]}'
```

### 3. OpenAI 透传模式

```bash
# 复制并编辑配置
cp configs/passthrough-openai.env .env
# 编辑 .env，填入你的 OpenAI API Key

# 启动
anthropic-proxy

# 测试 (透传到 OpenAI)
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]}'
```

### 4. Auto 模式 (智能路由)

```bash
# 复制并编辑配置
cp configs/auto-mode.env .env
# 编辑 .env，填入 Anthropic 和 OpenAI API Keys

# 启动
anthropic-proxy

# 测试 Claude (透传到 Anthropic)
curl http://localhost:3000/v1/messages \
  -d '{"model": "claude-3-5-sonnet", "max_tokens": 100, "messages": [{"role": "user", "content": "Hi"}]}'

# 测试 GPT (透传到 OpenAI)
curl http://localhost:3000/v1/chat/completions \
  -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]}'
```

## 模式对比

```
┌─────────────────────────────────────────────────────────────────┐
│                        Transform 模式                           │
├─────────────────────────────────────────────────────────────────┤
│  Anthropic 请求 ──→ [转换] ──→ OpenAI 兼容后端                   │
│                                                                 │
│  ✅ 支持: /v1/messages                                          │
│  ❌ 不支持: /v1/chat/completions                                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  Passthrough 模式 (Anthropic)                   │
├─────────────────────────────────────────────────────────────────┤
│  Anthropic 请求 ──→ [透传] ──→ Anthropic API                    │
│                                                                 │
│  ✅ 支持: /v1/messages                                          │
│  ❌ 不支持: /v1/chat/completions                                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   OpenAI 透传 (Auto 模式)                       │
├─────────────────────────────────────────────────────────────────┤
│  OpenAI 请求 + GPT ───────→ [透传] ──→ OpenAI API               │
│                                                                 │
│  ✅ 支持: /v1/chat/completions                                  │
│  ⚠️ /v1/messages 需要配置 Anthropic 后端                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      Auto/Gateway 模式                          │
├─────────────────────────────────────────────────────────────────┤
│  Anthropic 请求 + Claude ──→ [透传] ──→ Anthropic API           │
│  Anthropic 请求 + GPT ────→ [A→O] ──→ OpenAI API               │
│  OpenAI 请求 + GPT ───────→ [透传] ──→ OpenAI API               │
│  OpenAI 请求 + Claude ────→ [O→A] ──→ Anthropic API            │
│                                                                 │
│  ✅ 支持: /v1/messages                                          │
│  ✅ 支持: /v1/chat/completions                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 环境变量说明

### 必需变量

| 模式 | 必需变量 |
|------|----------|
| Transform | `UPSTREAM_BASE_URL`, `UPSTREAM_API_KEY` |
| Passthrough | `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY` |
| Auto/Gateway | 至少配置一个后端 |

### 可选变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PORT` | 3000 | 监听端口 |
| `ROUTING_MODE` | transform | 路由模式 |
| `REASONING_MODEL` | - | 扩展思考模型 |
| `COMPLETION_MODEL` | - | 普通补全模型 |
| `DEBUG` | false | 调试日志 |
| `VERBOSE` | false | 详细日志 |

## 测试脚本

```bash
#!/bin/bash
# test-all-modes.sh

echo "=== 测试 Transform 模式 ==="
anthropic-proxy --config configs/transform-mode.env &
PID=$!
sleep 2
curl -s http://localhost:3000/health
kill $PID

echo "=== 测试 Passthrough 模式 ==="
anthropic-proxy --config configs/passthrough-anthropic.env &
PID=$!
sleep 2
curl -s http://localhost:3000/health
kill $PID

echo "=== 测试 Auto 模式 ==="
anthropic-proxy --config configs/auto-mode.env &
PID=$!
sleep 2
curl -s http://localhost:3000/health
kill $PID

echo "=== 所有模式测试完成 ==="
```
