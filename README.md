# ai-navigator

A lightweight Python library that unifies LLM API calls across OpenAI, Anthropic, and Google Gemini — with YAML-driven structured output, image preprocessing, response parsing, batch inference, and a SQLite-backed storage layer built in.

```python
from ai_navigator import Navigator

nav = Navigator()

result = nav.chat(
    request_data={"message": "Summarise this in one sentence."},
    configs={"model_name": "my_claude"},
)
print(result["result"])   # content string
print(result["status"])   # {"status_code": StatusCode(200), "status_desc": "Ok", "status_detail": ""}
```

---

## Installation

```bash
# Core (no provider SDKs)
pip install ai-navigator

# With specific providers
pip install "ai-navigator[openai]"
pip install "ai-navigator[anthropic]"
pip install "ai-navigator[gemini]"

# Image preprocessing
pip install "ai-navigator[image]"

# Everything
pip install "ai-navigator[all]"
```

Requires Python 3.10+.

---

## Credentials setup

Create a `credentials.yaml` file (path overridable via `AI_NAVIGATOR_CREDENTIALS_PATH`):

```yaml
my_claude:
  - provider_type: anthropic
    model: claude-sonnet-4-6
    api_key: sk-ant-...
    max_tokens: 4096
    retry_max: 3        # optional — caps auto-retry count for this account

my_gpt4:
  - provider_type: openai
    model: gpt-4o
    api_key: sk-openai-...

my_gemini:
  - provider_type: gemini
    model: gemini-2.0-flash
    api_key: AIza...
```

Each top-level key is a `model_name` you pass in `configs`. The provider is auto-dispatched from `provider_type`.

---

## Quick start

### Single request

```python
from ai_navigator import Navigator

nav = Navigator()

result = nav.chat(
    request_data={"message": "What is the capital of France?"},
    params={"temperature": 0.0},
    configs={"model_name": "my_gpt4"},
)

print(result["result"])                        # "Paris"
print(result["usage"]["prompt_tokens"])        # 12
print(result["reference"]["model"])            # "gpt-4o"
print(result["status"]["status_code"])         # StatusCode(200)
```

### Checking for errors

No exceptions are raised for provider errors — check `status_code`:

```python
from ai_navigator import StatusCode

result = nav.chat(request_data=..., configs={"model_name": "my_gpt4"})

if result["status"]["status_code"] == StatusCode.OK:
    process(result["result"])
else:
    print(result["status"]["status_code"])    # e.g. StatusCode(429)
    print(result["status"]["status_desc"])    # "Too Many Requests"
    print(result["status"]["status_detail"])  # full error message
```

### Structured output

```python
result = nav.response(
    request_data={"message": "Review: 'Great laptop, fast and light.'"},
    params={"response_format": fmt},   # see Schema section
    configs={"model_name": "my_claude"},
)
```

### Request data shapes

| Key | Value type | Description |
|---|---|---|
| `"message"` | `str \| list` | One or more user messages — the AI has not replied yet |
| `"conversation"` | `list[Message]` | Full back-and-forth dialogue between user and assistant |
| `"prompt"` | `list` | Prompt-engineering preset (zero-shot or few-shot); pair with `"data_dict"` |

```python
from ai_navigator import make_message

# Conversation
result = nav.chat(
    request_data={
        "conversation": [
            make_message("system", "You are a concise assistant."),
            make_message("user", "Name three sorting algorithms."),
        ],
    },
    configs={"model_name": "my_claude"},
)

# YAML-driven prompt
result = nav.chat(
    request_data={
        "prompt": [...],            # loaded from YAML
        "data_dict": {"product": "laptop"},
    },
    configs={"model_name": "my_gpt4"},
)
```

---

## NavigatorResult

Every `chat()` and `response()` call returns a `NavigatorResult` TypedDict:

```python
{
    "result":    str,           # content string; empty on error
    "status":    StatusDetail,  # status_code, status_desc, status_detail
    "usage":     TokenUsage,    # prompt_tokens, completion_tokens, total_tokens
    "reference": dict,          # model, finish_reason, provider metadata
}
```

`StatusDetail` fields:

| Field | Type | Description |
|---|---|---|
| `status_code` | `StatusCode` | HTTP-style int code (`StatusCode.OK`, `StatusCode.TOO_MANY_REQUESTS`, …) |
| `status_desc` | `str` | Short label — callers may supply their own; independent of `status_describe()` |
| `status_detail` | `str` | Full error message; empty string on success |

---

## Status codes

`StatusCode` is an `int` subclass — every registered code is a singleton instance that behaves like an `IntEnum` member but supports runtime extension:

```python
from ai_navigator import StatusCode, status_describe

StatusCode.OK                  # StatusCode(200)
StatusCode.UNAUTHORIZED        # StatusCode(401)
StatusCode.TOO_MANY_REQUESTS   # StatusCode(429)
StatusCode.INTERNAL_ERROR      # StatusCode(500)
StatusCode.CONTEXT_LIMIT       # StatusCode(601)  — prompt too long
StatusCode.CONTENT_FILTERED    # StatusCode(602)  — blocked by safety policy
StatusCode.OUTPUT_TRUNCATED    # StatusCode(603)  — finish_reason = "length"
StatusCode.SCHEMA_MISMATCH     # StatusCode(604)  — structured output schema mismatch
StatusCode.EMPTY_RESPONSE      # StatusCode(605)  — model returned empty string
StatusCode.PROVIDER_TIMEOUT    # StatusCode(606)  — provider timeout

# Values compare equal to plain ints
StatusCode.OK == 200                         # True
isinstance(StatusCode.OK, int)               # True
isinstance(StatusCode.OK, StatusCode)        # True

# Each named constant is a singleton
StatusCode[200] is StatusCode.OK             # True

# Default description
StatusCode.OK.desc                           # "Ok"
status_describe(429)                         # "Too Many Requests"

# Integer lookup — validates registration
StatusCode[429]                              # StatusCode(429)
StatusCode[9999]                             # KeyError — not registered

# Register custom codes in-process
MY_TIMEOUT = StatusCode.register(709, "Custom Timeout")
MY_TIMEOUT == 709                            # True
StatusCode[709] is MY_TIMEOUT               # True
```

`status_desc` in `StatusDetail` is **independent** of `status_describe()` — servers may provide a more specific label (e.g. `"Gemini quota exceeded"` instead of `"Too Many Requests"`).

---

## Retry

Rate-limit responses (status 429) are retried automatically with exponential back-off. The effective retry count is `min(credentials.retry_max, configs.retry_max)`:

```python
result = nav.chat(
    request_data=...,
    configs={"model_name": "my_gpt4", "retry_max": 2},  # further cap on top of credentials
)
```

---

## Batch inference

### Online batch — concurrent, blocks until done

```python
results = nav.online_batch(
    source="requests.jsonl",            # or list[dict]
    params={"temperature": 0.3},
    configs={"model_name": "my_claude"},
    method="chat",                      # or "response"
    max_workers=10,
)
# → list of NavigatorResult, same order as input
```

### Offline batch — background processing

```python
# Submit and get job_id immediately
job_id = nav.offline_submit(
    source="requests.jsonl",
    params={"temperature": 0.3},
    configs={"model_name": "my_claude"},
    method="chat",
)

# Query progress at any time (survives process restart)
nav.offline_status(job_id)
# {"job_id": "...", "status": "running", "total": 200, "completed": 87, "failed": 0, ...}

# Retrieve results (partial available while running)
results = nav.offline_results(job_id)
# [{"item_idx": 0, "status": "completed", "result": {...}, "error": None}, ...]
```

JSONL format — one `request_data` dict per line:
```json
{"message": "Translate: Hello"}
{"message": "Translate: Goodbye"}
```

---

## Structured output with SchemaComposer

```yaml
# review_schema.yaml
meta:
  name: ProductReview
  description: Extract structured review data
  version: "1.0"

schema:
  title:
    type: str
    description: Product name
  sentiment:
    type: enum
    choices: [positive, negative, neutral]
    config_confidence: true
  detail:
    type: dict
    terms:
      reason:
        type: str
      score:
        type: int
  tags:
    type: list
    item_type: str
  optional_note:
    type: [str, null]
```

```python
from ai_navigator.schema.composer import SchemaComposer
from ai_navigator.schema.extractor import ResultExtractor
from ai_navigator.parser.response import ResponseParser

sc  = SchemaComposer.from_yaml_file("review_schema.yaml")
fmt = sc.schema_conversion()      # → response_format dict

result = nav.response(
    request_data={"message": "Review: 'Great laptop, fast and light.'"},
    params={"response_format": fmt},
    configs={"model_name": "my_gpt4"},
)

data   = ResponseParser().parse_response(result)
output = ResultExtractor().extract(data, sc)
# → {"title": "laptop", "sentiment": "positive",
#    "detail.reason": "fast and light", "detail.score": 8,
#    "tags": ["speed"], "optional_note": None}
```

### Dynamic schemas (runtime substitution)

```python
sc = SchemaComposer.from_yaml("""
meta:
  name: Analysis
  version: "1.0"
schema:
  sentiment:
    type: enum
    dynamic_choices: labels
    config_confidence: true
""")

resolved = sc.preprocess({"labels": ["positive", "negative", "neutral"]})
fmt = resolved.schema_conversion()
```

---

## YAML-driven prompts with PromptBuilder

```yaml
# prompt.yaml
- role: system
  message:
    - type: const_text
      content: You are a product review analyst.

- message:
    - type: const_text
      content: "Analyse this product:"
    - type: dynamic_text
      key: product_description
    - type: const_image_url
      content: "https://example.com/product.jpg"
```

```python
from ai_navigator.conf_parser.prompt import PromptBuilder

pb   = PromptBuilder.from_yaml_file("prompt.yaml")
msgs = pb.build(data_dict={"product_description": "Lightweight ergonomic mouse"})

result = nav.chat(
    request_data={"conversation": msgs},
    configs={"model_name": "my_claude"},
)
```

---

## Image inputs

```python
from ai_navigator import make_message
from ai_navigator.pre_processor.image import ImageProcessor

proc = ImageProcessor()

image_part = proc.from_path("screenshot.png")
image_part = proc.from_url("https://example.com/chart.png")
image_part = proc.resize("large_photo.jpg", max_px=768)   # requires [image]

result = nav.chat(
    request_data={
        "conversation": [
            make_message("user", [image_part, {"type": "text", "text": "What does this chart show?"}]),
        ],
    },
    configs={"model_name": "my_gpt4"},
)
```

---

## Logging

By default, `ai-navigator` logs to stderr at `INFO` level if your application has not configured any logging handlers. To suppress or redirect:

```python
import logging

# Silence completely
logging.getLogger("ai_navigator").setLevel(logging.WARNING)

# Or configure your own handler before creating Navigator
logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
# ai-navigator detects this and will not add its own handler
```

---

## Configuration

```python
from ai_navigator.param.const_configs import ConstConfigs

ConstConfigs.STORAGE_PATH      # AI_NAVIGATOR_STORAGE_PATH    (default: ai_navigator.db)
ConstConfigs.CREDENTIALS_PATH  # AI_NAVIGATOR_CREDENTIALS_PATH (default: credentials.yaml)
ConstConfigs.LOGGING_STREAM    # AI_NAVIGATOR_LOGGING_STREAM  (default: True)

# All config including any installed plugins
ConstConfigs.all()
ConstConfigs.get("MY_CUSTOM_PARAM", default="fallback")
```

---

## Storage (SQLite-backed, opt-in)

```python
from ai_navigator.monitor.storage import StorageBase, StoreStatus

storage = StorageBase()

storage.request_store("req-001", request_data)
storage.result_store("req-001",  result)
storage.metric_report("llm_calls", "add",    {"n": 1})
storage.metric_load("llm_calls")               # → {"n": 1}
storage.cache_store("rate:user-42", "add", {"hits": 1})
```

Five pipeline store/fetch pairs: `request` · `reference` · `response` · `status` · `result`

---

## Low-level server access

Direct server instantiation for fine-grained control:

```python
from ai_navigator.server import OpenAIServer, AnthropicServer, GeminiServer

llm = OpenAIServer("gpt-4o", credentials={"api_key": "sk-..."})
result = llm.chat([{"role": "user", "content": "Hello"}])
print(result["result"])                   # content string
print(result["status"]["status_code"])    # StatusCode(200)

for token in llm.stream([{"role": "user", "content": "Write a haiku."}]):
    print(token, end="", flush=True)
```

---

## Extensibility via Entry Points

ai-navigator uses Python Entry Points for all extension points. Install your plugin with `pip install` and it is picked up automatically — no code changes needed.

| Group | Behaviour | Interface |
|---|---|---|
| `ai_navigator.navigator` | **replace** BaseNavigator | subclass of `BaseNavigator` |
| `ai_navigator.credentials` | **replace** credentials loader | class with `fetch() -> dict` |
| `ai_navigator.storage` | **replace** batch storage | implements `BatchStorageProtocol` |
| `ai_navigator.configs` | **extend** ConstConfigs | callable returning `dict` |
| `ai_navigator.servers` | **supplement** provider registry | subclass of `BaseServer` |
| `ai_navigator.traffic` | **replace** rate limiter | `(configs, stats) -> bool` |
| `ai_navigator.status_codes` | **extend** StatusCode registry | `dict[int, str]` or callable returning one |

```toml
# your plugin's pyproject.toml
[project.entry-points."ai_navigator.servers"]
cohere = "my_package.server:CohereServer"

[project.entry-points."ai_navigator.credentials"]
vault  = "my_package.creds:VaultLoader"

[project.entry-points."ai_navigator.storage"]
redis  = "my_package.storage:RedisBatchStorage"

[project.entry-points."ai_navigator.navigator"]
custom = "my_package.nav:MyNavigator"

[project.entry-points."ai_navigator.configs"]
extra  = "my_package.config:get_extra_configs"

[project.entry-points."ai_navigator.traffic"]
limiter = "my_package.hooks:rate_limiter"

[project.entry-points."ai_navigator.status_codes"]
my_codes = "my_package.status:CODES"   # dict[int, str]
```

### Adding a new provider

1. Create `src/ai_navigator/server/<name>_server.py`.
2. Subclass `BaseServer`; set `provider = "<name>"` as a class attribute.
3. Override `_setup(**kwargs)` to read `self.credentials` and initialise the SDK client.
4. Implement `chat()` and `response()` and decorate each with `@server_method`. The decorator triggers auto-wrapping with retry and logging via `BaseServer.__init_subclass__`.
5. Return `NavigatorResult` directly — all provider exceptions must be caught internally and returned as error results (no exceptions cross the server boundary).
6. Register via entry point or pass in `extra_servers`.

```python
from ai_navigator import BaseServer, server_method, NavigatorResult, StatusCode, status_describe

class CohereServer(BaseServer):
    provider = "cohere"

    def _setup(self, **kwargs):
        import cohere
        self._client = cohere.Client(self.credentials["api_key"])

    @server_method
    def chat(self, messages, system=None, **kwargs) -> NavigatorResult:
        try:
            resp = self._client.chat(message=messages[-1]["content"])
        except Exception as exc:
            code = StatusCode.INTERNAL_ERROR
            self.logger.warning("Cohere error [%d]: %s", code, exc)
            return {
                "result": "",
                "status": {"status_code": code, "status_desc": status_describe(code), "status_detail": str(exc)},
                "usage": {},
                "reference": {},
            }
        return {
            "result": resp.text,
            "status": {"status_code": StatusCode.OK, "status_desc": StatusCode.OK.desc, "status_detail": ""},
            "usage": {},
            "reference": {"model": self.model},
        }
```

---

## Development

```bash
git clone https://github.com/shiyi-yinghao/ai_navigator
cd ai-navigator
pip install -e ".[dev]"

pytest tests/ -v
ruff check src/ tests/
mypy src/
```

---

## License

MIT

---

---

# ai-navigator（中文文档）

轻量级 Python 库，统一封装 OpenAI、Anthropic 和 Google Gemini 的 LLM 调用接口，内置 YAML 驱动的结构化输出、图像预处理、响应解析、批量推理和 SQLite 存储层。

```python
from ai_navigator import Navigator

nav = Navigator()

result = nav.chat(
    request_data={"message": "用一句话总结以下内容。"},
    params={"temperature": 0.3},
    configs={"model_name": "my_claude"},
)
print(result["result"])    # 内容字符串
print(result["status"])    # {"status_code": StatusCode(200), "status_desc": "Ok", "status_detail": ""}
```

---

## 安装

```bash
# 核心包（不含任何 provider SDK）
pip install ai-navigator

# 按需安装 provider
pip install "ai-navigator[openai]"
pip install "ai-navigator[anthropic]"
pip install "ai-navigator[gemini]"

# 图像预处理支持
pip install "ai-navigator[image]"

# 全量安装
pip install "ai-navigator[all]"
```

要求 Python 3.10+。

---

## 配置凭证

创建 `credentials.yaml` 文件（路径可通过 `AI_NAVIGATOR_CREDENTIALS_PATH` 环境变量覆盖）：

```yaml
my_claude:
  - provider_type: anthropic
    model: claude-sonnet-4-6
    api_key: sk-ant-...
    max_tokens: 4096
    retry_max: 3        # 可选，限制此账号的最大重试次数

my_gpt4:
  - provider_type: openai
    model: gpt-4o
    api_key: sk-openai-...

my_gemini:
  - provider_type: gemini
    model: gemini-2.0-flash
    api_key: AIza...
```

顶层 key 即为 `configs` 中传入的 `model_name`，`provider_type` 决定自动路由到哪个 provider。

---

## 快速上手

### 单次请求

```python
from ai_navigator import Navigator

nav = Navigator()

result = nav.chat(
    request_data={"message": "法国的首都是哪里？"},
    params={"temperature": 0.0},
    configs={"model_name": "my_gpt4"},
)

print(result["result"])                        # "巴黎"
print(result["usage"]["prompt_tokens"])        # 12
print(result["reference"]["model"])            # "gpt-4o"
print(result["status"]["status_code"])         # StatusCode(200)
```

### 错误处理

provider 错误不抛异常，通过 `status_code` 判断：

```python
from ai_navigator import StatusCode

result = nav.chat(request_data=..., configs={"model_name": "my_gpt4"})

if result["status"]["status_code"] == StatusCode.OK:
    process(result["result"])
else:
    print(result["status"]["status_code"])    # 例如 StatusCode(429)
    print(result["status"]["status_desc"])    # "Too Many Requests"
    print(result["status"]["status_detail"])  # 完整错误信息
```

### request_data 格式

| Key | Value 类型 | 说明 |
|---|---|---|
| `"message"` | `str \| list` | 用户发送的消息，AI 尚未回复 |
| `"conversation"` | `list[Message]` | 用户与助手的完整多轮对话 |
| `"prompt"` | `list` | Prompt Engineering 预设任务，配合 `"data_dict"` 使用 |

```python
from ai_navigator import make_message

# 多轮对话
result = nav.chat(
    request_data={
        "conversation": [
            make_message("system", "你是一个简洁的助手。"),
            make_message("user", "列举三种排序算法。"),
        ],
    },
    configs={"model_name": "my_claude"},
)
```

---

## NavigatorResult

每次 `chat()` 和 `response()` 调用返回一个 `NavigatorResult` TypedDict：

```python
{
    "result":    str,           # 内容字符串；出错时为空字符串
    "status":    StatusDetail,  # status_code、status_desc、status_detail
    "usage":     TokenUsage,    # prompt_tokens、completion_tokens、total_tokens
    "reference": dict,          # model、finish_reason 等 provider 元数据
}
```

`StatusDetail` 字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `status_code` | `StatusCode` | HTTP 风格整数码（`StatusCode.OK`、`StatusCode.TOO_MANY_REQUESTS`…） |
| `status_desc` | `str` | 简短标签；调用方可自定义，与 `status_describe()` 相互独立 |
| `status_detail` | `str` | 完整错误信息；成功时为空字符串 |

---

## 状态码（StatusCode）

`StatusCode` 是 `int` 的子类，每个已注册的码是一个单例实例，行为和 `IntEnum` 成员完全一致，但支持运行时扩展：

```python
from ai_navigator import StatusCode, status_describe

StatusCode.OK                  # StatusCode(200)
StatusCode.UNAUTHORIZED        # StatusCode(401)
StatusCode.TOO_MANY_REQUESTS   # StatusCode(429)
StatusCode.INTERNAL_ERROR      # StatusCode(500)
StatusCode.CONTEXT_LIMIT       # StatusCode(601)  — prompt 超过上下文长度
StatusCode.CONTENT_FILTERED    # StatusCode(602)  — 被内容安全策略拦截
StatusCode.OUTPUT_TRUNCATED    # StatusCode(603)  — finish_reason = "length"
StatusCode.SCHEMA_MISMATCH     # StatusCode(604)  — 结构化输出不符合 schema
StatusCode.EMPTY_RESPONSE      # StatusCode(605)  — 模型返回空字符串
StatusCode.PROVIDER_TIMEOUT    # StatusCode(606)  — provider 超时

# 与普通整数完全兼容
StatusCode.OK == 200                         # True
isinstance(StatusCode.OK, int)               # True
isinstance(StatusCode.OK, StatusCode)        # True

# 同一整数值返回同一对象
StatusCode[200] is StatusCode.OK             # True

# 默认描述
StatusCode.OK.desc                           # "Ok"
status_describe(429)                         # "Too Many Requests"

# 整数查找（验证码已注册）
StatusCode[429]                              # StatusCode(429)
StatusCode[9999]                             # KeyError — 未注册

# 进程内注册自定义码
MY_TIMEOUT = StatusCode.register(709, "Custom Timeout")
MY_TIMEOUT == 709                            # True
StatusCode[709] is MY_TIMEOUT               # True
```

`status_desc` 字段与 `status_describe()` **相互独立** — server 可提供更具体的描述（如 `"Gemini 配额已用尽"` 而非 `"Too Many Requests"`）。

---

## 重试机制

429 限流响应会自动以指数退避方式重试。有效重试次数取 `min(credentials.retry_max, configs.retry_max)`：

```python
result = nav.chat(
    request_data=...,
    configs={"model_name": "my_gpt4", "retry_max": 2},  # 在凭证限制基础上进一步约束
)
```

---

## 批量推理

### 在线批量 — 并发执行，阻塞直到完成

```python
results = nav.online_batch(
    source="requests.jsonl",            # 或 list[dict]
    params={"temperature": 0.3},
    configs={"model_name": "my_claude"},
    method="chat",                      # 或 "response"
    max_workers=10,
)
# → NavigatorResult 列表，顺序与输入一致
```

### 离线批量 — 后台处理

```python
# 提交后立即返回 job_id
job_id = nav.offline_submit(
    source="requests.jsonl",
    params={"temperature": 0.3},
    configs={"model_name": "my_claude"},
    method="chat",
)

# 随时查询进度（进程重启后仍可查询）
nav.offline_status(job_id)
# {"job_id": "...", "status": "running", "total": 200, "completed": 87, "failed": 0, ...}

# 获取结果（运行中也可获取已完成的部分）
results = nav.offline_results(job_id)
# [{"item_idx": 0, "status": "completed", "result": {...}, "error": None}, ...]
```

JSONL 格式 — 每行一个 `request_data` 字典：
```json
{"message": "翻译：Hello"}
{"message": "翻译：Goodbye"}
```

---

## SchemaComposer 结构化输出

```yaml
# review_schema.yaml
meta:
  name: ProductReview
  description: 提取结构化评论数据
  version: "1.0"

schema:
  title:
    type: str
    description: 产品名称
  sentiment:
    type: enum
    choices: [positive, negative, neutral]
    config_confidence: true
  detail:
    type: dict
    terms:
      reason:
        type: str
      score:
        type: int
  tags:
    type: list
    item_type: str
  optional_note:
    type: [str, null]
```

```python
from ai_navigator.schema.composer import SchemaComposer
from ai_navigator.schema.extractor import ResultExtractor
from ai_navigator.parser.response import ResponseParser

sc  = SchemaComposer.from_yaml_file("review_schema.yaml")
fmt = sc.schema_conversion()      # → response_format dict

result = nav.response(
    request_data={"message": "评论：'性能很好，轻薄便携。'"},
    params={"response_format": fmt},
    configs={"model_name": "my_gpt4"},
)

data   = ResponseParser().parse_response(result)
output = ResultExtractor().extract(data, sc)
# → {"title": "笔记本", "sentiment": "positive",
#    "detail.reason": "轻薄便携", "detail.score": 8,
#    "tags": ["speed"], "optional_note": None}
```

---

## PromptBuilder YAML 驱动的 Prompt

```yaml
# prompt.yaml
- role: system
  message:
    - type: const_text
      content: 你是一位产品评论分析师。

- message:
    - type: const_text
      content: "请分析这款产品："
    - type: dynamic_text
      key: product_description
    - type: const_image_url
      content: "https://example.com/product.jpg"
```

```python
from ai_navigator.conf_parser.prompt import PromptBuilder

pb   = PromptBuilder.from_yaml_file("prompt.yaml")
msgs = pb.build(data_dict={"product_description": "轻量人体工学鼠标"})

result = nav.chat(
    request_data={"conversation": msgs},
    configs={"model_name": "my_claude"},
)
```

---

## 图像输入

```python
from ai_navigator import make_message
from ai_navigator.pre_processor.image import ImageProcessor

proc = ImageProcessor()

image_part = proc.from_path("screenshot.png")
image_part = proc.from_url("https://example.com/chart.png")
image_part = proc.resize("large_photo.jpg", max_px=768)   # 需要 [image] 额外依赖

result = nav.chat(
    request_data={
        "conversation": [
            make_message("user", [image_part, {"type": "text", "text": "这张图表展示了什么？"}]),
        ],
    },
    configs={"model_name": "my_gpt4"},
)
```

---

## 日志

默认情况下，如果应用未配置任何 logging handler，`ai-navigator` 会自动向 stderr 输出 INFO 级别日志。如需关闭或重定向：

```python
import logging

# 完全静音
logging.getLogger("ai_navigator").setLevel(logging.WARNING)

# 或在创建 Navigator 前配置自己的 handler
logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
# ai-navigator 检测到已有 handler，不会再添加默认 handler
```

---

## 配置项

```python
from ai_navigator.param.const_configs import ConstConfigs

ConstConfigs.STORAGE_PATH      # AI_NAVIGATOR_STORAGE_PATH    （默认：ai_navigator.db）
ConstConfigs.CREDENTIALS_PATH  # AI_NAVIGATOR_CREDENTIALS_PATH（默认：credentials.yaml）
ConstConfigs.LOGGING_STREAM    # AI_NAVIGATOR_LOGGING_STREAM  （默认：True）

# 获取所有配置（含插件扩展）
ConstConfigs.all()
ConstConfigs.get("MY_CUSTOM_PARAM", default="fallback")
```

---

## 存储层（SQLite，按需使用）

```python
from ai_navigator.monitor.storage import StorageBase, StoreStatus

storage = StorageBase()

storage.request_store("req-001", request_data)
storage.result_store("req-001",  result)
storage.metric_report("llm_calls", "add",    {"n": 1})
storage.metric_load("llm_calls")               # → {"n": 1}
storage.cache_store("rate:user-42", "add", {"hits": 1})
```

五对 store/fetch 方法：`request` · `reference` · `response` · `status` · `result`

---

## 底层 Server 直接调用

```python
from ai_navigator.server import OpenAIServer, AnthropicServer, GeminiServer

llm = OpenAIServer("gpt-4o", credentials={"api_key": "sk-..."})
result = llm.chat([{"role": "user", "content": "你好"}])
print(result["result"])                   # 内容字符串
print(result["status"]["status_code"])    # StatusCode(200)

for token in llm.stream([{"role": "user", "content": "写一首俳句。"}]):
    print(token, end="", flush=True)
```

---

## 扩展性：Entry Points

ai-navigator 通过 Python Entry Points 机制支持全部扩展点。安装插件包后自动生效，无需修改任何代码。

| 组 | 行为 | 接口要求 |
|---|---|---|
| `ai_navigator.navigator` | **替换** BaseNavigator | `BaseNavigator` 的子类 |
| `ai_navigator.credentials` | **替换** 凭证加载器 | 实现 `fetch() -> dict` |
| `ai_navigator.storage` | **替换** 批量存储后端 | 实现 `BatchStorageProtocol` |
| `ai_navigator.configs` | **扩展** ConstConfigs | 可调用对象，返回 `dict` |
| `ai_navigator.servers` | **补充** provider 注册表 | `BaseServer` 的子类 |
| `ai_navigator.traffic` | **替换** 限流器 | `(configs, stats) -> bool` |
| `ai_navigator.status_codes` | **扩展** StatusCode 注册表 | `dict[int, str]` 或返回该类型的可调用对象 |

### 新增 provider

1. 创建 `src/ai_navigator/server/<name>_server.py`。
2. 继承 `BaseServer`，设置类属性 `provider = "<name>"`。
3. 重写 `_setup(**kwargs)` — 读取 `self.credentials`，初始化 SDK 客户端。
4. 用 `@server_method` 装饰 `chat()` 和 `response()` 方法。装饰器触发 `BaseServer.__init_subclass__` 自动注入重试和日志逻辑。
5. 直接返回 `NavigatorResult` — 所有 provider 异常必须在内部捕获并转为错误结果（不得抛出异常穿越 server 边界）。
6. 通过 entry point 注册，或在未打包时直接传入 `extra_servers`。

---

## 开发

```bash
git clone https://github.com/shiyi-yinghao/ai_navigator
cd ai-navigator
pip install -e ".[dev]"

pytest tests/ -v
ruff check src/ tests/
mypy src/
```

---

## 许可证

MIT
