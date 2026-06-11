# ai-navigator

A lightweight Python library that unifies LLM API calls across OpenAI, Anthropic, and Google Gemini — with YAML-driven structured output, image preprocessing, response parsing, batch inference, and a SQLite-backed storage layer built in.

```python
from ai_navigator import Navigator

nav = Navigator()

result = nav.chat(
    request_data={"message": "Summarise this in one sentence."},
    params={"temperature": 0.3},
    configs={"model_name": "my_claude"},
)
print(result["content"])
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

# Chat
result = nav.chat(
    request_data={"message": "What is the capital of France?"},
    params={"temperature": 0.0},
    configs={"model_name": "my_gpt4"},
)
print(result["content"])   # "Paris"
print(result["usage"])     # {"prompt_tokens": ..., "completion_tokens": ..., ...}

# Structured output
result = nav.response(
    request_data={"message": "Review: 'Great laptop, fast and light.'"},
    params={"response_format": fmt},   # see Schema section
    configs={"model_name": "my_claude"},
)
```

### Request data shapes

| Key | Value type | Description |
|---|---|---|
| `"message"` | `str \| list` | Single user message |
| `"conversation"` | `list[Message]` | Full multi-turn conversation |
| `"prompt"` | `list` | YAML template list; pair with `"data_dict"` |

```python
from ai_navigator import user_message, system_message

# Conversation
result = nav.chat(
    request_data={
        "conversation": [
            system_message("You are a concise assistant."),
            user_message("Name three sorting algorithms."),
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
# → list of result dicts, same order as input
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
from ai_navigator.pre_processor.image import ImageProcessor
from ai_navigator import user_message

proc = ImageProcessor()

image_part = proc.from_path("screenshot.png")
image_part = proc.from_url("https://example.com/chart.png")
image_part = proc.resize("large_photo.jpg", max_px=768)   # requires [image]

result = nav.chat(
    request_data={
        "conversation": [
            user_message([image_part, {"type": "text", "text": "What does this chart show?"}]),
        ],
    },
    configs={"model_name": "my_gpt4"},
)
```

---

## Configuration

```python
from ai_navigator.param.const_configs import ConstConfigs

ConstConfigs.STORAGE_PATH      # AI_NAVIGATOR_STORAGE_PATH   (default: ai_navigator.db)
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

## Extensibility via Entry Points

ai-navigator uses Python Entry Points for all extension points. Install your plugin with `pip install` and it is picked up automatically — no code changes needed.

| Group | Behaviour | Interface |
|---|---|---|
| `ai_navigator.navigator` | **replace** BaseNavigator | subclass of `BaseNavigator` |
| `ai_navigator.credentials` | **replace** credentials loader | class with `fetch() -> dict` |
| `ai_navigator.storage` | **replace** batch storage | implements `BatchStorageProtocol` |
| `ai_navigator.configs` | **extend** ConstConfigs | callable returning `dict` |
| `ai_navigator.servers` | **supplement** provider registry | subclass of `BaseServer` |

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
```

### Adding a new provider (built-in pattern)

1. Create `src/ai_navigator/server/<name>_server.py`.
2. Subclass `BaseServer`; set class attribute `provider = "<name>"`.
3. Override `_setup(**kwargs)` — read `self.credentials`, init the SDK client.
4. Implement `_chat(messages, **kwargs) -> Response` (and `_response`, `_stream`).
5. Add public `chat` / `response` / `stream` methods calling `self._invoke(...)`.
6. Register via entry point or pass in `extra_servers` if not installed as a package.

---

## Low-level server access

Direct server instantiation is available when you need fine-grained control:

```python
from ai_navigator.server import OpenAIServer, AnthropicServer, GeminiServer

llm = OpenAIServer("gpt-4o", credentials={"api_key": "sk-..."})
result = llm.chat([{"role": "user", "content": "Hello"}])
print(result["content"])

for token in llm.stream([{"role": "user", "content": "Write a haiku."}]):
    print(token, end="", flush=True)
```

---

## Error handling

```python
from ai_navigator.infra.exceptions import (
    AINavigatorError,     # base
    ProviderError,        # API call failed
    RateLimitError,       # 429 — auto-retried with exponential back-off
    AuthenticationError,  # 401 — bad API key
    ParseError,           # JSON extraction / validation failed
    SchemaError,          # YAML schema definition invalid
    PreProcessorError,    # image loading / encoding failed
)

try:
    result = nav.chat(request_data=..., configs={"model_name": "my_gpt4"})
except AuthenticationError as e:
    print(f"Bad key: {e}")
except RateLimitError as e:
    print(f"Rate limited after retries")
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
print(result["content"])
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

# 对话
result = nav.chat(
    request_data={"message": "法国的首都是哪里？"},
    params={"temperature": 0.0},
    configs={"model_name": "my_gpt4"},
)
print(result["content"])   # "巴黎"
print(result["usage"])     # {"prompt_tokens": ..., "completion_tokens": ..., ...}

# 结构化输出
result = nav.response(
    request_data={"message": "评论：'性能很好，轻薄便携。'"},
    params={"response_format": fmt},   # 详见 Schema 章节
    configs={"model_name": "my_claude"},
)
```

### request_data 格式

| Key | Value 类型 | 说明 |
|---|---|---|
| `"message"` | `str \| list` | 单条用户消息 |
| `"conversation"` | `list[Message]` | 完整多轮对话 |
| `"prompt"` | `list` | YAML 模板列表，配合 `"data_dict"` 使用 |

```python
from ai_navigator import user_message, system_message

# 多轮对话
result = nav.chat(
    request_data={
        "conversation": [
            system_message("你是一个简洁的助手。"),
            user_message("列举三种排序算法。"),
        ],
    },
    configs={"model_name": "my_claude"},
)

# YAML 驱动的 prompt
result = nav.chat(
    request_data={
        "prompt": [...],            # 从 YAML 加载
        "data_dict": {"product": "笔记本电脑"},
    },
    configs={"model_name": "my_gpt4"},
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
# → 与输入顺序一致的结果列表
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

### 动态 Schema（运行时注入）

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

resolved = sc.preprocess({"labels": ["正面", "负面", "中性"]})
fmt = resolved.schema_conversion()
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
from ai_navigator.pre_processor.image import ImageProcessor
from ai_navigator import user_message

proc = ImageProcessor()

image_part = proc.from_path("screenshot.png")
image_part = proc.from_url("https://example.com/chart.png")
image_part = proc.resize("large_photo.jpg", max_px=768)   # 需要 [image] 额外依赖

result = nav.chat(
    request_data={
        "conversation": [
            user_message([image_part, {"type": "text", "text": "这张图表展示了什么？"}]),
        ],
    },
    configs={"model_name": "my_gpt4"},
)
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

## 扩展性：Entry Points

ai-navigator 通过 Python Entry Points 机制支持全部扩展点。安装插件包后自动生效，无需修改任何代码。

| 组 | 行为 | 接口要求 |
|---|---|---|
| `ai_navigator.navigator` | **替换** BaseNavigator | `BaseNavigator` 的子类 |
| `ai_navigator.credentials` | **替换** 凭证加载器 | 实现 `fetch() -> dict` |
| `ai_navigator.storage` | **替换** 批量存储后端 | 实现 `BatchStorageProtocol` |
| `ai_navigator.configs` | **扩展** ConstConfigs | 可调用对象，返回 `dict` |
| `ai_navigator.servers` | **补充** provider 注册表 | `BaseServer` 的子类 |

```toml
# 插件包的 pyproject.toml
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
```

### 新增 provider（内置方式）

1. 创建 `src/ai_navigator/server/<name>_server.py`。
2. 继承 `BaseServer`，设置类属性 `provider = "<name>"`。
3. 重写 `_setup(**kwargs)` — 读取 `self.credentials`，初始化 SDK 客户端。
4. 实现 `_chat(messages, **kwargs) -> Response`（以及 `_response`、`_stream`）。
5. 添加公开方法 `chat` / `response` / `stream`，内部调用 `self._invoke(...)`。
6. 通过 entry point 注册，或在未打包时直接传入 `extra_servers`。

---

## 底层 Server 直接调用

需要精细控制时可直接实例化 Server：

```python
from ai_navigator.server import OpenAIServer, AnthropicServer, GeminiServer

llm = OpenAIServer("gpt-4o", credentials={"api_key": "sk-..."})
result = llm.chat([{"role": "user", "content": "你好"}])
print(result["content"])

for token in llm.stream([{"role": "user", "content": "写一首俳句。"}]):
    print(token, end="", flush=True)
```

---

## 错误处理

```python
from ai_navigator.infra.exceptions import (
    AINavigatorError,     # 基类
    ProviderError,        # API 调用失败
    RateLimitError,       # 429 — 自动指数退避重试
    AuthenticationError,  # 401 — API key 无效
    ParseError,           # JSON 提取 / 校验失败
    SchemaError,          # YAML schema 定义错误
    PreProcessorError,    # 图像加载 / 编码失败
)

try:
    result = nav.chat(request_data=..., configs={"model_name": "my_gpt4"})
except AuthenticationError as e:
    print(f"Key 无效：{e}")
except RateLimitError:
    print("已达速率限制，重试后仍失败")
```

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
