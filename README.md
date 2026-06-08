# ai-navigator

A lightweight Python library that unifies LLM API calls across OpenAI, Anthropic, and Google Gemini — with YAML-driven structured output, image preprocessing, response parsing, and a SQLite-backed storage layer built in.

```python
from ai_navigator.server import OpenAIServer
from ai_navigator.schema.composer import SchemaComposer
from ai_navigator.schema.extractor import ResultExtractor

llm   = OpenAIServer("gpt-4o", credentials={"api_key": "sk-..."})
sc    = SchemaComposer.from_yaml_file("review_schema.yaml")
fmt   = sc.schema_conversion()

response = llm.response("Review: 'Great laptop, fast and light.'",
                         response_format=fmt)

import json
data   = json.loads(response.content)
result = ResultExtractor().extract(data, sc)
# → {"title": "laptop", "sentiment": "positive", "detail.score": 9}
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

# Development
pip install "ai-navigator[dev]"
```

Requires Python 3.10+.

---

## Quick start

### Call an LLM

```python
from ai_navigator.server import OpenAIServer, AnthropicServer, GeminiServer

# OpenAI
llm = OpenAIServer("gpt-4o", credentials={"api_key": "sk-..."})
response = llm.chat("What is the capital of France?")
print(response.content)   # "Paris"
print(response.usage)     # TokenUsage(prompt_tokens=..., ...)

# Anthropic
llm = AnthropicServer("claude-sonnet-4-6",
                       credentials={"api_key": "sk-ant-..."})
response = llm.chat("Explain tail-call optimisation.")

# Gemini
llm = GeminiServer("gemini-2.0-flash",
                    credentials={"api_key": "AIza..."})
response = llm.chat("What are the SOLID principles?")

# Multi-turn
from ai_navigator.infra import Message

msgs = [
    Message.system("You are a concise assistant."),
    Message.user("Name three sorting algorithms."),
]
response = llm.chat(msgs)

# Streaming
for token in llm.stream("Write a haiku about Python."):
    print(token, end="", flush=True)
```

---

## Structured output with SchemaComposer

Define your output schema in YAML, then get an OpenAI `response_format` dict in two steps.

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
    config_confidence: true        # optional: flag for logprob extraction later
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
    type: [str, null]              # anyOf → allows null
```

```python
from ai_navigator.schema.composer import SchemaComposer
from ai_navigator.schema.extractor import ResultExtractor
from ai_navigator.parser.response import ResponseParser

sc  = SchemaComposer.from_yaml_file("review_schema.yaml")
fmt = sc.schema_conversion()      # → ready-to-use response_format dict

response = llm.response(
    "Review: 'Great laptop, fast and light. Battery could be better.'",
    response_format=fmt,
)

parser = ResponseParser()
data   = parser.parse_response(response)   # extract JSON from response

# Default: dict fields expanded, lists kept whole
result = ResultExtractor().extract(data, sc)
# → {"title": "laptop", "sentiment": "positive",
#    "detail.reason": "fast and light", "detail.score": 8,
#    "tags": ["speed"], "optional_note": None}

# Expand list elements into numbered keys
result = ResultExtractor().extract(data, sc,
             configs={"extract_list_elements": True})
# → {"tags_1": "speed", ...}

# Keep parent dict key alongside children
result = ResultExtractor().extract(data, sc,
             configs={"term_extract_discard": False})
# → {"detail": {...}, "detail.reason": "...", ...}
```

### Dynamic schemas (runtime substitution)

Any field attribute can be made dynamic by prefixing it with `dynamic_`:

```python
sc = SchemaComposer.from_yaml("""
meta:
  name: Analysis
  description: Sentiment analysis
  version: "1.0"
schema:
  sentiment:
    type: enum
    dynamic_choices: labels      # choices injected at runtime
    config_confidence: true
  aspect:
    type: list
    item_type: str
    dynamic_choices: aspects
""")

resolved = sc.preprocess({
    "labels":  ["正面", "负面", "中性"],
    "aspects": ["价格", "质量", "物流"],
})
fmt = resolved.schema_conversion()
```

### Reusable definitions with `defs`

```yaml
defs:
  score_def:
    type: int
    description: Score 0-10

schema:
  quality:
    ref: score_def          # → {"$ref": "#/$defs/score_def"}
  price:
    ref: score_def
```

---

## YAML-driven prompts with PromptBuilder

Assemble multi-turn conversations from a YAML template:

```yaml
# prompt.yaml
- role: system
  message:
    - type: const_text
      content: You are a product review analyst.

- message:                           # role defaults to "user"
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
response = llm.chat(msgs)
```

---

## Image inputs

```python
from ai_navigator.pre_processor.image import ImageProcessor
from ai_navigator.infra import Message

proc = ImageProcessor()

image_part = proc.from_path("screenshot.png")
image_part = proc.from_url("https://example.com/chart.png")
image_part = proc.from_url_download("https://example.com/photo.jpg")
image_part = proc.resize("large_photo.jpg", max_px=768)  # requires [image]

msg = Message(role="user", content=[
    image_part,
    {"type": "text", "text": "What does this chart show?"},
])
response = llm.chat([msg])
```

---

## Response parsing

```python
from ai_navigator.parser.response import ResponseParser

parser = ResponseParser()

# Handles plain JSON, ```json fences, or JSON buried in prose
data = parser.parse_json('Result: {"score": 9, "label": "positive"}')

# Soft variant — returns default instead of raising
data = parser.try_parse_json("no json here", default={})

# Validate enum values
parser.validate_enum("正面", ["正面", "负面", "中性"])

# Recursive key search in nested dicts
nested = {"detail": {"reason": "good price", "score": 9}}
parser.find_value(nested, "reason")   # → "good price"
```

---

## Pipeline state — RequestState

`RequestState` carries all data through the processing pipeline:

```python
from ai_navigator.infra.state import RequestState

state = RequestState(
    request_data={"type": "message", "content": "Hello"},
    params={"temperature": 0.2},           # forwarded to LLM
    configs={"extract_list_elements": True},# pkg-internal knobs
)
# reference["schema"] — processed SchemaComposer lives here
# result              — extracted output written here
# status              — pipeline status (PENDING / OK / ERROR)
```

Request data shapes:

| `type` | Fields | Usage |
|---|---|---|
| `"message"` | `content: str \| list` | plain user input |
| `"conversation"` | `messages: list[Message]` | pre-assembled conversation |
| `"prompt"` | `template: list`, `data_dict: dict` | YAML-driven |

---

## Configuration and credentials

```python
from ai_navigator.infra.const_configs import ConstConfigs
from ai_navigator.infra.credentials import CredentialsLoader

# Constants read from env at import time; override programmatically if needed
ConstConfigs.STORAGE_PATH     # AI_NAVIGATOR_STORAGE_PATH (default: ai_navigator.db)
ConstConfigs.CREDENTIALS_PATH # AI_NAVIGATOR_CREDENTIALS_PATH (default: credentials.yaml)

# Load credentials from YAML (override fetch() for Vault / Secrets Manager)
loader = CredentialsLoader()
creds  = loader.fetch()       # → {"openai_api_key": "...", ...}
```

---

## Storage (SQLite-backed, opt-in)

```python
from ai_navigator.infra.storage import StorageBase, StoreStatus

# Use the default SQLite backend (db path from ConstConfigs.STORAGE_PATH)
storage = StorageBase()

storage.request_store("req-001", state.request_data)   # StoreStatus.OK
storage.result_store("req-001",  result)

storage.metric_report("llm_calls", "add",    {"n": 1})
storage.metric_report("model",     "update", {"name": "gpt-4o"})
storage.metric_load("llm_calls")                       # → {"n": 1}

storage.cache_store("rate:user-42", "add", {"hits": 1})
storage.cache_fetch("rate:user-42", "add", {})         # → {"hits": 1}

# Override any pair to swap backend
class RedisStorage(StorageBase):
    def cache_store(self, name, method, data): ...
    def cache_fetch(self, name, method, data): ...
```

Five pipeline store/fetch pairs:  
`request` · `reference` · `response` · `status` · `result`

---

## Error handling

```python
from ai_navigator.infra.exceptions import (
    AINavigatorError,    # base
    ProviderError,       # API call failed
    RateLimitError,      # 429 — auto-retried up to max_retries
    AuthenticationError, # 401 — bad API key
    ParseError,          # JSON extraction / Pydantic validation failed
    SchemaError,         # YAML schema definition invalid
    PreProcessorError,   # image loading / encoding failed
)

llm = OpenAIServer("gpt-4o", credentials={"api_key": "..."},
                   max_retries=5, retry_delay=2.0)

try:
    response = llm.chat("Hello")
except AuthenticationError as e:
    print(f"Bad key for {e.provider}")
except RateLimitError as e:
    print(f"Still rate-limited after retries; retry_after={e.retry_after}")
```

`RateLimitError` is retried automatically with exponential back-off.

---

## Adding a new provider

1. Create `src/ai_navigator/server/<name>_server.py`.
2. Subclass `BaseServer`; set `provider` and `_supported_methods`.
3. Override `_setup(**kwargs)` — read `self.credentials`, init the SDK client.
4. Implement `_chat(messages, **kwargs) -> Response` (and `_response`, `_stream`).
5. Add public `chat` / `response` / `stream` methods calling `self._invoke(...)`.
6. Add `_raise_<name>_error(exc)` mapping SDK errors to package exceptions.
7. Export from `server/__init__.py`; add optional dep in `pyproject.toml`.

---

## Development

```bash
git clone https://github.com/your-org/ai-navigator
cd ai-navigator
pip install -e ".[dev]"

pytest tests/ -v      # no API keys required
ruff check src/ tests/
mypy src/
```

---

## License

MIT
