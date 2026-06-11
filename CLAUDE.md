# ai-navigator — Agent Guide

## Project purpose

`ai-navigator` is a PyPI package that provides a unified, provider-agnostic interface for calling LLMs (OpenAI, Anthropic, Google Gemini) and building structured prompts. It normalises request/response handling, batch inference, storage, schema definition, image pre-processing, and response parsing into a coherent Python API.

---

## Repository layout

```
ai-navigator/
├── pyproject.toml
├── src/ai_navigator/
│   ├── __init__.py                       # re-exports the most-used public symbols
│   ├── navigator.py                      # Navigator facade — user entry point
│   ├── infra/                            # FOUNDATION — zero external imports, no provider coupling
│   │   ├── exceptions.py                 # full exception hierarchy
│   │   ├── types.py                      # TypedDicts: ContentPart, Message, Response, TokenUsage
│   │   ├── state.py                      # RequestState pipeline container
│   │   ├── base_navigator.py             # shim → service.base_navigator (backward compat)
│   │   ├── const_configs.py              # shim → param.const_configs
│   │   ├── credentials.py                # shim → param.credentials
│   │   ├── models.py                     # shim → infra.types (backward compat)
│   │   └── storage.py                    # shim → monitor.storage (backward compat)
│   ├── param/                            # CONFIGURATION LAYER
│   │   ├── __init__.py
│   │   ├── const_configs.py              # ConstConfigs — env-var-backed package constants
│   │   └── credentials.py                # CredentialsLoader + get_credentials_class()
│   ├── monitor/                          # OBSERVABILITY LAYER
│   │   ├── __init__.py
│   │   ├── logger.py                     # get_logger(name) → stdlib Logger
│   │   └── storage.py                    # StorageBase (SQLite), StoreStatus
│   ├── server/                           # SERVER LAYER — BaseServer + provider implementations
│   │   ├── __init__.py
│   │   ├── registry.py                   # build_registry() — discovers built-in + EP servers
│   │   ├── base_server.py                # BaseServer (ABC) — credentials, dispatch
│   │   ├── openai_server.py              # OpenAIServer
│   │   ├── anthropic_server.py           # AnthropicServer
│   │   └── gemini_server.py              # GeminiServer
│   ├── service/                          # SERVICE LAYER — navigation, dispatch, call methods
│   │   ├── __init__.py
│   │   └── base_navigator.py             # BaseNavigator + get_navigator_class()
│   ├── batch_inference/                  # BATCH LAYER
│   │   ├── __init__.py
│   │   ├── online.py                     # OnlineBatch — concurrent, thread pool
│   │   ├── offline.py                    # OfflineBatch — background daemon thread + storage
│   │   └── storage.py                    # BatchStorage (SQLite) + BatchStorageProtocol
│   ├── schema/
│   │   ├── composer.py                   # SchemaComposer — YAML → preprocess + schema_conversion
│   │   └── extractor.py                  # ResultExtractor — LLM result → flat leaf-node dict
│   ├── conf_parser/
│   │   ├── parser.py                     # ConfParser: multi-schema YAML config files
│   │   └── prompt.py                     # PromptBuilder: YAML-driven conversation assembly
│   ├── pre_processor/
│   │   └── image.py                      # ImageProcessor: local/URL/bytes → ContentPart
│   └── parser/
│       └── response.py                   # ResponseParser: JSON extract, find_value, enum check
└── tests/
    └── test_basic.py                     # unit tests; no API keys needed
```

---

## Import topology (strict layering)

```
navigator.py
    └── service/             (BaseNavigator, get_navigator_class)
            └── server/      (build_registry, *Server classes)
            └── param/       (ConstConfigs, get_credentials_class)
            └── monitor/     (get_logger, StorageBase)
                    └── infra/   (types, exceptions, state)
                    └── param/   (ConstConfigs — param imports infra only)
batch_inference/
    └── service/ (lazy, in _get_nav())
    └── batch_inference/storage.py
```

**Rule: `infra/` has zero imports from within `ai_navigator`.** Every other layer may import from layers below it but not above.

The shim files in `infra/` (`base_navigator.py`, `const_configs.py`, `credentials.py`, `models.py`, `storage.py`) are backward-compatibility re-exports only — they do not contain logic.

---

## Entry Points

ai-navigator exposes five extension groups:

| Group | Behaviour | Expected interface |
|---|---|---|
| `ai_navigator.navigator` | **replace** BaseNavigator | subclass of `BaseNavigator` |
| `ai_navigator.credentials` | **replace** CredentialsLoader | class with `fetch() -> dict` |
| `ai_navigator.storage` | **replace** batch storage backend | implements `BatchStorageProtocol` |
| `ai_navigator.configs` | **extend** ConstConfigs | callable returning `dict[str, Any]` |
| `ai_navigator.servers` | **supplement** provider registry | subclass of `BaseServer` with `provider` attr |

Discovery follows the same pattern in every case:
1. Call `importlib.metadata.entry_points(group=...)`.
2. For **replace** groups — use the first entry only (warn if multiple); fall back to the default class on failure.
3. For **extend/supplement** groups — load all entries; merge or add to the built-in registry.

All discovery results are module-level cached (`_*_class_cache`) after the first call.

---

## Module responsibilities

### `param/const_configs.py` — ConstConfigs

Env-var-backed package constants.  Uses `logging.getLogger` directly (not `monitor.logger`) to avoid a circular import.

```python
ConstConfigs.STORAGE_PATH      # AI_NAVIGATOR_STORAGE_PATH   (default: ai_navigator.db)
ConstConfigs.CREDENTIALS_PATH  # AI_NAVIGATOR_CREDENTIALS_PATH (default: credentials.yaml)
ConstConfigs.LOGGING_STREAM    # AI_NAVIGATOR_LOGGING_STREAM  (default: True)

ConstConfigs.get("MY_KEY", default="x")  # also checks ai_navigator.configs EP extensions
ConstConfigs.all()                        # base attrs merged with EP extension dicts
```

### `param/credentials.py` — CredentialsLoader

Reads the YAML credentials file at `ConstConfigs.CREDENTIALS_PATH`.  Returns `{}` (not an exception) when the file is missing.

```python
def get_credentials_class() -> type:
    # checks ai_navigator.credentials EP; returns CredentialsLoader if none installed
```

### `server/registry.py` — build_registry

Builds the `{provider_type: ServerClass}` mapping.  Called once inside `BaseNavigator.__init__`.  Server files are imported lazily (inside function body) to avoid circular imports.

```python
def build_registry(extra: list | None = None) -> dict[str, type]:
    # built-ins: AnthropicServer, OpenAIServer, GeminiServer
    # supplements: ai_navigator.servers EP entries
    # extra: additional classes passed directly
```

### `service/base_navigator.py` — BaseNavigator

Core routing logic.  No constructor parameters — credentials and registry are always discovered via `param/` and `server/registry.py`.

```python
class BaseNavigator:
    def __init__(self) -> None: ...       # loads creds + registry
    def chat(request_data, params, configs) -> Response: ...
    def response(request_data, params, configs) -> Response: ...

def get_navigator_class() -> type:
    # checks ai_navigator.navigator EP; returns BaseNavigator if none installed
```

`_preprocess(request_data)` handles the three `request_data` shapes:

| Key present | Action |
|---|---|
| `"message"` | wrap value (str/list) in `[user_message(value)]` |
| `"conversation"` | pass value (list) through unchanged |
| `"prompt"` | call `PromptBuilder(value).build(data_dict=request_data.get("data_dict", {}))` |

### `navigator.py` — Navigator

Pure facade.  No logic beyond delegation.

```python
class Navigator:
    def chat(...)        # → self._nav.chat(...)
    def response(...)    # → self._nav.response(...)
    def __getattr__      # delegate plugin-added methods to self._nav
    def online_batch(...)
    def offline_submit(...)
    def offline_status(job_id)
    def offline_results(job_id)
```

### `batch_inference/online.py` — OnlineBatch

Concurrent dispatch via `ThreadPoolExecutor`.  Navigator is lazy-initialised on `run()`.  Returns results in input order.

```python
OnlineBatch(method="chat", max_workers=8).run(source, params, configs)
# source: path to JSONL file or list[dict]
```

### `batch_inference/offline.py` — OfflineBatch

Background processing with SQLite-backed progress tracking.  Navigator is lazy-initialised on `submit()`.  `job_status()` and `get_results()` never create a navigator.

```python
job_id = OfflineBatch(method="chat").submit(source, params, configs)
OfflineBatch().job_status(job_id)    # {"status": "running", "completed": 42, ...}
OfflineBatch().get_results(job_id)   # list[{"item_idx", "status", "result", "error"}]
```

Status progression: `pending` → `running` → `completed` | `completed_with_errors`

### `batch_inference/storage.py` — BatchStorage + BatchStorageProtocol

SQLite default (tables: `batch_jobs`, `batch_items`).  Thread-safe via per-operation connect/close.

`BatchStorageProtocol` is a `@runtime_checkable Protocol` — implement it to supply a custom backend via the `ai_navigator.storage` entry point.

```python
def get_batch_storage_class() -> type:
    # checks ai_navigator.storage EP; returns BatchStorage if none installed
```

### `monitor/storage.py` — StorageBase

General-purpose SQLite store for pipeline data, metrics, and cache.  Separate from batch storage.

| pair | stage | table |
|---|---|---|
| `request_store` / `request_fetch` | raw user input | `pipeline_data` |
| `reference_store` / `reference_fetch` | processed schema / prompts | `pipeline_data` |
| `response_store` / `response_fetch` | server raw LLM response | `pipeline_data` |
| `status_store` / `status_fetch` | processing status | `pipeline_data` |
| `result_store` / `result_fetch` | extracted / parsed result | `pipeline_data` |
| `metric_report` / `metric_load` | aggregate metrics | `metrics` |
| `cache_store` / `cache_fetch` | high-frequency counters | `cache` |

Store methods return `StoreStatus.OK` / `StoreStatus.ERROR`.  Fetch methods return value or `None`.

### `infra/types.py` — TypedDicts

Zero-dependency data types.  The true bottom of the import topology.

```python
class ContentPart(TypedDict, total=False): type, text, image_url, image_data, media_type
class Message(TypedDict): role: str; content: str | list
class TokenUsage(TypedDict, total=False): prompt_tokens, completion_tokens, thinking_tokens, total_tokens
class Response(TypedDict, total=False): content, model, usage, finish_reason, raw

make_message(role, content) -> Message
make_content_part(type_, **kwargs) -> ContentPart
```

All `Response` instances are plain dicts — access with `result["content"]`, not `result.content`.

### `infra/state.py` — RequestState

Pipeline state container for intermediate stages.

```
request_data  — raw input from caller:
                  {"message": str | list}
                  {"conversation": list[Message]}
                  {"prompt": list, "data_dict": dict}
params        — provider call parameters (temperature, max_tokens, …)
configs       — package control knobs (NOT forwarded to provider)
reference     — {"schema": SchemaComposer, ...} after preprocess
result        — populated by final stage
status        — Status(code=StatusCode.PENDING|OK|ERROR, message="")
```

### `schema/composer.py` — SchemaComposer

Request-side schema handling: YAML definition → OpenAI structured-output dict.

Two-phase workflow:
1. `preprocess(data_dict)` — resolves all `dynamic_*` attributes; returns a new `SchemaComposer`.
2. `schema_conversion(task_name=None)` — returns `{"type": "json_schema", "json_schema": {...}}`.

**YAML format**

```yaml
meta:
  name: ProductReview
  description: Extract structured review data
  version: "1.0"

defs:                              # optional — reusable definitions
  score_range:
    type: int
    description: Score from 0 to 10

schema:                            # dict: term-name → spec
  title:
    type: str
    description: Product title
  rating:
    ref: score_range               # → {"$ref": "#/$defs/score_range"}
  category:
    type: enum
    choices: [electronics, clothing, food]
    config_confidence: true        # pkg-internal: include in confidence_terms()
  category_dyn:
    type: enum
    dynamic_choices: cat_list      # choices from data_dict["cat_list"] at preprocess()
  optional_note:
    type: [str, null]              # list of types → anyOf in JSON Schema
  detail:
    type: dict
    terms:
      reason: {type: str}
      score: {ref: score_range}
  tags:
    type: list
    item_type: str
    choices: [fast, light]         # optional — constrains list items
```

Supported types: `str`, `int`, `float`, `bool`, `null`, `enum`, `list`, `dict`, `any`. A list of types produces `anyOf`.

`config_*` attributes are package-internal and never forwarded to JSON Schema.

Term names must not contain `"."`.

### `schema/extractor.py` — ResultExtractor

Response-side schema handling: raw LLM dict → flat result dict.

```python
extractor.extract(data, composer)
# Default: dict terms expanded to dot-paths, parent key discarded
# {"title": "Phone", "detail.reason": "fast", "detail.score": 9, "tags": [...]}

extractor.extract(data, composer, configs={"term_extract_discard": False})
# Parent key also kept alongside dot-paths

extractor.extract(data, composer, configs={"extract_list_elements": True})
# List items split into name_1, name_2, … (parent key dropped if term_extract_discard=True)
```

### `conf_parser/prompt.py` — PromptBuilder

YAML-driven conversation assembly.

```yaml
- role: system
  message:
    - type: const_text
      content: You are a helpful assistant.
- message:
    - type: const_text
      content: "Describe this:"
    - type: dynamic_text
      key: product_description
    - type: const_image_url
      content: "https://example.com/img.jpg"
```

`const_*` → literal `content`; `dynamic_*` → `data_dict[key]`.
Single-text messages collapse to `{"role": "...", "content": str}`.

### `parser/response.py` — ResponseParser

JSON extraction, `find_value`, enum validation from raw LLM response.

---

## Data flow

```
User
  │
  ▼
Navigator.chat({"message": ...} | {"conversation": ...} | {"prompt": ..., "data_dict": ...}, params, configs)
  │
  ▼
BaseNavigator.chat()
  ├─ _preprocess(request_data)     → list[Message]
  ├─ _get_server(model_name)       → looks up registry + credentials
  │     └─ build_registry()        → {provider: ServerClass} (cached)
  │     └─ get_credentials_class() → CredentialsLoader (cached)
  └─ server.chat(messages, **params)
        └─ provider SDK call
              └─ Response TypedDict {"content", "model", "usage", ...}

─── Structured output ───────────────────────────────────────────────

SchemaComposer.from_yaml_file("schema.yaml")
  .preprocess(data_dict)           → new SchemaComposer (dynamic attrs resolved)
  .schema_conversion()             → {"type": "json_schema", "json_schema": {...}}

Navigator.response(request_data, params={"response_format": fmt}, configs={...})
  └─ server.response(messages, response_format=fmt)

ResponseParser().parse_response(result)  → dict
ResultExtractor().extract(data, composer) → {"field": "...", "nested.field": "..."}

─── Batch ───────────────────────────────────────────────────────────

OnlineBatch(method, max_workers).run(source, params, configs)
  └─ ThreadPoolExecutor → nav.chat/response per item → list[Response]

OfflineBatch(method).submit(source, params, configs) → job_id
  └─ daemon thread → BatchStorage (SQLite) → progress tracking
OfflineBatch().job_status(job_id) | .get_results(job_id)
```

---

## Conventions

- `BaseServer` never reads or parses `self.credentials` — concrete server's job inside `_setup`.
- All provider SDK imports are **lazy** (inside `_setup` / function bodies). The package imports cleanly without any provider SDK installed.
- `_chat` / `_response` must raise a `ProviderError` subclass (never a raw SDK exception).
- Path separator throughout `parser/` and `schema/` is **"."** (dot).
- Term names in schemas **must not contain "."**. Validated at `schema_conversion()` time.
- `static_*` types are fully YAML-defined; `dynamic_*` types require `preprocess(data_dict)`.
- All `Response` and `Message` instances are plain `dict` — use `result["content"]`, not `result.content`.
- `get_navigator_class()`, `get_credentials_class()`, `get_batch_storage_class()` all cache after first call. Call in tests with a fresh import or reset the module-level cache variable to `None` to force re-discovery.

---

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

No API keys required — all tests exercise `infra`, `parser`, `schema`, and `param` only.
