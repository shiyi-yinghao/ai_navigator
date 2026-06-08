# ai-navigator — Agent Guide

## Project purpose

`ai-navigator` is a PyPI package that provides a unified, provider-agnostic interface for calling LLMs (OpenAI, Anthropic, Google Gemini) and building structured prompts. It normalises request/response handling, conversation storage, schema definition, image pre-processing, and response parsing into a coherent Python API.

---

## Repository layout

```
ai-navigator/
├── pyproject.toml
├── src/ai_navigator/
│   ├── __init__.py                 # re-exports the most-used public symbols
│   ├── infra/                      # FOUNDATION — pure data + utilities, no provider coupling
│   │   ├── exceptions.py           # full exception hierarchy
│   │   ├── logger.py               # get_logger(name) → stdlib Logger
│   │   ├── models.py               # Message, Response, TokenUsage, ContentPart  (Pydantic v2)
│   │   ├── const_configs.py        # ConstConfigs — env-var-backed package constants
│   │   ├── credentials.py          # CredentialsLoader — YAML fetch, override for other sources
│   │   ├── state.py                # RequestState pipeline container
│   │   └── storage.py              # StorageBase (SQLite default, no init args); StoreStatus
│   ├── server/                     # SERVER LAYER — BaseServer + all provider implementations
│   │   ├── base_server.py          # BaseServer (ABC) — credentials, conversation, dispatch
│   │   ├── openai_server.py        # OpenAIServer
│   │   ├── anthropic_server.py     # AnthropicServer
│   │   └── gemini_server.py        # GeminiServer
│   ├── schema/
│   │   ├── composer.py             # SchemaComposer — YAML → preprocess + schema_conversion
│   │   └── extractor.py            # ResultExtractor — LLM result → flat leaf-node dict
│   ├── conf_parser/
│   │   ├── parser.py               # ConfParser: multi-schema YAML config files
│   │   └── prompt.py               # PromptBuilder: YAML-driven conversation assembly
│   ├── pre_processor/
│   │   └── image.py                # ImageProcessor: local/URL/bytes → ContentPart
│   └── parser/
│       └── response.py             # ResponseParser: JSON extract, Pydantic validate,
│                                   #   find_value, enum check
└── tests/
    └── test_basic.py               # unit tests; no API keys needed
```

---

## Module responsibilities

### `infra/storage.py` — StorageBase

Concrete default implementation: all storage uses **SQLite** (three tables:
`pipeline_data`, `metrics`, `cache`).  All I/O is wrapped in `try/except` —
permission errors degrade gracefully (log warning, return `None` /
`StoreStatus.ERROR`).

No constructor args — db path from `ConstConfigs.STORAGE_PATH`
(env `AI_NAVIGATOR_STORAGE_PATH`, default `ai_navigator.db`).
Override `_get_db_path()` to change location (e.g. in tests).

| pair | pipeline stage | table |
|---|---|---|
| `request_store` / `request_fetch` | raw user input | `pipeline_data` |
| `reference_store` / `reference_fetch` | processed schema / prompts | `pipeline_data` |
| `response_store` / `response_fetch` | server raw LLM response | `pipeline_data` |
| `status_store` / `status_fetch` | processing status | `pipeline_data` |
| `result_store` / `result_fetch` | extracted / parsed result | `pipeline_data` |
| `metric_report` / `metric_load` | aggregate metrics | `metrics` |
| `cache_store` / `cache_fetch` | high-frequency counters | `cache` |

Store methods return `StoreStatus.OK` / `StoreStatus.ERROR` (string constants).
Fetch methods return the value or `None` (not found).

Override any pair to swap the backend — store/fetch pairs **must be overridden
together**.

---

### `infra/state.py` — RequestState

Pipeline state container.  Passed through processing stages so intermediate
steps don't require extra function arguments.

```
request_data  {"type": "message",      "content": str | list}
              {"type": "conversation", "messages": list[Message]}
              {"type": "prompt",       "template": list, "data_dict": dict}

params        LLM / server parameters passed directly to the provider call.
              Examples: temperature, max_tokens, top_p, logprobs, top_logprobs

configs       Package-internal control knobs (NOT forwarded to provider).
              Examples:
                term_extract_discard   bool  default True
                extract_list_elements  bool  default False

reference     derived artefacts: e.g. {"schema": <SchemaComposer>}
              — the processed schema lives here after .preprocess()

result        populated by final stage with parsed LLM output

status        Status(code=StatusCode.PENDING|OK|ERROR, message="")
```

### `schema/composer.py` — SchemaComposer

**Request-side** schema handling: YAML definition → OpenAI structured-output dict.

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

schema:                            # dict: term-name → spec (no "name:" key inside)
  title:
    type: str
    description: Product title

  rating:
    ref: score_range               # → {"$ref": "#/$defs/score_range"}

  category:
    type: enum
    choices: [electronics, clothing, food]
    config_confidence: true        # pkg-internal: include in LogProbParser extraction

  category_dyn:
    type: enum
    dynamic_choices: cat_list      # choices from data_dict["cat_list"] at preprocess()
    config_confidence: true

  optional_note:
    type: [str, null]              # list of types → anyOf in JSON Schema

  detail:
    type: dict
    terms:                         # nested terms also use dict format
      reason:
        type: str
      score:
        ref: score_range

  tags:
    type: list
    item_type: str                 # str / int / float / bool  (default: str)
    choices: [fast, light]         # optional — constrains list items
```

**Supported types** — `str` / `string` / `free-text`, `int` / `integer`,
`float` / `number`, `bool` / `boolean`, `null`, `enum`, `list`, `dict`, `any`.
A list of types (e.g. `[str, null]`) produces `anyOf` in JSON Schema.

**`defs:` section** — optional reusable term definitions.  Referenced inside
`schema:` (and nested `terms:`) with `ref: def_name`, which produces
`{"$ref": "#/$defs/def_name"}` in the output.  Defs themselves support all
the same attributes as regular terms, including `dynamic_*`.

**Dynamic attributes** — any attribute can be made dynamic by prefixing it
with `dynamic_`.  `dynamic_{attr}: key` sets `attr = data_dict[key]` at
`preprocess()` time, then removes the `dynamic_*` key.  Examples:

| Attribute | Effect |
|---|---|
| `dynamic_type: key` | sets `type` (resolved before all other logic) |
| `dynamic_description: key` | sets `description` |
| `dynamic_choices: key` | sets `choices` for `enum` / `list` |
| `dynamic_terms: key` | sets `terms` for a `dict` term |

**`config_*` attributes** — any attribute whose name starts with `config_` is
a package-internal directive and is **never forwarded** to the JSON Schema
output (silently ignored by `schema_conversion()`).

**`config_confidence: bool`** — optional, default `false`.  When `true` the
term is included in `confidence_terms()`, enabling `LogProbParser` to extract
a probability distribution for that field.  Meaningful only on `enum` terms.

**`required` is absent from YAML** — all terms are always included in the JSON
Schema `required` array (OpenAI strict mode mandates this).

**Field-name constraint** — Term names must not contain `"."`.  A `SchemaError`
is raised at `schema_conversion()` time.

**Two-phase workflow**

1. `preprocess(data_dict)` → resolves all `dynamic_*` attributes; returns a new `SchemaComposer`.
2. `schema_conversion(task_name=None)` → returns `{"type": "json_schema", "json_schema": {...}}`.

**Introspection helpers**
- `confidence_terms()` → `{path: [candidates]}` for `config_confidence: true` enum terms.
- `build_prompt_instruction()` → plain-text system-prompt fragment.

### `schema/extractor.py` — ResultExtractor

**Response-side** schema handling: raw LLM dict → flat result dict.

Extraction is driven by the active **parse-type set** (not a leaf/non-leaf flag):

| Condition | Behaviour |
|---|---|
| `type == "dict"` (always active) | Recurse into `terms`; keys become dot-notation paths |
| `type == "list"` + `configs["extract_list_elements"]=True` | Flatten to `name_1`, `name_2`, … |
| Anything else | Return value as-is |

`configs["term_extract_discard"]` (bool, default `True`) controls whether the **parent key is kept** when a term is expanded:

- `True` (default): parent key discarded, only expanded children in result
- `False`: parent key also written to result alongside expanded children

```python
extractor = ResultExtractor()

# Default — dict expanded, parent discarded
result = extractor.extract(data, composer)
# {"title": "Phone", "detail.reason": "fast", "detail.score": 9,
#  "tags": ["a", "b"]}

# Keep parent key alongside children
result = extractor.extract(data, composer,
                           configs={"term_extract_discard": False})
# {"title": "Phone",
#  "detail": {"reason": "fast", "score": 9},  ← also kept
#  "detail.reason": "fast", "detail.score": 9,
#  "tags": ["a", "b"]}

# List expansion + discard (default)
result = extractor.extract(data, composer,
                           configs={"extract_list_elements": True})
# {"detail.reason": "fast", "detail.score": 9,
#  "tags_1": "a", "tags_2": "b"}    ← "tags" gone
```

### `conf_parser/prompt.py` — PromptBuilder

```yaml
- role: system
  message:
    - type: const_text
      content: You are a helpful assistant.

- message:   # role defaults to "user"
    - type: const_text
      content: "Describe this product:"
    - type: dynamic_text
      key: product_description
```

`const_*` keeps literal `content`; `dynamic_*` reads `data_dict[key]`.
Single-text messages collapse to `Message(role, content: str)`.

### `parser/` — response parsing

| File | Responsibility |
|---|---|
| `response.py` | `ResponseParser`: JSON extraction, Pydantic validation, `find_value`, enum validation. |

> `logprob.py` and `position.py` are offline — moved to `_lab/` (outside the
> package, not shipped).  `config_confidence` is preserved in schema terms for
> future re-integration.

---

## Data flow

```
RequestState.request_data
  type="message"      → normalise str|list → list[Message]
  type="conversation" → pass through
  type="prompt"       → PromptBuilder.build(data_dict) → list[Message]

SchemaComposer.from_yaml(yaml_str)
  .preprocess(data_dict)              → stored in RequestState.reference["schema"]
  .schema_conversion()                → response_format dict for llm.response()

ConcreteServer.response(msgs, response_format=...)
  → Response(content=..., raw=completion)

ResponseParser.parse_response(response)
  → data: dict

ResultExtractor().extract(data, composer)
  → {"title": "...", "detail.reason": "...", "detail.score": 9}
  → stored in RequestState.result

─── Optional: logprob (offline — _lab/logprob.py + _lab/position.py) ─────────

composer.confidence_terms()
  → {"sentiment": ["正面", "负面", "中性"]}
  (feeds LogProbParser when re-integrated)
```

---

## Conventions

- `BaseServer` never reads or parses `self.credentials` — concrete server's job inside `_setup`.
- All SDK imports are **lazy** (inside `_setup`). The package imports cleanly without any provider SDK installed.
- `_chat` / `_response` must raise a `ProviderError` subclass (never a raw SDK exception).
- Path separator throughout `parser/` and `schema/` is **"."** (dot).
- Term names in schemas **must not contain "."**.  Validated at `schema_conversion()` time.
- `static_*` types are fully YAML-defined; `dynamic_*` types require `preprocess(data_dict)`.

---

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

No API keys required — all tests exercise `infra`, `parser`, and `schema` only.
