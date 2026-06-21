"""Unit tests that run without any provider API keys installed."""
import os
import tempfile

import pytest
from pydantic import BaseModel

from ai_navigator.infra.types import Message, TokenUsage, make_message
from ai_navigator.monitor.storage import StorageBase, StoreStatus
from ai_navigator.parser.response import ResponseParser
from ai_navigator.schema.composer import SchemaComposer
from ai_navigator.schema.extractor import ResultExtractor


# ── Message helpers ───────────────────────────────────────────────────────────

def test_make_message_roles():
    assert make_message("user", "hi")["role"] == "user"
    assert make_message("system", "be helpful")["role"] == "system"
    assert make_message("assistant", "hello")["content"] == "hello"


def test_make_message_list_content():
    parts = [{"type": "text", "text": "hello"}]
    msg = make_message("user", parts)
    assert msg["content"] == parts


# ── StorageBase (SQLite + file default impl) ──────────────────────────────────

def _tmp_storage() -> StorageBase:
    db = os.path.join(tempfile.mkdtemp(), "test.db")
    class _TmpStorage(StorageBase):
        def _get_db_path(self): return db
    return _TmpStorage()


def test_storage_five_pairs_roundtrip():
    s = _tmp_storage()
    pairs = [
        (s.request_store,   s.request_fetch),
        (s.reference_store, s.reference_fetch),
        (s.response_store,  s.response_fetch),
        (s.status_store,    s.status_fetch),
        (s.result_store,    s.result_fetch),
    ]
    for store_fn, fetch_fn in pairs:
        assert store_fn("k1", {"x": 1}) == StoreStatus.OK
        assert fetch_fn("k1") == {"x": 1}
        assert fetch_fn("missing") is None


def test_storage_metric_add_accumulates():
    s = _tmp_storage()
    s.metric_report("calls", "add", {"n": 1})
    s.metric_report("calls", "add", {"n": 2})
    assert s.metric_load("calls") == {"n": 3}


def test_storage_metric_update_overwrites():
    s = _tmp_storage()
    s.metric_report("info", "update", {"model": "gpt-4o", "tokens": 100})
    s.metric_report("info", "update", {"tokens": 200})
    assert s.metric_load("info") == {"model": "gpt-4o", "tokens": 200}


def test_storage_metric_load_missing():
    s = _tmp_storage()
    assert s.metric_load("nonexistent") is None


def test_storage_cache_store_and_fetch():
    s = _tmp_storage()
    result = s.cache_store("rate", "add", {"hits": 5})
    assert result == {"hits": 5}
    result = s.cache_store("rate", "add", {"hits": 3})
    assert result == {"hits": 8}
    assert s.cache_fetch("rate", "add", {}) == {"hits": 8}


def test_storage_override_single_pair():
    db = os.path.join(tempfile.mkdtemp(), "h.db")

    class _HybridStorage(StorageBase):
        def _get_db_path(self): return db

        def result_store(self, key, value):
            if not hasattr(self, "_results"):
                self._results: dict = {}
            self._results[key] = value
            return StoreStatus.OK

        def result_fetch(self, key):
            return getattr(self, "_results", {}).get(key)

    s = _HybridStorage()
    assert s.result_store("r1", "hello") == StoreStatus.OK
    assert s.result_fetch("r1") == "hello"
    assert s.request_store("q1", {"prompt": "hi"}) == StoreStatus.OK
    assert s.request_fetch("q1") == {"prompt": "hi"}


# ── ResponseParser ────────────────────────────────────────────────────────────

def test_parse_json_plain_object():
    p = ResponseParser()
    assert p.parse_json('{"name": "Alice", "age": 30}') == {"name": "Alice", "age": 30}


def test_parse_json_from_markdown_fence():
    p = ResponseParser()
    text = "Here you go:\n```json\n{\"score\": 9}\n```"
    assert p.parse_json(text) == {"score": 9}


def test_parse_json_from_bare_prose():
    p = ResponseParser()
    text = 'The answer is {"key": "value"} as requested.'
    assert p.parse_json(text) == {"key": "value"}


def test_parse_json_raises_when_missing():
    p = ResponseParser()
    with pytest.raises(ValueError, match="No JSON"):
        p.parse_json("no json here at all")


def test_try_parse_json_returns_default_on_failure():
    p = ResponseParser()
    result = p.try_parse_json("no json", default={"fallback": True})
    assert result == {"fallback": True}


def test_parse_pydantic():
    class Item(BaseModel):
        name: str
        count: int

    p = ResponseParser()
    obj = p.parse_pydantic('{"name": "widget", "count": 3}', Item)
    assert obj.name == "widget"
    assert obj.count == 3


# ── StatusCode ────────────────────────────────────────────────────────────────

def test_status_code_named_attrs():
    from ai_navigator.state.status import StatusCode
    assert StatusCode.OK == 200
    assert StatusCode.TOO_MANY_REQUESTS == 429
    assert StatusCode.CONTEXT_LIMIT == 601


def test_status_code_is_int_subclass():
    from ai_navigator.state.status import StatusCode
    assert isinstance(StatusCode.OK, int)
    assert isinstance(StatusCode.OK, StatusCode)
    assert isinstance(StatusCode.TOO_MANY_REQUESTS, StatusCode)


def test_status_code_singleton():
    from ai_navigator.state.status import StatusCode
    # Same object returned for same value
    assert StatusCode[200] is StatusCode.OK
    assert StatusCode[429] is StatusCode.TOO_MANY_REQUESTS


def test_status_code_index_lookup():
    from ai_navigator.state.status import StatusCode
    assert StatusCode[200] == 200
    assert StatusCode[429] == 429


def test_status_code_index_unknown_raises():
    from ai_navigator.state.status import StatusCode
    with pytest.raises(KeyError):
        StatusCode[9999]


def test_status_code_register():
    from ai_navigator.state.status import StatusCode, describe
    code = StatusCode.register(798, "Test Code")
    assert isinstance(code, StatusCode)
    assert code == 798
    assert StatusCode[798] == 798
    assert describe(798) == "Test Code"


def test_status_code_desc():
    from ai_navigator.state.status import StatusCode
    assert StatusCode.OK.desc == "Ok"
    assert StatusCode.TOO_MANY_REQUESTS.desc == "Too Many Requests"


def test_describe_unknown():
    from ai_navigator.state.status import describe
    assert describe(9998) == "Unknown"


# ── SchemaComposer ────────────────────────────────────────────────────────────

SAMPLE_YAML = """
meta:
  name: Review
  description: Product review extraction
  version: "1.0"
schema:
  title:
    type: str
    description: Product title
  rating:
    type: int
    description: Rating 1-5
  summary:
    type: str
    description: Brief summary
"""


def test_schema_composer_from_yaml():
    sc = SchemaComposer.from_yaml(SAMPLE_YAML)
    assert sc._name == "Review"
    assert sc._version == "1.0"
    assert list(sc._terms) == ["title", "rating", "summary"]


def test_schema_composer_requires_meta_and_schema():
    with pytest.raises(ValueError, match="meta"):
        SchemaComposer.from_yaml("name: Foo\nfields: []")


def test_schema_composer_schema_must_be_dict():
    bad = "meta:\n  name: X\nschema:\n  - name: foo\n    type: str\n"
    with pytest.raises(ValueError, match="dict"):
        SchemaComposer.from_yaml(bad)


def test_schema_conversion_flat():
    sc = SchemaComposer.from_yaml(SAMPLE_YAML)
    fmt = sc.schema_conversion()
    assert fmt["type"] == "json_schema"
    js = fmt["json_schema"]
    assert js["strict"] is True
    props = js["schema"]["properties"]
    assert props["title"] == {"type": "string", "description": "Product title"}
    assert props["rating"] == {"type": "integer", "description": "Rating 1-5"}
    assert set(js["schema"]["required"]) == {"title", "rating", "summary"}


def test_schema_conversion_enum():
    yaml_str = """
meta:
  name: X
  description: test
  version: "1.0"
schema:
  label:
    type: enum
    choices: [positive, negative, neutral]
"""
    sc = SchemaComposer.from_yaml(yaml_str)
    props = sc.schema_conversion()["json_schema"]["schema"]["properties"]
    assert props["label"] == {"type": "string", "enum": ["positive", "negative", "neutral"]}


def test_schema_conversion_nested_dict():
    yaml_str = """
meta:
  name: Analysis
  description: Nested
  version: "1.0"
schema:
  detail:
    type: dict
    terms:
      reason:
        type: str
      score:
        type: int
"""
    sc = SchemaComposer.from_yaml(yaml_str)
    props = sc.schema_conversion()["json_schema"]["schema"]["properties"]
    detail = props["detail"]
    assert detail["type"] == "object"
    assert detail["properties"]["reason"] == {"type": "string"}
    assert "score" in detail["required"]


def test_schema_conversion_anyof_type_list():
    yaml_str = """
meta:
  name: X
  description: test
  version: "1.0"
schema:
  note:
    type: [str, null]
    description: Optional note
"""
    sc = SchemaComposer.from_yaml(yaml_str)
    props = sc.schema_conversion()["json_schema"]["schema"]["properties"]
    assert props["note"] == {
        "anyOf": [{"type": "string"}, {"type": "null"}],
        "description": "Optional note",
    }


def test_schema_conversion_defs_and_ref():
    yaml_str = """
meta:
  name: X
  description: test
  version: "1.0"
defs:
  score_def:
    type: int
    description: Score 0-10
schema:
  title:
    type: str
  rating:
    ref: score_def
"""
    sc = SchemaComposer.from_yaml(yaml_str)
    fmt = sc.schema_conversion()
    inner = fmt["json_schema"]["schema"]
    assert inner["$defs"] == {"score_def": {"type": "integer", "description": "Score 0-10"}}
    assert inner["properties"]["rating"] == {"$ref": "#/$defs/score_def"}


def test_schema_conversion_enum_without_choices_raises():
    yaml_str = """
meta:
  name: X
  description: test
  version: "1.0"
schema:
  category:
    type: enum
"""
    sc = SchemaComposer.from_yaml(yaml_str)
    with pytest.raises(ValueError, match="choices"):
        sc.schema_conversion()


def test_preprocess_resolves_dynamic_choices():
    yaml_str = """
meta:
  name: X
  description: test
  version: "1.0"
schema:
  category:
    type: enum
    dynamic_choices: cats
    config_confidence: true
"""
    sc = SchemaComposer.from_yaml(yaml_str)
    resolved = sc.preprocess({"cats": ["electronics", "clothing"]})
    term = resolved._terms["category"]
    assert term["choices"] == ["electronics", "clothing"]
    assert "dynamic_choices" not in term
    assert term["type"] == "enum"


def test_preprocess_resolves_dynamic_type():
    yaml_str = """
meta:
  name: X
  description: test
  version: "1.0"
schema:
  score:
    dynamic_type: score_type
    description: A score
"""
    sc = SchemaComposer.from_yaml(yaml_str)
    resolved = sc.preprocess({"score_type": "int"})
    term = resolved._terms["score"]
    assert term["type"] == "int"
    assert "dynamic_type" not in term


def test_preprocess_resolves_dynamic_description():
    yaml_str = """
meta:
  name: X
  description: test
  version: "1.0"
schema:
  title:
    type: str
    dynamic_description: title_desc
"""
    sc = SchemaComposer.from_yaml(yaml_str)
    resolved = sc.preprocess({"title_desc": "The product title"})
    assert resolved._terms["title"]["description"] == "The product title"


def test_confidence_terms():
    yaml_str = """
meta:
  name: X
  description: test
  version: "1.0"
schema:
  sentiment:
    type: enum
    choices: [positive, negative, neutral]
    config_confidence: true
  title:
    type: str
"""
    sc = SchemaComposer.from_yaml(yaml_str)
    assert sc.confidence_terms() == {"sentiment": ["positive", "negative", "neutral"]}


def test_term_name_with_dot_raises():
    yaml_str = """
meta:
  name: X
  description: test
  version: "1.0"
schema:
  bad.name:
    type: str
"""
    sc = SchemaComposer.from_yaml(yaml_str)
    with pytest.raises(ValueError, match="dot"):
        sc.schema_conversion()


# ── ResultExtractor ───────────────────────────────────────────────────────────

NESTED_YAML = """
meta:
  name: X
  description: test
  version: "1.0"
schema:
  title:
    type: str
  detail:
    type: dict
    terms:
      reason:
        type: str
      score:
        type: int
  tags:
    type: list
  soldiers:
    type: list
"""


def test_result_extractor_flat():
    sc = SchemaComposer.from_yaml(SAMPLE_YAML)
    data = {"title": "Phone", "rating": 5, "summary": "Great"}
    result = ResultExtractor().extract(data, sc)
    assert result == {"title": "Phone", "rating": 5, "summary": "Great"}


def test_result_extractor_dict_expanded_by_default():
    sc = SchemaComposer.from_yaml(NESTED_YAML)
    data = {"title": "Phone", "detail": {"reason": "fast", "score": 9},
            "tags": ["a", "b"], "soldiers": ["Alice", "Bob"]}
    result = ResultExtractor().extract(data, sc)
    assert result == {
        "title": "Phone",
        "detail.reason": "fast",
        "detail.score": 9,
        "tags": ["a", "b"],
        "soldiers": ["Alice", "Bob"],
    }


def test_result_extractor_list_elements_flattened():
    sc = SchemaComposer.from_yaml(NESTED_YAML)
    data = {"title": "Phone", "detail": {"reason": "fast", "score": 9},
            "tags": ["speed", "price"], "soldiers": ["Alice", "Bob", "Carol"]}
    result = ResultExtractor().extract(
        data, sc, configs={"extract_list_elements": True}
    )
    assert result["tags_1"] == "speed"
    assert result["tags_2"] == "price"
    assert result["soldiers_1"] == "Alice"
    assert result["soldiers_3"] == "Carol"
    assert "tags" not in result
    assert "soldiers" not in result
    assert result["detail.reason"] == "fast"
    assert "detail" not in result


def test_result_extractor_discard_false_keeps_parent():
    sc = SchemaComposer.from_yaml(NESTED_YAML)
    data = {"title": "Phone", "detail": {"reason": "fast", "score": 9},
            "tags": ["a", "b"], "soldiers": ["X"]}
    result = ResultExtractor().extract(
        data, sc, configs={"term_extract_discard": False}
    )
    assert result["detail"] == {"reason": "fast", "score": 9}
    assert result["detail.reason"] == "fast"
    assert result["detail.score"] == 9


def test_result_extractor_discard_false_with_list():
    sc = SchemaComposer.from_yaml(NESTED_YAML)
    data = {"title": "X", "detail": {"reason": "ok", "score": 1},
            "tags": ["a", "b"], "soldiers": ["X"]}
    result = ResultExtractor().extract(
        data, sc, configs={"extract_list_elements": True, "term_extract_discard": False}
    )
    assert result["tags"] == ["a", "b"]
    assert result["tags_1"] == "a"
    assert result["tags_2"] == "b"


def test_result_extractor_empty_list():
    sc = SchemaComposer.from_yaml(NESTED_YAML)
    data = {"title": "X", "detail": {"reason": "ok", "score": 1},
            "tags": [], "soldiers": []}
    result = ResultExtractor().extract(
        data, sc, configs={"extract_list_elements": True}
    )
    assert "tags_1" not in result
    assert "tags" not in result
