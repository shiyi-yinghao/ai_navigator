# Parser & Schema 模块设计说明

## 当前架构

```
用户 YAML schema
      │
      ▼
SchemaComposer.from_yaml()
      │
      ├── preprocess(data_dict)    ← 解析所有 dynamic_* 属性
      │
      └── schema_conversion()     ← 输出 OpenAI response_format dict
                                     {"type": "json_schema", "json_schema": {...}}

LLM 调用（structured output）
      │
      ▼
ResponseParser.parse_response()  ← 从响应文本中提取 JSON

ResultExtractor.extract(data, composer, configs)
      │
      ├── 默认：dict 类型展开，list 原样保留
      │         结果是 flat dict，key 用 dot-notation（如 "detail.score"）
      │
      ├── configs["extract_list_elements"]=True
      │         list 展开为 term_1, term_2, ...
      │
      └── configs["term_extract_discard"]=False
                展开的同时保留父节点原始 key
```

## Schema YAML 格式

```
meta:                  必填 — name, description, version
defs:                  可选 — 可复用类型定义（$defs）
schema:                必填 — dict 格式，key 即字段名
  field_name:
    type:              str/int/float/bool/null/enum/list/dict/any
                       或 [str, null] 形式 → anyOf
    ref:               引用 defs 里的定义 → $ref
    config_*:          pkg 内部属性，不写入 JSON Schema
    config_confidence: 标记 enum 字段进行 logprob 提取（暂时下线）
    dynamic_*:         运行时从 data_dict 注入，preprocess() 后移除
```

## 路径分隔符约定

- 所有路径使用 `.`（点号）作为层级分隔符
- 字段名不允许含 `.`，在 `schema_conversion()` 时校验
- 数组展开后的 key 格式：`field_1`, `field_2`（下划线 + 1-based 序号）

## LogProb 支持（暂时下线）

`position.py` 和 `logprob.py` 移至 `_lab/`，不打包。
`config_confidence` 字段保留，待后续集成时直接启用。

重新集成步骤：
1. 将 `_lab/logprob.py` 和 `_lab/position.py` 移回 `src/ai_navigator/parser/`
2. 更新 `parser/__init__.py` 导出 `LogProbParser`, `JSONPositionParser` 等
3. `SchemaComposer.confidence_terms()` 已返回 `{path: [candidates]}`，直接传给 `LogProbParser`
