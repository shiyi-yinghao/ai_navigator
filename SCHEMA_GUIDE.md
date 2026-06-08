# Schema 用户手册

`SchemaComposer` 和 `ResultExtractor` 是 ai-navigator 结构化输出的核心模块。前者负责把 YAML 定义转成 LLM 可用的 JSON Schema，后者负责把 LLM 返回的 JSON 映射成扁平的结果字典。

---

## 目录

1. [YAML 格式总览](#1-yaml-格式总览)
2. [meta 部分](#2-meta-部分)
3. [defs 部分（可选）](#3-defs-部分可选)
4. [schema 部分](#4-schema-部分)
5. [字段类型详解](#5-字段类型详解)
6. [anyOf：多类型字段](#6-anyof多类型字段)
7. [ref：引用 defs](#7-ref引用-defs)
8. [dynamic_* 动态属性](#8-dynamic_-动态属性)
9. [config_confidence：置信度提取](#9-config_confidence置信度提取)
10. [字段名约束](#10-字段名约束)
11. [SchemaComposer API](#11-schemacomposer-api)
12. [ResultExtractor API](#12-resultextractor-api)
13. [完整工作流示例](#13-完整工作流示例)
14. [与 LogProbParser 配合](#14-与-logprobparser-配合)
15. [常见错误与排查](#15-常见错误与排查)

---

## 1. YAML 格式总览

一个完整的 schema 文件由三个顶层 key 组成：

```yaml
meta:       # 必填 — 描述这份 schema 的元信息
  ...

defs:       # 可选 — 可复用的类型定义
  ...

schema:     # 必填 — 实际的输出字段定义
  ...
```

`meta` 和 `schema` 是必须存在的；`defs` 可以省略。

---

## 2. meta 部分

```yaml
meta:
  name: ProductReview          # 用于 JSON Schema 的 name 字段（必填）
  description: 提取商品评论信息  # 任务描述，写进 JSON Schema description（可选）
  version: "1.0"               # 版本号，仅供记录，不影响输出（可选）
```

| 字段 | 是否必填 | 说明 |
|---|---|---|
| `name` | 建议填写 | 会被规范化（非字母数字转成 `_`），截断到 64 字符 |
| `description` | 可选 | 写进 LLM 的 JSON Schema 描述，有助于引导模型 |
| `version` | 可选 | 纯记录用途 |

---

## 3. defs 部分（可选）

`defs` 用来定义可复用的类型片段，避免在多个字段间重复定义相同结构。

```yaml
defs:
  score_range:                 # def 名称（即 $defs 里的 key）
    type: int
    description: 0 到 10 的评分

  address:
    type: dict
    terms:
      street:
        type: str
      city:
        type: str
      postcode:
        type: [str, null]      # defs 内部也支持所有类型特性
```

定义好之后，在 `schema` 里用 `ref: def_name` 引用（见 [第 7 节](#7-ref引用-defs)）。

输出的 JSON Schema 中，`defs` 内容会出现在 `$defs` 里：

```json
{
  "type": "object",
  "$defs": {
    "score_range": {"type": "integer", "description": "0 到 10 的评分"},
    "address": { "type": "object", "properties": {...}, ... }
  },
  "properties": { ... }
}
```

> `defs` 同样支持 `dynamic_*` 属性，会在 `preprocess()` 阶段一并解析。

---

## 4. schema 部分

`schema` 是一个 **dict**，key 是字段名，value 是字段的 spec。

```yaml
schema:
  title:                       # 字段名作为 key
    type: str
    description: 商品标题

  rating:                      # 另一个字段
    type: int
```

> **注意**：字段名不能包含 `.`（点号是路径分隔符，见 [第 10 节](#10-字段名约束)）。

嵌套字段（`dict` 类型）同样使用 dict 格式：

```yaml
schema:
  detail:
    type: dict
    terms:                     # 子字段，格式与 schema 完全相同
      reason:
        type: str
      score:
        type: int
```

---

## 5. 字段类型详解

### 5.1 标量类型

| YAML `type` 值 | JSON Schema 输出 | 说明 |
|---|---|---|
| `str` / `string` / `free-text` | `{"type": "string"}` | 字符串 |
| `int` / `integer` | `{"type": "integer"}` | 整数 |
| `float` / `number` | `{"type": "number"}` | 浮点数 |
| `bool` / `boolean` | `{"type": "boolean"}` | 布尔值 |
| `null` | `{"type": "null"}` | 空值（通常配合 anyOf 使用） |
| `any` | `{}` | 无类型约束 |

### 5.2 enum 类型

用于限定值只能是给定选项之一。必须提供 `choices`。

```yaml
schema:
  sentiment:
    type: enum
    choices: [正面, 负面, 中性]
    description: 情感倾向
    config_confidence: true    # pkg-internal，见第 9 节
```

JSON Schema 输出：
```json
{
  "type": "string",
  "enum": ["正面", "负面", "中性"],
  "description": "情感倾向"
}
```

### 5.3 list 类型

用于输出数组。可以指定元素类型和/或限定元素范围。

```yaml
schema:
  tags:
    type: list
    item_type: str             # 元素类型：str / int / float / bool（默认 str）
    description: 标签列表

  aspect:
    type: list
    item_type: str
    choices: [价格, 质量, 物流]  # 限定元素只能是给定值之一
```

JSON Schema 输出（有 choices）：
```json
{
  "type": "array",
  "items": {"type": "string", "enum": ["价格", "质量", "物流"]}
}
```

### 5.4 dict 类型

用于输出嵌套对象。子字段通过 `terms:` 定义，格式与顶层 `schema` 完全相同，支持递归嵌套。

```yaml
schema:
  analysis:
    type: dict
    description: 详细分析
    terms:
      reason:
        type: str
        description: 主要原因
      score:
        type: int
        description: 综合评分
      sub_detail:              # 可以继续嵌套
        type: dict
        terms:
          keyword:
            type: str
```

JSON Schema 输出：
```json
{
  "type": "object",
  "description": "详细分析",
  "properties": {
    "reason": {"type": "string", "description": "主要原因"},
    "score":  {"type": "integer", "description": "综合评分"},
    "sub_detail": {
      "type": "object",
      "properties": {"keyword": {"type": "string"}},
      "required": ["keyword"],
      "additionalProperties": false
    }
  },
  "required": ["reason", "score", "sub_detail"],
  "additionalProperties": false
}
```

> **所有字段均隐式 required**：YAML 里不需要写 `required: true`，转换时会自动把所有字段加入 JSON Schema 的 `required` 数组（OpenAI strict 模式要求）。

---

## 6. anyOf：多类型字段

把 `type` 写成列表，输出 `anyOf` 结构。最常见的用法是允许空值：

```yaml
schema:
  note:
    type: [str, null]          # 可选字符串
    description: 备注（可为空）

  score_or_na:
    type: [int, null]          # 可选整数
```

JSON Schema 输出：
```json
{
  "description": "备注（可为空）",
  "anyOf": [
    {"type": "string"},
    {"type": "null"}
  ]
}
```

> **支持范围**：anyOf 列表内只允许标量类型（`str`、`int`、`float`、`bool`、`null`）。
> `enum`、`list`、`dict` 不支持放在列表里。

YAML 中 `null`（不加引号）会被解析成 Python `None`，已自动处理，无需加引号。

---

## 7. ref：引用 defs

在 `schema` 或 `terms` 中用 `ref: def_name` 引用 `defs` 里的定义：

```yaml
defs:
  score_def:
    type: int
    description: 评分 0-10
  address_def:
    type: dict
    terms:
      city:
        type: str
      postcode:
        type: str

schema:
  rating:
    ref: score_def             # 引用

  shipping:
    ref: address_def           # 引用复杂类型

  billing:
    ref: address_def           # 同一个 def 可多次引用
```

JSON Schema 输出：
```json
{
  "properties": {
    "rating":   {"$ref": "#/$defs/score_def"},
    "shipping": {"$ref": "#/$defs/address_def"},
    "billing":  {"$ref": "#/$defs/address_def"}
  },
  "$defs": {
    "score_def": {"type": "integer", "description": "评分 0-10"},
    "address_def": { ... }
  }
}
```

> `ref` 字段不支持同时指定 `type` 或 `description`——`$ref` 替换整个 schema 片段。

---

## 8. dynamic_* 动态属性

### 原理

任何字段属性都可以变成"运行时注入"的形式，方法是加上 `dynamic_` 前缀：

```
dynamic_{attr}: lookup_key
```

在调用 `preprocess(data_dict)` 时：
1. 找到所有 `dynamic_*` 键。
2. 对每个 `dynamic_{attr}`，从 `data_dict[lookup_key]` 取值。
3. 将值写入 `term[attr]`，删除 `dynamic_*` 键。
4. **处理顺序优先于所有其他逻辑**，所以即使替换的是 `type`，后续类型处理也完全正确。

原始 spec 不会被修改，`preprocess()` 返回一个新的 `SchemaComposer`。

### 常见用法

**动态描述**（`dynamic_description`）
```yaml
schema:
  title:
    type: str
    dynamic_description: title_desc_key  # data_dict["title_desc_key"] → str
```

**动态枚举选项**（`dynamic_choices`）
```yaml
schema:
  category:
    type: enum
    dynamic_choices: cat_list_key        # data_dict["cat_list_key"] → list[str]
    config_confidence: true
```

**动态列表选项**（`dynamic_choices` 对 list 同样适用）
```yaml
schema:
  aspects:
    type: list
    item_type: str
    dynamic_choices: aspect_list_key
```

**动态子字段结构**（`dynamic_terms`）
```yaml
schema:
  detail:
    type: dict
    dynamic_terms: detail_spec_key       # data_dict["detail_spec_key"] → dict
```

**动态类型**（`dynamic_type`）
```yaml
schema:
  value:
    dynamic_type: value_type_key         # data_dict["value_type_key"] → "str" / "int" 等
    description: 动态类型字段
```

**任意属性均可动态化**

规则是通用的——不限于以上列举的例子：

```yaml
schema:
  tags:
    type: list
    dynamic_item_type: item_type_key     # data_dict["item_type_key"] → "str" / "int"
```

### 代码示例

```python
from ai_navigator.schema.composer import SchemaComposer

yaml_str = """
meta:
  name: SentimentAnalysis
  description: 情感分析
  version: "1.0"
schema:
  sentiment:
    type: enum
    dynamic_choices: sentiment_labels
    config_confidence: true
  aspect:
    type: list
    item_type: str
    dynamic_choices: aspect_labels
"""

sc = SchemaComposer.from_yaml(yaml_str)

# 运行时注入
resolved = sc.preprocess({
    "sentiment_labels": ["正面", "负面", "中性"],
    "aspect_labels":    ["价格", "质量", "物流", "服务"],
})

# 此时 resolved._terms["sentiment"]["choices"] == ["正面", "负面", "中性"]
fmt = resolved.schema_conversion()
```

---

## 9. config_confidence：置信度提取

给 `enum` 字段加上 `config_confidence: true`，表示希望对这个字段做 logprob 级别的概率分布提取。

```yaml
schema:
  sentiment:
    type: enum
    choices: [正面, 负面, 中性]
    config_confidence: true    # pkg-internal：标记为需要置信度提取
  rating:
    type: int                  # 没有 config_confidence，正常字段
```

`config_confidence` 是 `config_*` 约定下的第一个内部属性——所有 `config_*` 属性均不会出现在 `schema_conversion()` 输出的 JSON Schema 里，仅供 pkg 内部使用。`config_confidence` 的值仅通过 `confidence_terms()` 暴露。

```python
ct = sc.confidence_terms()
# → {"sentiment": ["正面", "负面", "中性"]}
# key 是 dot-notation 路径，value 是候选列表
```

嵌套字段同样支持：

```yaml
schema:
  detail:
    type: dict
    terms:
      sentiment:
        type: enum
        choices: [正面, 负面]
        config_confidence: true
```

```python
sc.confidence_terms()
# → {"detail.sentiment": ["正面", "负面"]}
```

---

## 10. 字段名约束

**字段名（`schema` 的 key、`terms` 的 key、`defs` 的 key）不能包含 `.`（点号）。**

原因：点号被保留用于表示叶子节点路径（如 `detail.score`）。如果字段名含点，路径会产生歧义。

验证时机：`schema_conversion()` 调用时检查，违反则抛出 `SchemaError`。

```python
# 错误示例
yaml_str = """
meta:
  name: X
  description: test
  version: "1.0"
schema:
  bad.name:      # ← 含点，会在 schema_conversion() 时报错
    type: str
"""
sc = SchemaComposer.from_yaml(yaml_str)
sc.schema_conversion()  # → SchemaError: Term name 'bad.name' contains a dot.
```

---

## 11. SchemaComposer API

### 构造

```python
from ai_navigator.schema.composer import SchemaComposer

# 从字符串
sc = SchemaComposer.from_yaml(yaml_str)

# 从文件
sc = SchemaComposer.from_yaml_file("path/to/schema.yaml")
```

`from_yaml` 会校验：
- 根节点是 dict
- `meta` 和 `schema` 均存在
- `schema` 是 dict（不是 list）
- `defs` 如果存在，也必须是 dict

### `preprocess(data_dict)`

```python
resolved = sc.preprocess(data_dict)
# data_dict: dict[str, Any]，提供所有 dynamic_* 所需的值
# 返回新的 SchemaComposer，原始 sc 不变
```

如果没有任何 `dynamic_*` 字段，传 `None` 或直接不调用都可以。

### `schema_conversion(task_name=None)`

```python
response_format = sc.schema_conversion()
# 或指定任务名（覆盖 meta.name）
response_format = sc.schema_conversion(task_name="my_task")
```

返回值可直接用作 LLM 调用的 `response_format` 参数：

```python
response = llm.response(
    "分析这条评论的情感",
    response_format=sc.schema_conversion(),
)
```

输出结构：
```python
{
    "type": "json_schema",
    "json_schema": {
        "name": "ProductReview",
        "strict": True,
        "description": "...",          # 来自 meta.description（如有）
        "schema": {
            "type": "object",
            "$defs": { ... },           # 来自 defs（如有）
            "properties": { ... },
            "required": [...],
            "additionalProperties": False,
        }
    }
}
```

### `confidence_terms()`

返回所有标记了 `config_confidence: true` 的 enum 字段的候选列表，格式为 `{dot_path: [choices]}`。

```python
sc.confidence_terms()
# → {"sentiment": ["正面", "负面", "中性"], "detail.tone": ["强烈", "温和"]}
```

### `build_prompt_instruction()`

生成可放入系统 prompt 的纯文本描述，适用于不支持 structured output 的场景。

```python
print(sc.build_prompt_instruction())
# Respond ONLY with valid JSON matching schema: ProductReview
# Description: 提取商品评论信息
#
# Terms:
#   - title (str): 商品标题
#   - sentiment (enum) choices=['正面', '负面', '中性']: 情感倾向
#   - detail (dict):
#       - reason (str): 原因
#       - score (int): 评分
```

---

## 12. ResultExtractor API

`ResultExtractor` 把 LLM 返回的嵌套 JSON dict 映射成扁平结果字典。提取行为由**解析类型集合**（parse types）决定，而不是固定的叶子/非叶子规则。

### 解析类型集合

| 解析类型 | 激活条件 | 展开方式 |
|---|---|---|
| `dict` | 始终激活 | 递归展开子字段，key 变为 dot-notation 路径（如 `detail.score`） |
| `list` | `params["extract_list_elements"] = True` | 展开数组元素，key 变为 `字段名_序号`（1 起始） |

不在解析类型集合中的类型，值原样写入结果，不做任何处理。

### `term_extract_discard`（默认 `True`）

当一个 term 被展开（递归或拆分）时，`term_extract_discard` 控制**原字段 key 是否保留**：

| 值 | 行为 |
|---|---|
| `True`（默认）| 展开后**丢弃**原字段 key，结果中只有子字段 / 拆分后的 key |
| `False` | 展开后**保留**原字段 key（值为原始内容），同时结果中也包含展开的子字段 |

### `extract(data, composer, configs=None)`

```python
from ai_navigator.schema.extractor import ResultExtractor

extractor = ResultExtractor()

data = {
    "title":    "充电宝",
    "detail":   {"reason": "容量大", "score": 9},
    "tags":     ["便携", "大容量"],
    "soldiers": ["Alice", "Bob"],
}
```

**默认（dict 展开，原字段丢弃）**

```python
extractor.extract(data, composer)
# → {
#     "title":         "充电宝",
#     "detail.reason": "容量大",          ← dict 展开，"detail" 被丢弃
#     "detail.score":  9,
#     "tags":          ["便携", "大容量"], ← list 未激活，原样保留
#     "soldiers":      ["Alice", "Bob"],
# }
```

**保留原字段（`term_extract_discard=False`）**

```python
extractor.extract(data, composer, configs={"term_extract_discard": False})
# → {
#     "title":         "充电宝",
#     "detail":        {"reason": "容量大", "score": 9},  ← 保留
#     "detail.reason": "容量大",
#     "detail.score":  9,
#     "tags":          ["便携", "大容量"],
#     "soldiers":      ["Alice", "Bob"],
# }
```

**开启 list 展开（原字段丢弃）**

```python
extractor.extract(data, composer, configs={"extract_list_elements": True})
# → {
#     "title":         "充电宝",
#     "detail.reason": "容量大",
#     "detail.score":  9,
#     "tags_1":        "便携",    ← 展开，"tags" 被丢弃
#     "tags_2":        "大容量",
#     "soldiers_1":    "Alice",
#     "soldiers_2":    "Bob",
# }
```

**开启 list 展开并保留原字段**

```python
extractor.extract(data, composer,
                  configs={"extract_list_elements": True,
                           "term_extract_discard": False})
# → {
#     ...
#     "tags":   ["便携", "大容量"],  ← 保留
#     "tags_1": "便携",
#     "tags_2": "大容量",
# }
```

空列表展开时不产生任何 `_N` key；若 `term_extract_discard=False`，空列表仍会以原 key 写入结果（值为 `[]`）。

### 与 RequestState 配合

```python
result = extractor.extract(
    data,
    state.reference["schema"],
    configs=state.configs,   # 包含 term_extract_discard, extract_list_elements 等
)
state.result = result
```

---

## 13. 完整工作流示例

以一个中文商品评论情感分析任务为例：

### schema.yaml

```yaml
meta:
  name: SentimentAnalysis
  description: 从商品评论中提取结构化情感信息
  version: "2.0"

defs:
  confidence_score:
    type: float
    description: 置信度分数，0.0 到 1.0

schema:
  sentiment:
    type: enum
    choices: [正面, 负面, 中性]
    description: 整体情感倾向
    config_confidence: true

  confidence:
    ref: confidence_score

  aspects:
    type: list
    item_type: str
    dynamic_choices: aspect_list   # 运行时注入候选方面

  detail:
    type: dict
    terms:
      reason:
        type: str
        description: 判断依据
      keywords:
        type: list
        item_type: str
        description: 关键词列表
      score:
        type: [int, null]
        description: 综合评分，无法判断时为 null
```

### Python 代码

```python
from ai_navigator.schema.composer import SchemaComposer
from ai_navigator.schema.extractor import ResultExtractor
from ai_navigator.parser.response import ResponseParser
from ai_navigator.server import OpenAIServer

# 1. 加载 schema
sc = SchemaComposer.from_yaml_file("schema.yaml")

# 2. 注入动态内容（preprocess 返回新对象，原 sc 不变）
resolved = sc.preprocess({
    "aspect_list": ["价格", "质量", "物流", "售后"],
})

# 3. 准备 response_format（用于 LLM 调用）
response_format = resolved.schema_conversion()

# 4. 调用 LLM
llm = OpenAIServer("gpt-4o", credentials={"api_key": "..."})
response = llm.response(
    "这款充电宝性价比超高，快递也很快，就是包装有点简陋。",
    system="你是专业的商品评论分析助手。",
    response_format=response_format,
    logprobs=True,
    top_logprobs=5,
)

# 5. 解析 JSON
parser = ResponseParser()
data = parser.parse_response(response)
# data = {
#   "sentiment": "正面",
#   "confidence": 0.87,
#   "aspects": ["价格", "物流"],
#   "detail": {
#     "reason": "性价比高，物流快",
#     "keywords": ["性价比", "快递", "包装"],
#     "score": 8
#   }
# }

# 6. 提取结果（dict 自动展开，list 默认保留整体）
extractor = ResultExtractor()
result = extractor.extract(data, resolved)
# result = {
#   "sentiment":       "正面",
#   "confidence":      0.87,
#   "aspects":         ["价格", "物流"],       ← list 保留整体
#   "detail.reason":   "性价比高，物流快",
#   "detail.keywords": ["性价比", "快递", "包装"],
#   "detail.score":    8,
# }

# 若需要把 list 也展开：
result = extractor.extract(data, resolved,
                           configs={"extract_list_elements": True})
# "aspects_1": "价格", "aspects_2": "物流"
# "detail.keywords_1": "性价比", ...
```

---

## 14. 与 LogProbParser 配合（暂时下线）

> **LogProbParser 目前不在包内**（源码保留在 `_lab/logprob.py` 和 `_lab/position.py`，
> 不会被打包发布）。`config_confidence` 字段保留，后续集成时直接启用。

`confidence_terms()` 返回 `{dot_path: [candidates]}`，供未来 LogProbParser 使用：

```python
candidates = resolved.confidence_terms()
# → {"sentiment": ["正面", "负面", "中性"]}
# 将来传给 LogProbParser.extract_enum_probs(response.raw, candidates)
```

---

## 15. 常见错误与排查

### `SchemaError: YAML must have top-level 'meta' and 'schema' keys`

YAML 根节点缺少 `meta` 或 `schema`，或两者都缺失。检查缩进和拼写。

### `SchemaError: 'schema' must be a dict`

`schema:` 使用了旧的 list 格式（`- name: xxx`）。改成 dict 格式：

```yaml
# 错误
schema:
  - name: title
    type: str

# 正确
schema:
  title:
    type: str
```

### `SchemaError: Term 'xxx' has type 'enum' but no 'choices' defined`

`enum` 类型必须有 `choices`。如果是运行时注入，需要先调用 `preprocess(data_dict)`，确保 `data_dict` 里有对应的 key。

### `SchemaError: Term name 'xxx' contains a dot`

字段名含有点号。把 `a.b` 这样的字段名改成 `a_b` 或者用嵌套 `dict` 表达层级关系。

### `SchemaError: Type 'xxx' in list type for term 'yyy' is not supported inside anyOf`

anyOf 列表里只允许标量类型（`str`、`int`、`float`、`bool`、`null`）。`enum`、`list`、`dict` 不支持。

### dynamic_* 没有生效

- 确认调用了 `preprocess(data_dict)` 并使用了返回值（原对象不变）。
- 确认 `data_dict` 里 key 名与 YAML 中 `dynamic_xxx` 的值完全匹配。
- `dynamic_*` 在 `preprocess` 里只在 key 存在时才赋值；key 不存在时静默跳过。
