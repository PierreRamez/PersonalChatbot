# Data Preprocessing Documentation

**Project:** Personalized Chatbot (Finetuning Dataset)
**Notebook:** `dialogues_preprocessing.ipynb`
**Purpose:** Provide a clear, reproducible, professional explanation of how raw multi-source dialog data was transformed into a clean JSONL dataset suitable for LLM finetuning. This document is intended for the project team, presentation reviewers, and judges.

---

## Summary

We began with noisy multi-source conversational data from three Kaggle datasets containing inconsistent quoting, escaped versus real newline artifacts, mojibake (encoding errors), fused conversational turns, and no explicit role labels. Through a sequence of controlled preprocessing steps — converting each dialog row into structured messages, fixing encoding issues, removing quote artifacts, and applying a minimal cleanup (including a surgical apostrophe fix) — we produced a clean, finetune-ready JSONL file.

**Datasets used:**

1. Mental Health Conversations Dataset
   [https://www.kaggle.com/code/jocelyndumlao/chatbot-for-mental-health-conversations](https://www.kaggle.com/code/jocelyndumlao/chatbot-for-mental-health-conversations)

2. Daily Dialog (Multi-turn Dialog)
   [https://www.kaggle.com/datasets/thedevastator/dailydialog-unlock-the-conversation-potential-in/data](https://www.kaggle.com/datasets/thedevastator/dailydialog-unlock-the-conversation-potential-in/data)

3. Human Bot Chat Generator
   [https://www.kaggle.com/code/projjal1/human-bot-conversation-generator/input](https://www.kaggle.com/code/projjal1/human-bot-conversation-generator/input)

---

## Initial State of Raw Data

When the data was first combined and imported into the notebook, it contained:

* **Mixed quote styles**: `'`, `"`, and hybrid combinations used inconsistently.
* **Escaped vs real newlines**: some dialogs contained actual newlines (`\n`), others contained literal string sequences (`\\n`).
* **Fused turns**: multiple utterances sometimes appeared in a single line separated by artifacts like
  `' '` or `" '"`.
* **Mojibake / encoding issues** such as:
  `I â m`, `you â re`, `ã`, etc.
* **Stray quote clusters** and duplicated quotes:
  `\" '`, `' '`, `''`, `""`.
* **No explicit role labels**: dialogs alternated turns but had no speaker metadata.
* **Whitespace and punctuation anomalies**: spaces before periods, commas, or around apostrophes.

These issues made the data unsuitable for direct use in LLM finetuning.

---

## Final State of Data (Output)

After preprocessing, the final file was:

`processed_dialogues_apostrophes_fixed.jsonl`

Structure of each line:

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ]
}
```

This JSONL format is stable, UTF-8 clean, and directly usable for finetuning.

---
<br>
## Preprocessing Principles

* **Minimalism**: Only remove noise; do not alter linguistic meaning or rewrite sentences.
* **Determinism**: Every transformation is explicit and reproducible.
* **Safety**: Ensure JSONL validity and predictable alternating roles.
* **Faithfulness**: Preserve the original dialog content except where noise removal is unavoidable.

---

## Full Preprocessing Pipeline (Step-by-Step)

### 1. Environment Setup

```python
import pandas as pd
import json, re
from ftfy import fix_text
```

`ftfy` was used specifically to fix mojibake and encoding errors.

---

### 2. Confirming DataFrame Integrity

We verified the preprocessing output already existed in memory as `df` and examined the available columns:

```python
df.columns
```

The output confirmed:

```
Index(['dialog', 'dialog_str', 'new_dialog_str'], dtype='object')
```

We selected `new_dialog_str` as our canonical source column for all further processing.

---

### 3. Splitting Dialog Rows into Turns (`row_to_messages`)

Each row contained multiple turns, usually separated by newline characters or escaped newline sequences.

Processing steps:

1. Replace `\r\n` with `\n` for consistency.
2. Convert literal `\\n` to real `\n` when necessary.
3. Split into lines using `\n`.
4. Detect and split further when fused lines contained quote artifacts (e.g., `' '`).
5. Clean each line using `clean_line`.
6. Assign roles using **strict alternation**:

   * Index 0 → `user`
   * Index 1 → `assistant`
   * Index 2 → `user`
   * …
7. Preserve final odd turns as user-only messages.

This preserved dialog structure without inventing speaker metadata.

---
<br><br>
### 4. Minimal Cleaning per Utterance (`clean_line` / `clean_minimal`)

Each utterance went through the following deterministic cleanup:

* **Unicode fixes with ftfy:**
  Repairs mojibake such as
  `â → ’`, `ã → 。`, etc.

* **Strip outer quotes:**
  Remove unwanted leading/trailing `'` or `"`.

* **Remove quote artifacts:**
  Replace clusters like `" '`, `' '`, `''`, `""` with a single space.

* **Normalize whitespace:**
  Collapse multiple spaces into one.

* **Fix punctuation spacing:**

  * Remove space before punctuation (`word .` → `word.`)
  * Ensure one space after punctuation when necessary (`word.Next` → `word. Next`)

* **Decode escaped content:**
  Convert `It\'s` → `It's` safely.

This represented **Minimal Cleanup Level A** — safe, predictable, nondestructive.

---

### 5. Exporting Initial JSONL

We exported the structured but not-yet-perfect dialogs:

```python
with open('/content/processed_dialogues.jsonl','w',encoding='utf-8') as fout:
    for idx, row in df.iterrows():
        messages = row_to_messages(row['new_dialog_str'])
        json.dump({"messages": messages}, fout, ensure_ascii=False)
        fout.write("\n")
```

This file was close to usable but still contained spaced apostrophes and a few minor artifacts.

---

### 6. Minimal Cleaning Pass on JSONL

We applied `clean_minimal` again, but this time to every `message["content"]` directly from the JSONL, producing:

`processed_dialogues_cleaned.jsonl`

This fixed most remaining quote artifacts and spacing anomalies.

---

### 7. Final Fix: Apostrophe Normalization

Remaining issue:
Contractions of the form:

```
I ' m
You ' re
It ' s
That ' s
We ' ll
Don ' t
```

To solve this, we applied a **surgical regex** that only collapses apostrophes between alphanumeric characters:

```python
APOSTROPHE_FIX = re.compile(r"(\w)\s*'\s*(\w)")

def fix_apostrophes(text):
    return APOSTROPHE_FIX.sub(r"\1'\2", text)
```
<br><br>
This pass produced the final dataset:

`processed_dialogues_apostrophes_fixed.jsonl`

---

## Explicit Problem → Resolution Examples

| <u>Problem</u>                            | <u>Example</u>                  | <u>Fix</u>              |
| -----------------------------------| -------------------------------------- | ---------------------- |
| **Literal escaped newlines**       | `\\n`                                  | Convert to real `\n`   |
| **Fused utterances**              | `music . ' ' What is the difference ?` | Split on `' '`         |
| **Mojibake**                      | `I â m`                              | `I'm`                  |
| **Stray quote clusters**          | `" ' All right .`                      | Remove and normalize   |
| **Spaced apostrophes**            | `I ' m`                                | `I'm`                  |
| **Space before punctuation**      | `word .`                               | `word.`                |
| **No role labels**                | —                                      | Use strict alternation |

---

## Validation Steps

We performed the following sanity checks:

1. **Column integrity:** confirmed `new_dialog_str` existed.
2. **Row structure:** ensured each `messages` list alternated roles and was non-empty.
3. **Encoding:** verified UTF-8 correctness using `ensure_ascii=False`.
4. **Manual inspection:** looked at the first ~20 lines after each pass.

Each intermediate file was preserved for auditing.

---

## Tools Actually Used

* `pandas`
* `json`
* `re`
* `ftfy`
* Custom Python functions (`clean_line`, `row_to_messages`, `df_to_jsonl`, apostrophe fix)

No speculative text generation was used. No rewriting or summarization. Only cleaning and structural normalization.

---

## Files Produced

1. **`processed_dialogues.jsonl`**

   * Basic conversion (row → messages).

2. **`processed_dialogues_cleaned.jsonl`**

   * Minimal artifact cleanup.

3. **`processed_dialogues_apostrophes_fixed.jsonl`**

   * Final dataset used for finetuning.

---

## Final Remarks

* The dataset remains faithful to the original Kaggle sources.
* All changes were purely structural or encoding-related — not semantic.
* The final JSONL is clean, consistent, and ready for LLM finetuning.
* Every step is transparent and replicable.



