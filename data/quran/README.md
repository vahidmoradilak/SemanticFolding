---
language:
  - ar
  - bn
  - en
  - es
  - fr
  - id
  - ru
  - tr
  - zh
license: cc-by-4.0
tags:
  - text
  - multilingual
  - translation
  - quran
pretty_name: Multilingual Quran Dataset
---

# Multilingual Quran Dataset

A **multilingual Quran dataset** containing the original Arabic text along with translations in multiple languages: Bangla, English, Spanish, French, Indonesian, Russian, Turkish, and Chinese. Each verse includes metadata such as chapter number, verse number, chapter names, transliterations, and verse text in all available languages.

---

## Dataset Summary

- **Number of examples**: 6236 verses  
- **Languages**: `ar`, `bn`, `en`, `es`, `fr`, `id`, `ru`, `tr`, `zh`  
- **File formats**: JSONL, Parquet  
- **License**: CC BY 4.0  
- **Homepage / Source**: [Website](https://anisafifi.com)

---

## Dataset Structure

Each record in the dataset represents a single verse and contains the following fields:

| Field | Description |
|-------|-------------|
| `id` | Verse ID in the format `chapter:verse` (e.g., `1:1`) |
| `chapter_number` | Surah (chapter) number |
| `chapter_name` | Chapter name in Arabic |
| `verse_number` | Verse number within the chapter |
| `chapter_type` | `meccan` or `medinan` |
| `total_verses` | Total number of verses in the chapter |
| `text_ar` | Original Arabic text of the verse |
| `text_bn` | Verse translation in Bangla |
| `text_en` | Verse translation in English |
| `text_es` | Verse translation in Spanish |
| `text_fr` | Verse translation in French |
| `text_id` | Verse translation in Indonesian |
| `text_ru` | Verse translation in Russian |
| `text_tr` | Verse translation in Turkish |
| `text_zh` | Verse translation in Chinese |

---

## Usage

### Load the dataset using `pandas`:

```python
import pandas as pd

# JSONL version
df = pd.read_json("quran_combined.jsonl", lines=True)
print(df.head())

# Parquet version
df_parquet = pd.read_parquet("quran_combined.parquet")
print(df_parquet.head())
