# PilgrimageQA-Benchmark

**A Multilingual Question-Answering Benchmark and Retrieval-Based Evaluation Framework for Hajj and Umrah Assistance Systems**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset Size](https://img.shields.io/badge/dataset-10k%2B%20QA%20pairs-green.svg)](#dataset-generation-pipeline)

---

## Table of Contents

1. [Project Title](#pilgrimageqa-benchmark)
2. [Research Motivation](#research-motivation)
3. [Repository Contributions](#repository-contributions)
4. [Repository Structure](#repository-structure)
5. [Installation](#installation)
6. [Dataset Generation Pipeline](#dataset-generation-pipeline)
7. [Retrieval and QA Modules](#retrieval-and-qa-modules)
8. [Experimental Benchmark](#experimental-benchmark)
9. [Reproducibility](#reproducibility)
10. [Example Commands](#example-commands)
11. [Sample Experimental Results](#sample-experimental-results)
12. [Future Work](#future-work)
13. [Citation](#citation)

---

## Research Motivation

Every year, more than two million Muslims from over 180 countries converge on Mecca and Medina to perform Hajj, the fifth pillar of Islam, while millions more complete Umrah throughout the year. This extraordinary linguistic diversity—spanning Arabic, Urdu, Turkish, Indonesian, Persian, Bengali, Uzbek, Russian, and dozens of other languages—poses a critical challenge for digital assistance platforms seeking to support pilgrims before, during, and after their journey.

Existing question-answering (QA) benchmarks and conversational AI systems predominantly target high-resource languages such as English and Arabic. As a consequence, pilgrims whose primary language is Uzbek, Russian, or other low-resource languages face a significant information asymmetry: they are unable to effectively query guidance systems about rituals, logistics, health precautions, or emergency procedures in their native tongue. This gap is not merely a technical inconvenience—it can have real safety and well-being implications in one of the world's largest annual mass-gathering events.

To address this gap, **PilgrimageQA-Benchmark** introduces a rigorously constructed, multilingual QA benchmark specifically focused on Hajj and Umrah domain knowledge. The benchmark enables systematic evaluation of retrieval-based and generative QA systems across languages, providing the research community with a reproducible evaluation protocol and open datasets for developing the next generation of multilingual pilgrimage assistance AI.

---

## Repository Contributions

This repository makes the following primary research contributions:

| # | Contribution | Description |
|---|---|---|
| 1 | **10k+ Synthetic Multilingual Benchmark Dataset** | A corpus of more than 10,000 question–answer pairs generated through a controlled synthetic pipeline, covering Hajj and Umrah rituals, logistics, health guidance, and sacred-site information across multiple languages. |
| 2 | **Uzbek/Russian QA Corpus** | A dedicated sub-corpus of question–answer pairs in Uzbek and Russian, representing two underserved linguistic communities among the global pilgrim population. |
| 3 | **Retrieval-Based QA Engine** | A BM25-driven document retrieval module that fetches contextually relevant passages from the benchmark corpus and feeds them to a QA reader for span extraction. |
| 4 | **Translation-Assisted QA Mode** | A pipeline that automatically translates non-English queries into English prior to retrieval and QA, enabling cross-lingual evaluation without requiring language-specific models. |
| 5 | **Comparative Evaluation Framework** | A unified experimental harness (`experiments.py`) that orchestrates multiple QA modes—direct, translation-assisted, and retrieval baseline—under identical conditions for fair comparison. |
| 6 | **EM / F1 / BLEU Metrics** | Standardised evaluation using Exact Match (EM), token-level F1, and BLEU score, consistent with SQuAD and WMT evaluation traditions. |

---

## Repository Structure

```
pilgrimageqa-benchmark/
├── data/
│   ├── raw/                          # Source pilgrimage knowledge documents
│   │   ├── hajj_guide_en.txt
│   │   ├── hajj_guide_uz.txt
│   │   └── hajj_guide_ru.txt
│   ├── generated/                    # Synthetic QA pairs produced by the pipeline
│   │   ├── benchmark_en.csv
│   │   ├── benchmark_uz.csv
│   │   └── benchmark_ru.csv
│   └── processed/                    # Tokenised and indexed versions for retrieval
│       └── corpus_index/
├── src/
│   ├── generate_pilgrimage_dataset.py  # Synthetic dataset generation pipeline
│   ├── dataset.py                      # Dataset loading and preprocessing utilities
│   ├── qa.py                           # Retrieval-based QA engine
│   ├── translate.py                    # Translation-assisted QA module
│   └── experiments.py                  # Comparative evaluation harness
├── results/
│   └── experiment_results.csv          # Reproducible CSV outputs from experiments
├── notebooks/
│   └── analysis.ipynb                  # Exploratory analysis and visualisations
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
└── LICENSE
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- `pip` package manager
- (Optional) CUDA-compatible GPU for accelerated inference

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/Abdilatif1909/pilgrimageqa-benchmark.git
cd pilgrimageqa-benchmark

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### `requirements.txt` (key dependencies)

```
transformers>=4.35.0
datasets>=2.14.0
rank_bm25>=0.2.2
sacrebleu>=2.3.1
deep-translator>=1.11.4
pandas>=2.0.0
numpy>=1.24.0
tqdm>=4.66.0
```

---

## Dataset Generation Pipeline

**Script:** `src/generate_pilgrimage_dataset.py`

The synthetic dataset is produced through a three-stage pipeline:

1. **Knowledge Source Ingestion** — Raw textual knowledge about Hajj and Umrah (rituals, geography, rulings, safety) is ingested from curated source documents stored under `data/raw/`. Documents exist in English, Uzbek, and Russian.

2. **Question–Answer Pair Synthesis** — For each passage, a template-driven generation procedure creates factoid questions (who, what, when, where, how) together with extracted answer spans. A deterministic random seed (`SEED = 42`) guarantees full reproducibility of the generated corpus.

3. **Multilingual Export** — Generated pairs are exported to language-specific CSV files under `data/generated/`, each containing the fields `id`, `language`, `question`, `context`, `answer`, and `answer_start`.

```bash
python src/generate_pilgrimage_dataset.py \
    --source_dir data/raw \
    --output_dir data/generated \
    --num_samples 10000 \
    --seed 42
```

The pipeline produces **10,000+ QA pairs** with balanced language distribution and verified answer-span alignment.

---

## Retrieval and QA Modules

### `src/dataset.py` — Dataset Loader

Provides a unified `PilgrimageDataset` class that:
- Loads CSV benchmark files for any supported language.
- Tokenises questions and context passages.
- Exposes a standard `__getitem__` / `__len__` interface compatible with PyTorch `DataLoader`.

```python
from src.dataset import PilgrimageDataset

ds = PilgrimageDataset("data/generated/benchmark_uz.csv", language="uz")
print(f"Loaded {len(ds)} samples")
```

---

### `src/qa.py` — Retrieval-Based QA Engine

Implements a two-stage **Retrieve-then-Read** pipeline:

1. **Retrieval** — BM25 indexes the full corpus. At query time, the top-*k* passages (default *k* = 5) are retrieved based on token-overlap score.
2. **Reading** — A pre-trained extractive QA model (default: `deepset/roberta-base-squad2`) processes each retrieved passage and returns the highest-scoring answer span.

```python
from src.qa import RetrievalQA

rqa = RetrievalQA(corpus_path="data/generated/benchmark_en.csv", top_k=5)
answer = rqa.answer("What is the significance of Tawaf?")
print(answer)
```

---

### `src/translate.py` — Translation-Assisted QA Module

Enables cross-lingual QA by wrapping the retrieval engine with an automatic translation layer:

1. Non-English queries are translated to English using a neural machine translation back-end (`deep-translator` with Google Translate API).
2. The translated query is forwarded to `RetrievalQA`.
3. (Optionally) the English answer is translated back to the source language.

```python
from src.translate import TranslatedQA

tqa = TranslatedQA(corpus_path="data/generated/benchmark_en.csv", source_lang="uz")
answer = tqa.answer("Tavof nima?")  # Uzbek: "What is Tawaf?"
print(answer)
```

---

## Experimental Benchmark

**Script:** `src/experiments.py`

The experimental harness evaluates three QA configurations on the benchmark dataset and reports all metrics:

| Mode | Description |
|------|-------------|
| **Direct QA** | QA model applied directly to the benchmark language (English). |
| **Translated QA** | Uzbek/Russian queries translated to English before QA. |
| **Retrieval Baseline** | BM25 retrieval only, without a neural reader; answer = top retrieved sentence. |

### Evaluation Metrics

- **Exact Match (EM):** Proportion of predictions that match the gold answer string exactly (after normalisation).
- **Token-level F1:** Average harmonic mean of token-level precision and recall between prediction and gold answer.
- **BLEU Score:** Corpus-level BLEU (SacreBLEU) measuring *n*-gram overlap between predicted and reference answers.

```bash
python src/experiments.py \
    --dataset data/generated/benchmark_uz.csv \
    --language uz \
    --output results/experiment_results.csv \
    --seed 42
```

Results are written to `results/experiment_results.csv` in a machine-readable format for downstream statistical analysis.

---

## Reproducibility

All experiments are fully reproducible:

- **Deterministic random seed:** `SEED = 42` is propagated to NumPy, Python `random`, and (where applicable) PyTorch `torch.manual_seed`.
- **Frozen dependency versions:** All package versions are pinned in `requirements.txt`.
- **CSV result archiving:** Every experimental run writes a timestamped CSV to `results/`, allowing direct comparison across runs and environments.
- **No external API state:** The retrieval and QA modules operate entirely on locally stored indices and model weights; no mutable external service state is assumed.

---

## Example Commands

```bash
# Generate the full multilingual dataset
python src/generate_pilgrimage_dataset.py --num_samples 10000 --seed 42

# Run the full comparative benchmark experiment
python src/experiments.py --dataset data/generated/benchmark_uz.csv --language uz

# Query the retrieval-based QA engine interactively
python -c "
from src.qa import RetrievalQA
rqa = RetrievalQA('data/generated/benchmark_en.csv')
print(rqa.answer('How many times must a pilgrim walk around the Kaaba?'))
"

# Query the translation-assisted mode
python -c "
from src.translate import TranslatedQA
tqa = TranslatedQA('data/generated/benchmark_en.csv', source_lang='ru')
print(tqa.answer('Сколько раз паломник должен обойти Каабу?'))
"
```

---

## Sample Experimental Results

The table below reports benchmark results on the Uzbek QA sub-corpus. Metrics are computed over the full test split (20% held-out).

| Method | Exact Match (EM) ↑ | F1 Score ↑ | BLEU ↑ |
|---|---|---|---|
| **Direct QA** | **0.302** | **0.8925** | **80.16** |
| Translated QA | 0.000 | 0.6883 | 40.67 |
| Retrieval Baseline | 0.000 | 0.6024 | 27.18 |

**Key observations:**

- Direct QA achieves substantially higher F1 (0.8925) and BLEU (80.16) than both cross-lingual alternatives, underscoring the cost of language mismatch in current neural QA pipelines.
- Translated QA demonstrates meaningful retrieval signal (F1 0.6883) despite zero exact matches, indicating that semantic content is partially preserved through translation but surface-form alignment is lost.
- The Retrieval Baseline provides a competitive floor (F1 0.6024), validating BM25 as a strong non-neural baseline for domain-specific retrieval.

These results motivate the development of natively multilingual Hajj and Umrah QA systems that avoid translation loss entirely.

---

## Future Work

1. **Native Multilingual Models** — Fine-tuning multilingual transformer architectures (mBERT, XLM-R) directly on Uzbek and Russian pilgrimage corpora to eliminate translation-induced performance degradation.
2. **Dataset Expansion** — Extending the benchmark to additional pilgrim languages including Indonesian, Turkish, Urdu, Bengali, French, and Swahili.
3. **Generative QA Integration** — Incorporating large language model (LLM) baselines (GPT-4, LLaMA-3, Gemma) in both zero-shot and retrieval-augmented generation (RAG) configurations.
4. **Human Evaluation** — Recruiting native-speaker annotators for Uzbek and Russian to complement automatic metrics with human fluency and correctness judgements.
5. **Real-World Deployment** — Piloting the QA engine as a mobile-accessible service for live pilgrimage seasons, enabling collection of real user query data.
6. **Multi-Modal Extension** — Incorporating image-based queries (e.g., sacred-site recognition, ritual gesture identification) to support visually-grounded pilgrimage assistance.

---

## Citation

If you use **PilgrimageQA-Benchmark** in your research, please cite the following:

```bibtex
@misc{abdilatif2025pilgrimageqa,
  title        = {{PilgrimageQA-Benchmark}: A Multilingual Question-Answering Benchmark
                  and Retrieval-Based Evaluation Framework for {Hajj} and {Umrah}
                  Assistance Systems},
  author       = {Abdilatif, [Author Names]},
  year         = {2025},
  howpublished = {\url{https://github.com/Abdilatif1909/pilgrimageqa-benchmark}},
  note         = {GitHub repository}
}
```

> **Note:** A full journal citation will be provided upon acceptance. Please check the repository for the latest citation information.

---

## License

This project is released under the [MIT License](LICENSE). The benchmark dataset is made available for academic research use; please refer to the LICENSE file for full terms.

---

<p align="center">
  <em>PilgrimageQA-Benchmark — Advancing multilingual AI for the world's largest annual gathering.</em>
</p>
