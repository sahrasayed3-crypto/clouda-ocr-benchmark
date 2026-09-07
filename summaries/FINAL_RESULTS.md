# Verified OCR Benchmark Results

The canonical benchmark contains 177 distorted samples. Seven models have scored stored outputs. Rankings use mean normalized Arabic CER over each model's valid evaluated pairs; no OCR inference was rerun.

| Rank | Model | Statuses | Valid pairs | Failures | Normalized Arabic CER | WER | CER |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | HunyuanOCR-1.5 | 177 | 177 | 0 | 0.391497102343 | 0.719510772462 | 0.564666303999 |
| 2 | MBZUAI/AIN-7B | 177 | 177 | 0 | 0.837028332500 | 0.643977368106 | 0.752904696351 |
| 3 | Qari OCR 0.4.0 | 177 | 177 | 0 | 1.076259820088 | 1.638313456431 | 1.419240556704 |
| 4 | Qwen3-VL-4B-Instruct | 177 | 177 | 0 | 1.286567026274 | 1.529543234329 | 2.007940166793 |
| 5 | DeepSeek-OCR-2 | 177 | 177 | 0 | 1.486473469125 | 1.549825480979 | 1.727660839069 |
| 6 | Arabic Nougat Large | 177 | 177 | 0 | 2.226340721331 | 1.876030865759 | 2.391472153897 |
| 7 | dots.mocr | 177 | 169 | 8 | 20.377487907976 | 38.884466129364 | 20.676199710648 |

## Unranked runs

| Model | Status | Status rows | Valid pairs | Failures |
|---|---|---:|---:|---:|
| PaddleOCR-VL-1.6 | FAILED | 1 | 0 | 0 |

## Common successful subset

The seven scored models share 169 successful sample IDs. HunyuanOCR-1.5 remains first on this subset.

| Rank | Model | Samples | Normalized Arabic CER | WER | CER |
|---:|---|---:|---:|---:|---:|
| 1 | HunyuanOCR-1.5 | 169 | 0.372044885032 | 0.706726272539 | 0.547388622015 |
| 2 | MBZUAI/AIN-7B | 169 | 0.843021836060 | 0.630505459918 | 0.749066183178 |
| 3 | Qwen3-VL-4B-Instruct | 169 | 1.026305411235 | 1.473953846717 | 1.691950485542 |
| 4 | Qari OCR 0.4.0 | 169 | 1.086816720886 | 1.667768700353 | 1.441300871514 |
| 5 | Arabic Nougat Large | 169 | 1.201064817859 | 1.418598683580 | 1.294888245771 |
| 6 | DeepSeek-OCR-2 | 169 | 1.414038617023 | 1.507213669428 | 1.650864047871 |
| 7 | dots.mocr | 169 | 20.377487907976 | 38.884466129364 | 20.676199710648 |
