# dots.mocr benchmark report

- Final status coverage: 177/177
- Successful OCR outputs: 169/177
- Failed outputs: 8 deterministic-empty pages after repeated identical-setting retries
- Strict CER (169 evaluated pairs): 20.676200
- Normalized Arabic CER: 20.377488
- WER: 38.884466
- Normalized WER: unavailable (not defined by the canonical evaluator)
- Final accuracy rank: 7/7
- Strongest distortion bucket: faded (9 successful pages, normalized CER 0.172643)
- Weakest distortion bucket: jpeg (21 successful pages, normalized CER 74.288842)
- Strongest source dataset: craneset/arabic-ocr (normalized CER 0.003011)
- Weakest source dataset: calfa-ai/RASAM-1 (normalized CER 194.801431)
- Leading-model conclusion: unchanged; HunyuanOCR-1.5 remains first.

The canonical evaluator averages per-page metrics over successful pairs and excludes explicit failures. Eight failed pages therefore make the dots.mocr aggregate coverage-qualified; the common-169-page ranking also places dots.mocr last.
