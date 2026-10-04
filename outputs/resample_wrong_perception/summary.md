# Re-solving baseline wrong answers (8 samples each)

baseline: `outputs/qwen3vl4b_mmmu_val/predictions.jsonl` · model: `Qwen/Qwen3-VL-4B-Instruct` · max_new_tokens 16384

| subject | wrong at baseline | never solved (0/k) | recovered (>=1/k) | mean pass rate | correct-count histogram | truncated samples |
|---|---|---|---|---|---|---|
| Diagnostics_and_Laboratory_Medicine | 21 | 15 (71%) | 6 (29%) | 8.9% | 0:15 1:3 2:1 4:1 6:1 | 2 |
| Music | 21 | 9 (43%) | 12 (57%) | 13.1% | 0:9 1:7 2:3 3:1 6:1 | 78 |
| History | 9 | 4 (44%) | 5 (56%) | 11.1% | 0:4 1:3 2:1 3:1 | 5 |
| ALL | 51 | 28 (55%) | 23 (45%) | 11.0% | 0:28 1:13 2:5 3:2 4:1 6:2 | 85 |

- never solved high -> knowledge gap: augment with external data.
- recovered high -> unstable reasoning: self-generated data / DPO likely more effective.
- histogram `c:n` = n questions answered correctly in c of the k samples.
