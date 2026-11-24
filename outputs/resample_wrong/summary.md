# Re-solving baseline wrong answers (8 samples each)

baseline: `outputs/qwen3vl4b_mmmu_val/predictions.jsonl` · model: `Qwen/Qwen3-VL-4B-Instruct` · max_new_tokens 16384

| subject | wrong at baseline | never solved (0/k) | recovered (>=1/k) | mean pass rate | correct-count histogram | truncated samples |
|---|---|---|---|---|---|---|
| Agriculture | 14 | 11 (79%) | 3 (21%) | 5.4% | 0:11 1:2 4:1 | 0 |
| Geography | 17 | 7 (41%) | 10 (59%) | 27.2% | 0:7 1:2 2:2 3:1 4:1 5:1 6:2 7:1 | 7 |
| Sociology | 10 | 8 (80%) | 2 (20%) | 13.8% | 0:8 5:1 6:1 | 0 |
| Chemistry | 20 | 9 (45%) | 11 (55%) | 23.8% | 0:9 1:6 5:1 6:2 7:1 8:1 | 20 |
| ALL | 61 | 35 (57%) | 26 (43%) | 18.9% | 0:35 1:10 2:2 3:1 4:2 5:3 6:5 7:2 8:1 | 27 |

- never solved high -> knowledge gap: augment with external data.
- recovered high -> unstable reasoning: self-generated data / DPO likely more effective.
- histogram `c:n` = n questions answered correctly in c of the k samples.
