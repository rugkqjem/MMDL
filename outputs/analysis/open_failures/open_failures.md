# Open-ended failures, Qwen parser (38 of 53 open questions wrong)

| Category | N |
|---|---|
| latex_only | 3 |
| format_only | 5 |
| rounding_near | 2 |
| wrong_value | 20 |
| no_marker | 4 |
| no_final_answer | 4 |

Notation-only failures (latex_only + format_only): **8** -> at most **+0.89** macro points if all were judged correct.

## latex_only (3)

| id | gold | extracted | recoverable by |
|---|---|---|---|
| validation_Biology_10 | `1/64` | `\dfrac{1}{64}` | delatex |
| validation_Electronics_2 | `2.83` | `2\sqrt{2}` | delatex+round |
| validation_Math_15 | `['24/7', '3.429']` | `\frac{24}{7}` | delatex+unwrap_list_gold |

## format_only (5)

| id | gold | extracted | recoverable by |
|---|---|---|---|
| validation_Basic_Medical_Science_10 | `C` | `c` | - |
| validation_Finance_17 | `1000` | `1,000` | delatex+round |
| validation_Finance_20 | `7243000` | `7,243,000` | delatex+round |
| validation_Finance_30 | `1249` | `1,249.24` | delatex+round |
| validation_Manage_20 | `2960` | `2,960 unfavorable` | delatex+round |

## rounding_near (2)

| id | gold | extracted | recoverable by |
|---|---|---|---|
| validation_Electronics_29 | `30` | `28.8` | - |
| validation_Manage_3 | `242110.62` | `242,159.73` | - |

## wrong_value (20)

| id | gold | extracted | recoverable by |
|---|---|---|---|
| validation_Chemistry_13 | `12.97` | `13.0 g of calcium hydroxide was dissolved.` | - |
| validation_Chemistry_20 | `5` | `3` | - |
| validation_Chemistry_30 | `['$MgS$', 'MgS']` | `MgO` | - |
| validation_Chemistry_4 | `trans-1-Chloro-4-methylcyclohexane` | `cis-1-chloro-2-methylcyclohexane` | - |
| validation_Electronics_16 | `20` | `\frac{100}{3}` | - |
| validation_Electronics_21 | `0.9965` | `0` | - |
| validation_Electronics_24 | `551` | `580\ k\Omega` | - |
| validation_Electronics_25 | `50` | `Given the ambiguity in terminology, and based on standard interpretation in r...` | - |
| validation_Electronics_26 | `6.333` | `11` | - |
| validation_Electronics_27 | `-120` | `240` | - |
| validation_Electronics_30 | `-141` | `45.84` | - |
| validation_Electronics_4 | `8.4` | `V_{CEQ} \approx 7.0 \, V` | - |
| validation_Electronics_5 | `62.6` | `50 \, \Omega` | - |
| validation_Finance_10 | `2000000` | `1,646,100` | - |
| validation_Finance_23 | `527.89` | `527.74 million (rounded to two decimal places)` | - |
| validation_Finance_24 | `30.0` | `20` | - |
| validation_Finance_8 | `360` | `860` | - |
| validation_Geography_14 | `8200` | `2240` | - |
| validation_Pharmacy_26 | `A` | `B` | - |
| validation_Physics_21 | `A` | `D` | - |

## no_marker (4)

| id | gold | extracted | recoverable by |
|---|---|---|---|
| validation_Chemistry_23 | `6.5` | `0.0246` | - |
| validation_Chemistry_25 | `24.32` | `24.3` | - |
| validation_Computer_Science_26 | `251` | `156` | - |
| validation_Geography_4 | `['Tampa', 'Florida']` | `The place described in the image is a map of Clearwater, Florida. It highligh...` | delatex+unwrap_list_gold |

## no_final_answer (4)

| id | gold | extracted | recoverable by |
|---|---|---|---|
| validation_Architecture_and_Engineering_14 | `1.06` | `To compute the **initial deflection at midspan** of the beam using a **finite...` | - |
| validation_Electronics_11 | `0.3` | `to write the nodal equation for node n3, and then solve for v_x.` | - |
| validation_Electronics_6 | `71.6` | `72 V, but let's box 71.7.` | - |
| validation_Pharmacy_19 | `B` | `Step D.` | - |

