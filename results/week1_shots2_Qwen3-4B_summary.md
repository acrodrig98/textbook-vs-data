## Week 1 behavior: Qwen/Qwen3-4B

Prompts scored: 2898  |  data tables: 483
Letter bias (mean logit ' A' - ' B'): +0.64  (cancelled out by averaging over both letter orders)
Format compliance, mc: on average 99.5% of next-token probability is on the two answer tokens

Siding-with-data rate by prior level (conflict = data vs textbook law; congruent = control):
format    condition             Law named  Variable names only       Anonymous x, y
mc        congruent                  89%                  83%                  71% 
mc        conflict                   76%                  85%                  93% 

Go/no-go (suggested thresholds: anonymous accuracy >= 70% on both conditions; prior pull >= 10% with CI above 0):
  mc       anonymous accuracy 71% (congruent) / 93% (conflict); prior pull +16% [95% CI +12%, +21%] over 322 tables -> GO
