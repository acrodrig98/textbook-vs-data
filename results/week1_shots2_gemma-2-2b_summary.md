## Week 1 behavior: google/gemma-2-2b

Prompts scored: 2898  |  data tables: 483
Letter bias (mean logit ' A' - ' B'): -0.01  (cancelled out by averaging over both letter orders)
Format compliance, mc: on average 92.9% of next-token probability is on the two answer tokens

Siding-with-data rate by prior level (conflict = data vs textbook law; congruent = control):
format    condition             Law named  Variable names only       Anonymous x, y
mc        congruent                  43%                  39%                  37% 
mc        conflict                   47%                  55%                  65% 

Go/no-go (suggested thresholds: anonymous accuracy >= 70% on both conditions; prior pull >= 10% with CI above 0):
  mc       anonymous accuracy 37% (congruent) / 65% (conflict); prior pull +18% [95% CI +14%, +22%] over 322 tables -> NO-GO: model can't do the anonymous task
