## Week 1 behavior: google/gemma-2-2b

Prompts scored: 4347  |  data tables: 483
Letter bias (mean logit ' A' - ' B'): -0.61  (cancelled out by averaging over both letter orders)
Format compliance, mc: on average 47.5% of next-token probability is on the two answer tokens
Format compliance, numeric: on average 44.5% of next-token probability is on the two answer tokens

Siding-with-data rate by prior level (conflict = data vs textbook law; congruent = control):
format    condition             Law named  Variable names only       Anonymous x, y
mc        congruent                  54%                  50%                  38% 
mc        conflict                   27%                  23%                  42% 
numeric   congruent                  76%                  70%                  73% 
numeric   conflict                   67%                  68%                  70% 

Go/no-go (suggested thresholds: anonymous accuracy >= 70% on both conditions; prior pull >= 10% with CI above 0):
  mc       anonymous accuracy 38% (congruent) / 42% (conflict); prior pull +15% [95% CI +11%, +20%] over 322 tables -> NO-GO: model can't do the anonymous task
  numeric  anonymous accuracy 73% (congruent) / 70% (conflict); prior pull +3% [95% CI -1%, +7%] over 322 tables -> NO-GO: no clear prior pull
