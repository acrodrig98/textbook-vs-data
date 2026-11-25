## Week 1 behavior: Qwen/Qwen3-4B

Prompts scored: 5313  |  data tables: 483
Letter bias (mean logit ' A' - ' B'): +0.26  (cancelled out by averaging over both letter orders)
Format compliance, mc: on average 89.2% of next-token probability is on the two answer tokens
Format compliance, numeric: on average 55.6% of next-token probability is on the two answer tokens

Siding-with-data rate by prior level (conflict = data vs textbook law; congruent = control):
format    condition             Law named    Made-up law named  Variable names only       Anonymous x, y
mc        congruent                  94%                  84%                  88%                  84% 
mc        conflict                   43%                  60%                  74%                  82% 
numeric   congruent                  93%                    -                  89%                  83% 
numeric   conflict                   72%                    -                  75%                  79% 

Go/no-go (suggested thresholds: anonymous accuracy >= 70% on both conditions; prior pull >= 10% with CI above 0):
  mc       anonymous accuracy 84% (congruent) / 82% (conflict); prior pull +39% [95% CI +34%, +45%] over 322 tables -> GO
  numeric  anonymous accuracy 83% (congruent) / 79% (conflict); prior pull +7% [95% CI +3%, +11%] over 322 tables -> NO-GO: no clear prior pull

Where the textbook pull comes from (conflict tables; change in siding-with-data rate, 95% CI):
  mc       real law name -> made-up name   (recall of the specific law): +17% [+13%, +21%]  (322 tables)
  mc       made-up name -> variable names (any named law + setting): +13% [+9%, +18%]  (322 tables)
  mc       variable names -> anonymous     (physical variable names): +9% [+5%, +13%]  (322 tables)
  mc       total: real law name -> anonymous: +39% [+34%, +45%]  (322 tables)
  numeric  variable names -> anonymous     (physical variable names): +4% [+0%, +7%]  (322 tables)
  numeric  total: real law name -> anonymous: +7% [+3%, +11%]  (322 tables)
