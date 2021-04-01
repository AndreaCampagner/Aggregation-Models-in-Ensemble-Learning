import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman

metric = '.Results/complete_f1weighted.csv'

metrics = pd.read_csv(metric, index_col="Dataset")

for a in metrics.columns:
  print("%s : %.3f [%.3f-%.3f]" % (a, metrics[a].mean(), metrics[a].mean() - metrics[a].std()*1.96/np.sqrt(32),
                        metrics[a].mean() + metrics[a].std()*1.96/np.sqrt(32)))
						


print(friedmanchisquare(*[metrics.loc[:,c] for c in metrics.columns]))
posthoc_nemenyi_friedman(metrics).to_csv('results_posthocs.csv')

ranks=metrics.rank(axis=1, ascending=False)
ranks.to_csv("ranks.csv")


for a in ranks.columns:
  print("%s : %.3f [%.3f-%.3f]" % (a, ranks[a].mean(), ranks[a].mean() - ranks[a].std()*1.96/np.sqrt(32),
                        ranks[a].mean() + ranks[a].std()*1.96/np.sqrt(32)))