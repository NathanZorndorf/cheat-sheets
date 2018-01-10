import scipy
import scipy.stats as stats

#from scipy.stats import skew 
#from scipy.stats import kurtosis
#from scipy.stats.stats import pearsonr


#---- Find skew 
skew(list_to_calc_skew)

#---- Find kurtosis : Kurtosis measures whether the distribution is short and flat, or tall and skinny.
kurtosis(list_to_calc_kurt)

#---- Modality : Modality refers to the number of modes, or peaks, in a distribution.

#---- Pearsons coefficient : 
r, p_value = pearsonr(nba_stats["fga"], nba_stats["pts"])


#-------------- T-TESTS -----------------#
import scipy.stats as stats

stats.ttest_ind(a, b)

