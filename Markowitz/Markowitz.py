import pandas as pd
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier

prbr = pd.read_excel(r'C:\Users\Jhona\OneDrive\Área de Trabalho\PRBR11.xlsx', index_col='Data')
prbr.sort_index()
###Expectativa de retornos de mu
mu = expected_returns.mean_historical_return(prbr)

###Matrix de covariancia]
sigma = risk_models.sample_cov(prbr)

###Retornos
retornos = prbr.pct_change()
