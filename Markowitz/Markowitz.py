import pandas as pd
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier

prbr = pd.read_excel(r'C:\Users\Jhona\OneDrive\Área de Trabalho\PRBR11.xlsx', index_col='Data', parse_dates=True)
###Expectativa de retornos de mu
mu = expected_returns.mean_historical_return(prbr)

###Matrix de covariancia
sigma = risk_models.sample_cov(prbr)

###Fronteira eficiente
ef = EfficientFrontier(mu, sigma)
et = ef.max_sharpe()
ef.efficient_risk(2.3)
ef.efficient_return(1.5)
print(mu, sigma)

###Retornos
retornos = prbr.pct_change()

###Matriz de covariancia anualizada
covmatriz = retornos.cov()*252
Sigma = risk_models.sample_cov(prbr)
print(covmatriz, Sigma)

weights = ef.efficient_return(0.05)
print(weights)
ef.portfolio_performance(verbose=True)