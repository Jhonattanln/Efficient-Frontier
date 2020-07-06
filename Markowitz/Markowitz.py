import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.black_litterman import BlackLittermanModel

prbr = pd.read_excel(r'C:\Users\Jhona\OneDrive\Área de Trabalho\Atualizado.xlsx', index_col='Data')

###Expectativa de retornos de mu
mu = expected_returns.mean_historical_return(prbr)

###Matrix de covariancia
sigma = risk_models.sample_cov(prbr)

###Fronteira eficiente Max sharpe
ef = EfficientFrontier(mu, sigma)
weights = ef.max_sharpe()
ef.efficient_risk(2.0)
ef.efficient_return(1.5)
cleaned_weights = ef.clean_weights()
print(weights, cleaned_weights)
ef.portfolio_performance(verbose=True, risk_free_rate = 0.0225)


###Fronteira eficiente Min Vol
ef = EfficientFrontier(mu, sigma)
raw_weights = ef.min_volatility()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose=True, risk_free_rate = 0.0225)

