import pandas as pd
import numpy as np
import train.edhec_risk_kit as erk

e = erk.get_ind_returns()
print(e)
ind = erk.get_ind_returns()
er = erk.annualize_rets(ind['1996': "2000"], 12)


#Random walk Generation

def gbm(n_years=10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0):
    """
    Evolution of stock Price using a Geometric Brownian Motion Model
    """
    dt = 1/steps_per_year
    n_steps = int(n_years ** steps_per_year)
    xi = np.random.normal(size=(n_steps, n_scenarios))
    rets = mu * dt + sigma*np.sqrt(dt) * xi
    rets = pd.DataFrame(rets)
    prices = s_0 * (1 + rets).cumprod()
    return prices

p = gbm()
print(p)