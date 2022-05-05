import pandas as pd
import numpy as np
import train.edhec_risk_kit as erk
import matplotlib.pyplot as plt
e = erk.get_ind_returns()
# print(e)
ind = erk.get_ind_returns()
er = erk.annualize_rets(ind['1996': "2000"], 12)


#Random walk Generation

def gbm(n_years=10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0):
    """
    Evolution of stock Price using a Geometric Brownian Motion Model
    """
    dt = 1/steps_per_year
    n_steps = int(n_years * steps_per_year)
    rets_plus_1 = np.random.normal(loc=(1+ mu*dt), scale=(sigma * np.sqrt(dt)), size=(n_steps, n_scenarios))

    prices = s_0 * pd.DataFrame(rets_plus_1).cumprod()

    return prices

p = gbm()
print(p)
p.plot(legend=False)
plt.show()