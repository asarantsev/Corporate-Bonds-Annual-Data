import pandas
import numpy
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats
from statsmodels.graphics.gofplots import qqplot

DF = pandas.read_excel('aaa.xlsx', sheet_name = 'Annual')
data = DF.values[:, 1:].astype(float)
wealth = data[:, 0]
rate = data[:, 1]/100
volatility = data[:, 2]
returns = numpy.diff(numpy.log(wealth))
dreturns = returns - rate[:-1]
qqplot(dreturns, line = 's')
plt.show()
plot_acf(dreturns)
plt.show()
plot_acf(abs(dreturns))
plt.show()
Reg = stats.linregress(numpy.diff(rate), dreturns)
resid = numpy.array([dreturns[k] - Reg.slope * numpy.diff(rate)[k] - Reg.intercept for k in range(len(dreturns))])
qqplot(resid, line = 's')
plt.show()
plot_acf(resid)
plt.show()
plot_acf(abs(resid))
plt.show()
nresid = resid/volatility[1:]
qqplot(nresid, line = 's')
plt.show()
plot_acf(nresid)
plt.show()
plot_acf(abs(nresid))
plt.show()
RegDF = pandas.DataFrame({'Duration' : numpy.diff(rate)/volatility[1:], 'Volatility': 1, 'Constant' : 1/volatility[1:]})
Reg = sm.OLS(dreturns/volatility[1:], RegDF).fit()
print(Reg.summary())

