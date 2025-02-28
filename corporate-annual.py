import pandas
import numpy
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot

DF = pandas.read_excel('corporate.xlsx', sheet_name = 'Annual')
data = DF.values[:, 1:].astype(float)
wealth = data[:, 0]
rate = data[:, 1]/100
volatility = data[:, 2]
returns = numpy.diff(numpy.log(wealth))
dreturns = returns - rate[:-1]
N = len(dreturns)

print('Simple autoregression for rate')
Reg = stats.linregress(rate[:-1], numpy.diff(rate))
print(Reg)
resid = numpy.array([numpy.diff(rate)[k] - Reg.slope * rate[k] - Reg.intercept for k in range(N)])
qqplot(resid, line = 's')
plt.title('QQ Plot Residuals of Simple AR(1)')
plt.show()
plot_acf(resid)
plt.title('ACF Plot Residuals of Simple AR(1)')
plt.show()
plot_acf(abs(resid))
plt.title('ACF Plot |Residuals| of Simple AR(1)')
plt.show()

print('Auto Regression of Rates with VIX')
RegDF = pandas.DataFrame({'Lag' : rate[:-1]/volatility[1:], 'Volatility': 1, 'Constant' : 1/volatility[1:]})
Reg = sm.OLS(numpy.diff(rate)/volatility[1:], RegDF).fit()
print(Reg.summary())
residuals = Reg.resid
qqplot(residuals, line = 's')
plt.title('QQ Plot Residuals of AR with Volatility')
plt.show()
plot_acf(residuals)
plt.title('ACF Plot Residuals of AR with Volatility')
plt.show()
plot_acf(abs(residuals))
plt.title('ACF Plot |Residuals| of AR with Volatility')
plt.show()
print('Shapiro-Wilk p = ', stats.shapiro(residuals))

print('Returns minus rate')
print('Mean, stdev = ', numpy.mean(dreturns), numpy.std(dreturns))
qqplot(dreturns, line = 's')
plt.title('QQ Plot Returns Minus Rate')
plt.show()
plot_acf(dreturns)
plt.title('ACF Plot Returns Minus Rate')
plt.show()
plot_acf(abs(dreturns))
plt.title('ACF Plot |Returns Minus Rate|')
plt.show()

print('Regression Simple of Returns minus rate vs rate change: Duration')
Reg = stats.linregress(numpy.diff(rate), dreturns)
print(Reg)
resid = numpy.array([dreturns[k] - Reg.slope * numpy.diff(rate)[k] - Reg.intercept for k in range(N)])
qqplot(resid, line = 's')
plt.title('QQ Plot Residuals of Simple Regression')
plt.show()
plot_acf(resid)
plt.title('ACF Plot Residuals of Simple Regression')
plt.show()
plot_acf(abs(resid))
plt.title('ACF Plot |Residuals| of Simple Regression')
plt.show()

print('Regression of Returns minus rate with VIX')
RegDF = pandas.DataFrame({'Duration' : numpy.diff(rate)/volatility[1:], 'Volatility': 1, 'Constant' : 1/volatility[1:]})
Reg = sm.OLS(dreturns/volatility[1:], RegDF).fit()
print(Reg.summary())
residuals = Reg.resid
qqplot(residuals, line = 's')
plt.title('QQ Plot Residuals of Regression with Volatility')
plt.show()
plot_acf(residuals)
plt.title('ACF Plot Residuals of Regression with Volatility')
plt.show()
plot_acf(abs(residuals))
plt.title('ACF Plot |Residuals| of Regression with Volatility')
plt.show()
print('Shapiro-Wilk p = ', stats.shapiro(residuals))