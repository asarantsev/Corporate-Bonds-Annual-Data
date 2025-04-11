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
inflation = data[:, 3]
plt.plot(rate, label = 'BAA Rate')
plt.plot(inflation, label = 'Inflation')
plt.legend()
plt.title('Bond and Inflation Rates, 1972-2024')
plt.savefig('rates.png')
plt.close()
rrate = rate - inflation
plt.plot(rrate)
plt.title('Real Rate, 1972-2024')
plt.savefig('real-rates.png')
plt.close()
volatility = data[:, 2]
returns = numpy.diff(numpy.log(wealth))
dreturns = returns - rate[:-1]
N = len(dreturns)

def plots(data, label):
    plot_acf(data, zero = False)
    plt.title(label + '\n ACF for Original Values')
    plt.savefig('O-' + label + '.png')
    plt.close()
    plot_acf(abs(data), zero = False)
    plt.title(label + '\n ACF for Absolute Values')
    plt.savefig('A-' + label + '.png')
    plt.close()
    qqplot(data, line = 's')
    plt.title(label + '\n Quantile-Quantile Plot vs Normal')
    plt.savefig('QQ-' + label + '.png')
    plt.close()
    
def analysis(data, label):
    print(label + ' analysis of residuals normality')
    print('Skewness:', stats.skew(data))
    print('Kurtosis:', stats.kurtosis(data))
    print('Shapiro-Wilk p = ', stats.shapiro(data)[1])
    print('Jarque-Bera p = ', stats.jarque_bera(data)[1])

print('Simple autoregression for rate')
Reg = stats.linregress(rrate[:-1], numpy.diff(rrate))
print(Reg)
resid = numpy.array([numpy.diff(rrate)[k] - Reg.slope * rrate[k] - Reg.intercept for k in range(N)])
plots(resid, 'rate-simple')
analysis(resid, 'rate-simple')

print('Auto Regression of Rates with VIX')
RegDF = pandas.DataFrame({'Lag' : rrate[:-1]/volatility[1:], 'Volatility': 1, 'Constant' : 1/volatility[1:]})
Reg = sm.OLS(numpy.diff(rrate)/volatility[1:], RegDF).fit()
print(Reg.summary())
resid = Reg.resid
plots(resid, 'rate-vol')
analysis(resid, 'rate-vol')

print('Regression Simple of returns minus rate vs rate change: Duration')
Reg = stats.linregress(numpy.diff(rrate), dreturns)
print(Reg)
resid = numpy.array([dreturns[k] - Reg.slope * numpy.diff(rrate)[k] - Reg.intercept for k in range(N)])
plots(resid, 'return-simple')
analysis(resid, 'return-simple')

print('Regression of returns minus rate with VIX')
RegDF = pandas.DataFrame({'Duration' : numpy.diff(rrate)/volatility[1:], 'Volatility': 1, 'Constant' : 1/volatility[1:]})
Reg = sm.OLS(dreturns/volatility[1:], RegDF).fit()
print(Reg.summary())
resid = Reg.resid
plots(resid, 'return-vol')
analysis(resid, 'return-vol')