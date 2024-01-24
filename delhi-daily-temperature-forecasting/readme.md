<h1 align="center">Time Series: Temperature Forecasting</h1>

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/daily-temperature-forecasting/images/temperature.png?raw=true)

New Delhi experiences large seasonal temperature swings typical of North India’s inland plains climate. Summers are notoriously hot, with temperatures often crossing 40°C (104°F). Monsoon rains bring some relief during July-August before autumn's comfortable period. Come December through February, cold waves drop minimums to freezing levels.

These extremes and variability of New Delhi's weather significantly influence the routines and plans of the city's 18 million inhabitants. The searing summer heat forces many to stay indoors during the daytime, but outdoor activities slowly resumes during the pleasant late autumn and early spring. Also, schools often switch to half-days in peak summer and winter. 

This project aims to build a machine learning model to accurately predict the mean daily temperature in New Delhi based on historical data and relevant weather variables such as humidity, wind speed, and air pressure. The modeled weather outlooks can help people better schedule their routines, while businesses can optimize operations and demand forecasting. Overall, improving temperature forecasting can bring significant economic and social benefits for Delhi and boost preparedness for extreme heat and cold events.

## THE PROJECT

The dataset used here is the [Daily Climate Time Series Data](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data) by SUMANTHVRAO which can be found on Kaggle, it contains the recorded daily humidity, wind speed, mean pressure and mean temperature from 1/1/2013 to 24/4/2017.

Besides some usual libraries such as pandas, numpy, matplotlib, and seaborn, we will also import a few from statsmodels.tsa which is tailored for Time Series Analysis:

<details>
<summary>View Code</summary>

```python
from statsmodels.graphics.tsaplots import plot_pacf as pacf, plot_acf as acf
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import kpss, adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pmdarima.arima import auto_arima, ndiffs, nsdiffs

# Data summary
def summary(dtf):
    sumary=pd.concat([dtf.isna().sum(),((dtf == 0).sum())/dtf.shape[0],dtf.dtypes], axis=1)
    sumary=sumary.rename(columns={sumary.columns[0]: 'NaN'})
    sumary=sumary.rename(columns={sumary.columns[1]: 'Zeros'})
    sumary=sumary.rename(columns={sumary.columns[2]: 'Type'})
    sumary['NaN']=sumary['NaN'].astype(str)+' ('+((sumary['NaN']*100/dtf.shape[0]).astype(int)).astype(str)+'%)'
    sumary['Zeros']=(sumary['Zeros']*100).astype(int)
    sumary['Zeros']=(dtf == 0).sum().astype(str)+' ('+sumary['Zeros'].astype(str)+'%)'
    sumary=sumary[['Type','NaN','Zeros']]
    return print(sumary)

def pair_plot(df):
    def corrdot(*args, **kwargs):
        corr_r = args[0].corr(args[1], 'pearson')
        corr_text = f"{corr_r:2.2f}".replace("0.", ".")
        ax = plt.gca()
        ax.set_axis_off()
        marker_size = abs(corr_r) * 10000
        ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
                   vmin=-1, vmax=1, transform=ax.transAxes)
        font_size = abs(corr_r) * 40 + 5
        ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                    ha='center', va='center', fontsize=font_size)
    numerical_columns = df.select_dtypes(include='number').columns
    g = sns.PairGrid(df[numerical_columns])
    g.map_lower(sns.scatterplot, alpha=0.5)
    g.map_diag(sns.histplot, color='blue')
    g.map_upper(corrdot)
    g.fig.suptitle('Pairplot of Numerical Variables', y=1.02)
    plt.show()

```
</details>

```python
df_train = pd.read_csv("/kaggle/input/daily-climate-time-series-data/DailyDelhiClimateTrain.csv")
df_test  = pd.read_csv("/kaggle/input/daily-climate-time-series-data/DailyDelhiClimateTest.csv")
df_train=df_train[:-1]
```

```python
summary(df_train)
df_train.head()
```

### Output

<pre>
                 Type     NaN    Zeros
date           object  0 (0%)   0 (0%)
meantemp      float64  0 (0%)   0 (0%)
humidity      float64  0 (0%)   0 (0%)
wind_speed    float64  0 (0%)  26 (1%)
meanpressure  float64  0 (0%)   0 (0%)
<table border="0" class="dataframe"> <tbody>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>meantemp</th>
      <th>humidity</th>
      <th>wind_speed</th>
      <th>meanpressure</th>
    </tr>
    <tr>
      <th>0</th>
      <td>2013-01-01</td>
      <td>10.000000</td>
      <td>84.500000</td>
      <td>0.000000</td>
      <td>1015.666667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-01-02</td>
      <td>7.400000</td>
      <td>92.000000</td>
      <td>2.980000</td>
      <td>1017.800000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-01-03</td>
      <td>7.166667</td>
      <td>87.000000</td>
      <td>4.633333</td>
      <td>1018.666667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-01-04</td>
      <td>8.666667</td>
      <td>71.333333</td>
      <td>1.233333</td>
      <td>1017.166667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-01-05</td>
      <td>6.000000</td>
      <td>86.833333</td>
      <td>3.700000</td>
      <td>1016.500000</td>
    </tr>
  </tbody></table></pre>

Everything looks fine, except for wind_speed which, although it is reasonable to have no wind at all, we should probably assign a value to improve our model performance later. Notice that the date is having an 'object' data type which should be converted too.

# Exploratory Data Analysis (EDA)

### Univariate Analysis
We will visualize the distribution of each feature. 
* *meantemp* looks slightly negatively skewed but should be fine.
* *humidity* looks fine, no transformation needed.
* *wind_speed* looks positively skewed but most probably due to a few outliers.
* *meanpressure* just doesn't look right, we have a serious outlier problem on this one!

<details>
<summary>View Code</summary>

```python
fig, axs = plt.subplots(1, 4, figsize=(12, 3))
sns.histplot(data=df_train['meantemp'], ax=axs[0])
sns.histplot(data=df_train['humidity'], ax=axs[1]).set(ylabel=None)
sns.histplot(data=df_train['wind_speed'], ax=axs[2]).set(ylabel=None)
sns.histplot(data=df_train['meanpressure'], ax=axs[3]).set(ylabel=None)
plt.show()
```
</details>

### Output

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/daily-temperature-forecasting/images/1.png?raw=true)

### Bivariate Analysis
This is a good way to observe if there are any interaction between meantemp and other features. 
* It also confirmed our suspicion on the outlier problem for meanpressure especially on the day with meanpressure reaching 8000.
* Humidity seems to have a negative correlation with meantemp
* wind_speed is hard to visualize, also due to some outliers on the right end. Although I suspect it has a positive correlation with meantemp.

<details>
<summary>View Code</summary>

```python
fig, axs = plt.subplots(1, 3, figsize=(10, 3))
sns.scatterplot(x=(df_train['meanpressure']), y=df_train['meantemp'], ax=axs[0])
sns.scatterplot(x=(df_train['humidity']), y=df_train['meantemp'], ax=axs[1]).set(ylabel=None)
sns.scatterplot(x=(df_train['wind_speed']), y=df_train['meantemp'], ax=axs[2]).set(ylabel=None)
plt.show()
```
</details>

### Output

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/daily-temperature-forecasting/images/2.png?raw=true)

# Data Cleansing
From what we saw on *wind_speed* and *meanpressure*, it's time to cleanse these features.

Our assumptions were right, both features are heavily influenced by outliers.

A simple 1% & 99% percentile clipping and cube root transformation will normalize them for good.

### wind_speed
* Since its distribution is right skewed, the median will be a better representation of the whole distribution, so we will **impute zeroes** with the **median**.
* We perform a **cube-root** transformation to normalize it.
* extreme values exceeding the 99th percentile will be capped at this percentile.

<details>
<summary>View Code</summary>

```python
fig, axs = plt.subplots(1, 2, figsize=(10, 2))
sns.boxplot(data=df_train, x='wind_speed', ax=axs[0]).set_title('Before')
df_train['wind_speed'].replace(0,df_train['wind_speed'].median(), inplace=True)
df_train['wind_speed']=np.cbrt(df_train['wind_speed'])
percentiles = df_train['wind_speed'].quantile([0.01,0.99]).values
df_train['wind_speed']=np.clip(df_train['wind_speed'], a_min=None, a_max=percentiles[1])
sns.boxplot(data=df_train, x='wind_speed', ax=axs[1]).set_title('After')
plt.show()
```
</details>

### Output

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/daily-temperature-forecasting/images/3.png?raw=true)

### meanpressure
* Outliers in both extreme ends will be capped to the 1st and 99th percentile.

<details>
<summary>View Code</summary>

```python
fig, axs = plt.subplots(1, 2, figsize=(10, 2))
sns.boxplot(x=df_train['meanpressure'], ax=axs[0]).set_title('Before')
percentiles = df_train['meanpressure'].quantile([0.01, 0.99]).values
df_train['meanpressure']=np.clip(df_train['meanpressure'], percentiles[0], percentiles[1])
sns.boxplot(x=df_train['meanpressure'], ax=axs[1]).set_title('After')
plt.show()
```
</details>

### Output

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/daily-temperature-forecasting/images/4.png?raw=true)

The treated features should give us a clearer presentation of their distribution and pairwise relationship, let's see if this is true:

```python
pair_plot(df_train)
```

### Output

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/daily-temperature-forecasting/images/5.png?raw=true)


## Time Series decomposition of Temperature

<details>
<summary>View Code</summary>

```python
df_train['date'] = pd.to_datetime(df_train['date'])
df_train = df_train.set_index('date').asfreq('W')
df_test['date'] = pd.to_datetime(df_test['date'])
df_test = df_test.set_index('date').asfreq('W')
dec=seasonal_decompose(df_train['meantemp'], model='additive')
fig = dec.plot()
fig.set_size_inches((16, 8))
fig.tight_layout()
plt.show()
```
</details>

### Output

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/daily-temperature-forecasting/images/6.png?raw=true)

Let's look at the different components of *meantemp*.
* Trend - There seems to be an upward trend, global warming is real!
* Seasonal - There's an obvious repeating annual cycle.
* Residuals - Looks random which is good.

This does not seem to be a stationary series, let's check using **AD Fuller** and **KPSS tests**.

## AD Fuller and KPSS tests on stationarity
Keep in mind that in order to prove that our series is stationary, we have to
* **Reject** null hypothesis in **AD Fuller** test
* **Accept** null hypothesis in **KPSS** Test

<details>
<summary>View Code</summary>

```python
adtest=adfuller(df_train['meantemp'][:])
print('ADF Statistic: %f' % adtest[0])
print('p-value: %f' % adtest[1])
print('Critical Values:')
for key, value in adtest[4].items():
 print('\t%s: %.3f' % (key, value))
print("Since p-value < 0.05, we reject the null hypothesis, therefore the series is STATIONARY. \n" if adtest[1] < 0.05 else "Since p-value > 0.05, we fail to reject the null hypothesis, the series is NOT STATIONARY. \n")

kptest=kpss(df_train['meantemp'][:])
print('KPSS Statistic: %f' % kptest[0])
print('p-value: %f' % kptest[1])
print('Critical Values:')
for key, value in kptest[3].items():
 print('\t%s: %.3f' % (key, value))
print("Since p-value < 0.05, we reject the null hypothesis and the series is NOT STATIONARY. \n" if kptest[1] < 0.05 else "Since p-value > 0.05, we fail to reject the null hypothesis, therefore the series is STATIONARY. \n")
```
</details>

### Output
<pre>
ADF Statistic: -4.428957
p-value: 0.000264
Critical Values:
	1%: -3.463
	5%: -2.876
	10%: -2.574
Since p-value < 0.05, we reject the null hypothesis, therefore the series is STATIONARY. 

KPSS Statistic: 0.076783
p-value: 0.100000
Critical Values:
	10%: 0.347
	5%: 0.463
	2.5%: 0.574
	1%: 0.739
Since p-value > 0.05, we fail to reject the null hypothesis, therefore the series is STATIONARY. 
</pre>

Contrary to our belief, it passed both stationarity tests even without differencing. So we may proceed to the next step with out stationary series.

## ACF and PACF visualization
It's time to tackle the MA and AR components *p* & *q* with the help of our ACF and PACF plots.

```python
fig, ax = plt.subplots(1,2,figsize=(10,5))
acf(df_train['meantemp'][:], ax=ax[0])
pacf(df_train['meantemp'][:], ax=ax[1])
plt.show()
```
### Output

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/daily-temperature-forecasting/images/7.png?raw=true)

Based on the ACF and PACF plots
* m=52 since it has an annual cycle equivalent to 52 weeks
* ACF : Significant at lag 7 or 8 (*q*=7) 
* PACF: Significant at lag 2 (*p*=2)

However the ACF plot flats out too slowly, this suggests that differencing is still required, therefore let's try out on the differenced series too:

# BONUS: Differencing of series
Although the original series is already stationary, we will see if the 1st order of differencing makes it better.

```python
df_train['meantemp1']=df_train['meantemp'].diff(periods=1)
dec=seasonal_decompose(df_train['meantemp1'][1:], model='additive')
fig = dec.plot()
fig.set_size_inches((16, 8))
fig.tight_layout()
plt.show()
```

### Output

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/daily-temperature-forecasting/images/8.png?raw=true)

Now there is definitely no trend, no further differencing required.

We run both stationarity tests again on the differenced series, just to be sure:

<details>
<summary>View Code</summary>

```python
adtest=adfuller(df_train['meantemp1'][1:])
print('ADF Statistic: %f' % adtest[0])
print('p-value: %f' % adtest[1])
print('Critical Values:')
for key, value in adtest[4].items():
  print('\t%s: %.3f' % (key, value))
print("Since p-value < 0.05, we reject the null hypothesis, therefore the series is STATIONARY. \n" if adtest[1] < 0.05 else "Since p-value > 0.05, we fail to reject the null hypothesis, the series is NOT STATIONARY \n")

kptest=kpss(df_train['meantemp1'][1:])
print('KPSS Statistic: %f' % kptest[0])
print('p-value: %f' % kptest[1])
print('Critical Values:')
for key, value in kptest[3].items():
  print('\t%s: %.3f' % (key, value))
print("Since p-value < 0.05, we reject the null hypothesis and the series is NOT STATIONARY \n" if kptest[1] < 0.05 else "Since p-value > 0.05, we fail to reject the null hypothesis, therefore the series is STATIONARY. \n")
```
</details>

### Output

<pre>
ADF Statistic: -3.366485
p-value: 0.012156
Critical Values:
	1%: -3.463
	5%: -2.876
	10%: -2.574
Since p-value < 0.05, we reject the null hypothesis, therefore the series is STATIONARY. 

KPSS Statistic: 0.246443
p-value: 0.100000
Critical Values:
	10%: 0.347
	5%: 0.463
	2.5%: 0.574
	1%: 0.739
Since p-value > 0.05, we fail to reject the null hypothesis, therefore the series is STATIONARY.
</pre>

### ACF and PACF of Differenced Series (d=1)

```python
fig, ax = plt.subplots(1,2,figsize=(10,5))
acf(df_train['meantemp1'][1:], ax=ax[0])
pacf(df_train['meantemp1'][1:], ax=ax[1])
plt.show()
```
### Output

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/daily-temperature-forecasting/images/9.png?raw=true)

For the differenced series d=1, both ACF and PACF are significant after lag 1, this seems like a better model to be fitted so **we will use this model instead**.

# Model Training & Evaluation

## 1. SARIMA Model
A SARIMA model is necessary in our case due to the obvious annual seasonality. Let's build the SARIMA model using values of p,d,q determined earlier.

<details>
<summary>View Code</summary>

```python
from pmdarima.arima import auto_arima, ndiffs, nsdiffs
sarima_model = auto_arima(df_train['meantemp'],
                          exog=df_train.drop(['meantemp','meantemp1','meanpressure'],axis=1),
                          start_p=1, 
                          start_q=1,
                          test='adf',
                          max_p=3, 
                          max_q=3,
                          m=52,             
                          d=1,          
                          seasonal=True,
                          start_P=3,
                          D=1,
                          start_Q=0,
                          max_P=3,
                          max_D=1,
                          max_Q=3,
                          trace=True,
                          error_action='ignore',  
                          stepwise=True
                         )
```
</details>

### Output
<pre>
Performing stepwise search to minimize aic
 ARIMA(1,1,1)(3,1,0)[52]             : AIC=-305.460, Time=154.55 sec
 ARIMA(0,1,0)(0,1,0)[52]             : AIC=-187.151, Time=0.69 sec
 ARIMA(1,1,0)(1,1,0)[52]             : AIC=-242.633, Time=6.82 sec
 ARIMA(0,1,1)(0,1,1)[52]             : AIC=inf, Time=13.10 sec
 ARIMA(1,1,1)(2,1,0)[52]             : AIC=-300.444, Time=70.98 sec
 ARIMA(1,1,1)(3,1,1)[52]             : AIC=inf, Time=181.42 sec
 ARIMA(1,1,1)(2,1,1)[52]             : AIC=inf, Time=77.43 sec
 ARIMA(0,1,1)(3,1,0)[52]             : AIC=-305.859, Time=134.67 sec
 ARIMA(0,1,1)(2,1,0)[52]             : AIC=-300.037, Time=34.54 sec
 ARIMA(0,1,1)(3,1,1)[52]             : AIC=inf, Time=190.69 sec
 ARIMA(0,1,1)(2,1,1)[52]             : AIC=inf, Time=40.39 sec
 ARIMA(0,1,0)(3,1,0)[52]             : AIC=inf, Time=39.32 sec
 ARIMA(0,1,2)(3,1,0)[52]             : AIC=-305.406, Time=173.38 sec
 ARIMA(1,1,0)(3,1,0)[52]             : AIC=inf, Time=142.42 sec
 ARIMA(1,1,2)(3,1,0)[52]             : AIC=-304.011, Time=196.35 sec
 ARIMA(0,1,1)(3,1,0)[52] intercept   : AIC=inf, Time=175.88 sec

Best model:  ARIMA(0,1,1)(3,1,0)[52]          
Total fit time: 1633.227 seconds
</pre>

```python
sarima_model=SARIMAX(df_train['meantemp'], 
                     order=(1,1,1), 
                     seasonal_order=(2,1,0,52), 
                     exog=df_train.drop(['meantemp','meantemp1'], axis=1)
                    ) #1,1,2,2,0,0
model=sarima_model.fit()
print(model.summary())
```

### Output
<pre>
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =            8     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f= -8.93544D-01    |proj g|=  3.46494D+00
 This problem is unconstrained.

At iterate    5    f= -9.30562D-01    |proj g|=  4.89032D-01

At iterate   10    f= -9.41992D-01    |proj g|=  3.91968D-01

At iterate   15    f= -9.42336D-01    |proj g|=  1.90493D-01

At iterate   20    f= -9.42909D-01    |proj g|=  8.95016D-02

At iterate   25    f= -9.43517D-01    |proj g|=  6.76433D-02

At iterate   30    f= -9.43840D-01    |proj g|=  1.00839D-01

At iterate   35    f= -9.44196D-01    |proj g|=  8.48446D-02

At iterate   40    f= -9.44596D-01    |proj g|=  1.73890D-01

At iterate   45    f= -9.46567D-01    |proj g|=  4.31194D-01

At iterate   50    f= -9.47366D-01    |proj g|=  5.43491D-02

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
    8     50     60      1     0     0   5.435D-02  -9.474D-01
  F = -0.94736621185899061     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
                                      SARIMAX Results                                      
===========================================================================================
Dep. Variable:                            meantemp   No. Observations:                  209
Model:             SARIMAX(1, 1, 1)x(2, 1, [], 52)   Log Likelihood                 198.000
Date:                             Wed, 17 Jan 2024   AIC                           -379.999
Time:                                     08:47:52   BIC                           -355.600
Sample:                                          0   HQIC                          -370.089
                                             - 209                                         
Covariance Type:                               opg                                         
================================================================================
                   coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
humidity        -0.2951      0.032     -9.283      0.000      -0.357      -0.233
wind_speed      -0.0236      0.021     -1.100      0.271      -0.066       0.018
meanpressure    -0.1596      0.049     -3.253      0.001      -0.256      -0.063
ar.L1            0.0665      0.094      0.711      0.477      -0.117       0.250
ma.L1           -0.9388      0.035    -26.896      0.000      -1.007      -0.870
ar.S.L52        -0.6527      0.095     -6.888      0.000      -0.838      -0.467
ar.S.L104       -0.4368      0.123     -3.559      0.000      -0.677      -0.196
sigma2           0.0037      0.001      7.167      0.000       0.003       0.005
===================================================================================
Ljung-Box (L1) (Q):                   0.15   Jarque-Bera (JB):                 4.99
Prob(Q):                              0.70   Prob(JB):                         0.08
Heteroskedasticity (H):               0.74   Skew:                            -0.21
Prob(H) (two-sided):                  0.28   Kurtosis:                         3.77
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
</pre>

After thorough trial-and-error, this **SARIMA(0,1,1),(3,1,0,52)** model works best with an AIC of -380. 

We should check on the residuals too:

<details>
<summary>View Code</summary>

```python
residuals = pd.DataFrame(model.resid)
fig, ax = plt.subplots(1,2,figsize=(10, 4))
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()
```
</details>

### Output

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/daily-temperature-forecasting/images/10.png?raw=true)

The residuals are normally distributed, another tick on our checklist!

Let's visualize the whole model and our training data:

```python
train=model.predict()
plt.figure(figsize=(12, 6))
plt.plot(df_train['meantemp'], label='Actual')
plt.plot(train, color='r', label='SARIMA')
plt.legend()
plt.title('SARIMA(1,1,1)(2,1,0,52) Model Predictions vs. Actual meantemp')
plt.xlabel('Date')
plt.ylabel('meantemp')
plt.show()
```

### Output

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/daily-temperature-forecasting/images/11.png?raw=true)

This looks like a fairly good fit, but we are more interested in it's prediction accuracy on unseen data. Let's bring out our test data and put this model to the ultimate test! (no pun intended)

## Forecasting using SARIMA
We will now evaluate our model based on the unseen test data, but first the test data has to be converted from Days to Weeks too.
