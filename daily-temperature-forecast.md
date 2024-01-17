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
    
<table border="1" class="dataframe">
  <tbody>
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
  </tbody>
</table>
</pre>

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

<img src=>

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

<img src=>

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

<img src=>

The treated features should give us a clearer presentation of their distribution and pairwise relationship, let's see if this is true:

```python
pair_plot(df_train)
```

### Output

<img src=>


## Time Series decomposition of Temperature

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

### Output

<img src=>

Let's look at the different components of *meantemp*.
* Trend - There seems to be an upward trend, global warming is real!
* Seasonal - There's an obvious repeating annual cycle.
* Residuals - Looks random which is good.

This does not seem to be a stationary series, let's check using **AD Fuller** and **KPSS tests**.
