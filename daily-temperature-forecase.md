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
from statsmodels.graphics.tsaplots import plot_pacf as pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot as acf
from statsmodels.tsa.stattools import kpss, adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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
```
</details>
