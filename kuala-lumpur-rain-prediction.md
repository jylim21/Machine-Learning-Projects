<h1 align="center">Rain Prediction in Kuala Lumpur</h1>

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/kuala-lumpur-rain-prediction/images/kl-rain.jpg?raw=true)

## THE PROJECT

The dataset used here can be downloaded from [visualcrossing.com](https://www.visualcrossing.com), the free version allows users to export daily data on temperature, humidity, wind, solar, and precipitation etc. for different location with a daily limit of 1000 rows. Since this dataset contains data from July 2016 to Dec 2023, it took me 3 days to get the data required.

The main objective of this project is to be able to predict whether it will rain on the next day given the weather details of previous days, it would be useless to predict based on same day conditions as same day data could only be obtained at the end of that day. 

To begin, we import the usual pandas, numpy, matplotlib, and seaborn library:

<details>
<summary>View Code</summary>

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)

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

# First Glance
Our rule of thumb is to have an overview at the dataset and the attributes within it, this would give us a quick idea on what could possibly contribute to our analysis, and whether they are valid enough to be included in our model later on.

```python
df=pd.read_csv('/kaggle/input/kuala-lumpur-weather-1st-jul-2016-7th-jan-2024/kuala lumpur Rain Jul16 - Jan24.csv')
summary(df)
df.head()
```

### Output

<pre>
                     Type         NaN      Zeros
name               object      0 (0%)     0 (0%)
datetime           object      0 (0%)     0 (0%)
tempmax           float64      0 (0%)     0 (0%)
tempmin           float64      0 (0%)     0 (0%)
temp              float64      0 (0%)     0 (0%)
feelslikemax      float64      0 (0%)     0 (0%)
feelslikemin      float64      0 (0%)     0 (0%)
feelslike         float64      0 (0%)     0 (0%)
dew               float64      0 (0%)     0 (0%)
humidity          float64      0 (0%)     0 (0%)
precip            float64      0 (0%)  903 (32%)
precipprob          int64      0 (0%)  903 (32%)
precipcover       float64      0 (0%)  903 (32%)
preciptype         object   720 (26%)     0 (0%)
snow              float64  2019 (73%)  728 (26%)
snowdepth         float64  2019 (73%)  728 (26%)
windgust          float64  1837 (66%)     0 (0%)
windspeed         float64      0 (0%)     0 (0%)
winddir           float64      0 (0%)     0 (0%)
sealevelpressure  float64      1 (0%)     0 (0%)
cloudcover        float64      0 (0%)     0 (0%)
visibility        float64      0 (0%)     0 (0%)
solarradiation    float64      0 (0%)     0 (0%)
solarenergy       float64      0 (0%)     0 (0%)
uvindex             int64      0 (0%)     0 (0%)
severerisk        float64  2019 (73%)     0 (0%)
sunrise            object      0 (0%)     0 (0%)
sunset             object      0 (0%)     0 (0%)
moonphase         float64      0 (0%)    93 (3%)
conditions         object      0 (0%)     0 (0%)
description        object      0 (0%)     0 (0%)
icon               object      0 (0%)     0 (0%)
stations           object      0 (0%)     0 (0%)

<table border="0" class="dataframe">  <tbody>    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>datetime</th>
      <th>tempmax</th>
      <th>tempmin</th>
      <th>temp</th>
      <th>feelslikemax</th>
      <th>feelslikemin</th>
      <th>feelslike</th>
      <th>dew</th>
      <th>humidity</th>
      <th>precip</th>
      <th>precipprob</th>
      <th>precipcover</th>
      <th>preciptype</th>
      <th>snow</th>
      <th>snowdepth</th>
      <th>windgust</th>
      <th>windspeed</th>
      <th>winddir</th>
      <th>sealevelpressure</th>
      <th>cloudcover</th>
      <th>visibility</th>
      <th>solarradiation</th>
      <th>solarenergy</th>
      <th>uvindex</th>
      <th>severerisk</th>
      <th>sunrise</th>
      <th>sunset</th>
      <th>moonphase</th>
      <th>conditions</th>
      <th>description</th>
      <th>icon</th>
      <th>stations</th>
    </tr>
    <tr>
      <th>0</th>
      <td>kuala lumpur</td>
      <td>7/1/2016</td>
      <td>34.5</td>
      <td>27.8</td>
      <td>30.6</td>
      <td>44.2</td>
      <td>32.1</td>
      <td>36.8</td>
      <td>25.0</td>
      <td>73.3</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>16.4</td>
      <td>154.6</td>
      <td>1009.1</td>
      <td>89.0</td>
      <td>9.0</td>
      <td>174.0</td>
      <td>15.0</td>
      <td>6</td>
      <td>NaN</td>
      <td>2016-07-01T07:08:07</td>
      <td>2016-07-01T19:26:07</td>
      <td>0.88</td>
      <td>Partially cloudy</td>
      <td>Partly cloudy throughout the day.</td>
      <td>partly-cloudy-day</td>
      <td>48647099999,48650099999,WMSA,WMKK</td>
    </tr>
    <tr>
      <th>1</th>
      <td>kuala lumpur</td>
      <td>7/2/2016</td>
      <td>34.8</td>
      <td>25.9</td>
      <td>30.3</td>
      <td>42.8</td>
      <td>25.9</td>
      <td>35.0</td>
      <td>24.4</td>
      <td>71.9</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14.8</td>
      <td>147.9</td>
      <td>1009.8</td>
      <td>89.0</td>
      <td>9.1</td>
      <td>193.3</td>
      <td>16.6</td>
      <td>7</td>
      <td>NaN</td>
      <td>2016-07-02T07:08:20</td>
      <td>2016-07-02T19:26:17</td>
      <td>0.92</td>
      <td>Partially cloudy</td>
      <td>Partly cloudy throughout the day.</td>
      <td>partly-cloudy-day</td>
      <td>48647099999,48650099999,WMSA,WMKK</td>
    </tr>
    <tr>
      <th>2</th>
      <td>kuala lumpur</td>
      <td>7/3/2016</td>
      <td>35.8</td>
      <td>25.4</td>
      <td>30.7</td>
      <td>44.0</td>
      <td>25.4</td>
      <td>35.4</td>
      <td>23.9</td>
      <td>68.5</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15.7</td>
      <td>151.6</td>
      <td>1009.9</td>
      <td>87.9</td>
      <td>9.8</td>
      <td>196.4</td>
      <td>17.1</td>
      <td>7</td>
      <td>NaN</td>
      <td>2016-07-03T07:08:32</td>
      <td>2016-07-03T19:26:27</td>
      <td>0.95</td>
      <td>Partially cloudy</td>
      <td>Partly cloudy throughout the day.</td>
      <td>partly-cloudy-day</td>
      <td>48647099999,48650099999,WMSA,WMKK</td>
    </tr>
    <tr>
      <th>3</th>
      <td>kuala lumpur</td>
      <td>7/4/2016</td>
      <td>34.8</td>
      <td>26.0</td>
      <td>30.4</td>
      <td>44.3</td>
      <td>26.0</td>
      <td>36.1</td>
      <td>24.7</td>
      <td>72.4</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12.5</td>
      <td>188.0</td>
      <td>1009.1</td>
      <td>89.0</td>
      <td>9.1</td>
      <td>221.2</td>
      <td>19.0</td>
      <td>8</td>
      <td>NaN</td>
      <td>2016-07-04T07:08:44</td>
      <td>2016-07-04T19:26:36</td>
      <td>0.00</td>
      <td>Partially cloudy</td>
      <td>Partly cloudy throughout the day.</td>
      <td>partly-cloudy-day</td>
      <td>48647099999,48650099999,WMSA,WMKK</td>
    </tr>
    <tr>
      <th>4</th>
      <td>kuala lumpur</td>
      <td>7/5/2016</td>
      <td>33.9</td>
      <td>26.1</td>
      <td>30.0</td>
      <td>43.2</td>
      <td>26.1</td>
      <td>36.1</td>
      <td>25.3</td>
      <td>76.6</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.4</td>
      <td>182.1</td>
      <td>1008.4</td>
      <td>89.0</td>
      <td>8.4</td>
      <td>139.2</td>
      <td>11.9</td>
      <td>5</td>
      <td>NaN</td>
      <td>2016-07-05T07:08:56</td>
      <td>2016-07-05T19:26:45</td>
      <td>0.02</td>
      <td>Partially cloudy</td>
      <td>Partly cloudy throughout the day.</td>
      <td>partly-cloudy-day</td>
      <td>48647099999,48650099999,WMSA,WMKK</td>
    </tr>
  </tbody>
</table>
</pre>

For Malaysians, the first columns that we would remove are 'snow' and 'snowdepth' simply because it NEVER snows here, we have basically just sunny mornings and rainy afternoons here!

Other columns that are deemed redundant are 'name' because this column only takes one 'Kuala Lumpur' value, and also 'stations' which is probably unrelated to rainfall.

Just to be sure, let's take a look at all the possible values of these categorical columns.

<details>
<summary>View Code</summary>

```python
print(str(df['name'].value_counts())
+"\n\n"+str(df['preciptype'].value_counts())
+"\n\n"+str(df['conditions'].value_counts())
+"\n\n"+str(df['description'].value_counts())
+"\n\n"+str(df['icon'].value_counts())
+"\n\n"+str(df['stations'].value_counts()))
```

### Output
</details>
<pre>
name
kuala lumpur             1838
Kuala Lumpur,Malaysia     909
Name: count, dtype: int64

preciptype
rain    2027
Name: count, dtype: int64

conditions
Rain, Partially cloudy    1751
Partially cloudy           896
Rain, Overcast              93
Overcast                     7
Name: count, dtype: int64

description
Partly cloudy throughout the day.                                             896
Partly cloudy throughout the day with rain.                                   593
Partly cloudy throughout the day with late afternoon rain.                    477
Partly cloudy throughout the day with early morning rain.                     200
Partly cloudy throughout the day with morning rain.                           198
Partly cloudy throughout the day with afternoon rain.                         176
Cloudy skies throughout the day with rain.                                     50
Partly cloudy throughout the day with rain clearing later.                     45
Partly cloudy throughout the day with rain in the morning and afternoon.       44
Partly cloudy throughout the day with a chance of rain throughout the day.     18
Cloudy skies throughout the day with late afternoon rain.                      15
Cloudy skies throughout the day with afternoon rain.                            8
Cloudy skies throughout the day.                                                7
Cloudy skies throughout the day with rain in the morning and afternoon.         7
Cloudy skies throughout the day with early morning rain.                        6
Cloudy skies throughout the day with a chance of rain throughout the day.       3
Cloudy skies throughout the day with morning rain.                              2
Cloudy skies throughout the day with rain clearing later.                       2
Name: count, dtype: int64

icon
rain                 1844
partly-cloudy-day     896
cloudy                  7
Name: count, dtype: int64

stations
48647099999,48650099999,WMSA,WMKK           2737
WMSA,WMKK                                      5
48647099999,48650099999,WMSA,remote,WMKK       3
4,864,709,999,948,650,000,000                  1
48647099999,48650099999,remote,WMSA,WMKK       1
Name: count, dtype: int64
</pre>

Indeed, 'name' and 'stations' should be ditched as expected. 

However, there is something interesting with the 'description' column, it does not only tells us whether it rained on that day, but specifically which **part** of the day. (Morning, afternoon, evening etc.). We should extract the time of the day when it rained.

# Data Cleansing
We will now extract the features from 'description' mentioned above, along with other miscellaneous tasks.

<details>
<summary>View Code</summary>

```python
warnings.filterwarnings(action='ignore', category=UserWarning)

df['datetime'] = pd.to_datetime(df['datetime'])
df['Year'] = df['datetime'].dt.year
df['Month'] = df['datetime'].dt.month
df['Day'] = df['datetime'].dt.day
df = df.set_index('datetime').asfreq('D')
df.drop(['windgust','name','precipprob','precipcover','preciptype','snow','snowdepth','severerisk','icon','conditions'], axis=1, inplace=True)
df['description'] = df['description'].str.replace('Partly cloudy throughout the day','')
df['description']=df['description'].str.replace('early morning','dawn')
df['description']=df['description'].str.replace('late afternoon','evening')
df['description']=df['description'].str.replace('with rain.','with rain in the dawn, morning, afternoon, and evening.')
df['description']=df['description'].str.replace('with a chance to rain throughout the day.','with rain in the dawn, morning, afternoon, and evening.')
df['description']=df['description'].str.replace('with rain clearing later','with rain in the dawn, morning, and afternoon')
df['rain_dawn'] = df['description'].str.contains(r'(dawn)').astype(int)
df['rain_morning'] = df['description'].str.contains(r'(morning)').astype(int)
df['rain_afternoon'] = df['description'].str.contains(r'(afternoon)').astype(int)
df['rain_evening'] = df['description'].str.contains(r'(evening)').astype(int)
warnings.resetwarnings()
df.drop(['description','stations'], axis=1, inplace=True)
df.head()
```


### Output
</details>
<pre><table border="0" class="dataframe">  <tbody>
    <tr style="text-align: right;">
      <th></th>
      <th>tempmax</th>
      <th>tempmin</th>
      <th>temp</th>
      <th>feelslikemax</th>
      <th>feelslikemin</th>
      <th>feelslike</th>
      <th>dew</th>
      <th>humidity</th>
      <th>precip</th>
      <th>windspeed</th>
      <th>winddir</th>
      <th>sealevelpressure</th>
      <th>cloudcover</th>
      <th>visibility</th>
      <th>solarradiation</th>
      <th>solarenergy</th>
      <th>uvindex</th>
      <th>sunrise</th>
      <th>sunset</th>
      <th>moonphase</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
      <th>rain_dawn</th>
      <th>rain_morning</th>
      <th>rain_afternoon</th>
      <th>rain_evening</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
    <tr>
      <th>2016-07-01</th>
      <td>34.5</td>
      <td>27.8</td>
      <td>30.6</td>
      <td>44.2</td>
      <td>32.1</td>
      <td>36.8</td>
      <td>25.0</td>
      <td>73.3</td>
      <td>0.0</td>
      <td>16.4</td>
      <td>154.6</td>
      <td>1009.1</td>
      <td>89.0</td>
      <td>9.0</td>
      <td>174.0</td>
      <td>15.0</td>
      <td>6</td>
      <td>2016-07-01T07:08:07</td>
      <td>2016-07-01T19:26:07</td>
      <td>0.88</td>
      <td>2016</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-07-02</th>
      <td>34.8</td>
      <td>25.9</td>
      <td>30.3</td>
      <td>42.8</td>
      <td>25.9</td>
      <td>35.0</td>
      <td>24.4</td>
      <td>71.9</td>
      <td>0.0</td>
      <td>14.8</td>
      <td>147.9</td>
      <td>1009.8</td>
      <td>89.0</td>
      <td>9.1</td>
      <td>193.3</td>
      <td>16.6</td>
      <td>7</td>
      <td>2016-07-02T07:08:20</td>
      <td>2016-07-02T19:26:17</td>
      <td>0.92</td>
      <td>2016</td>
      <td>7</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-07-03</th>
      <td>35.8</td>
      <td>25.4</td>
      <td>30.7</td>
      <td>44.0</td>
      <td>25.4</td>
      <td>35.4</td>
      <td>23.9</td>
      <td>68.5</td>
      <td>0.0</td>
      <td>15.7</td>
      <td>151.6</td>
      <td>1009.9</td>
      <td>87.9</td>
      <td>9.8</td>
      <td>196.4</td>
      <td>17.1</td>
      <td>7</td>
      <td>2016-07-03T07:08:32</td>
      <td>2016-07-03T19:26:27</td>
      <td>0.95</td>
      <td>2016</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-07-04</th>
      <td>34.8</td>
      <td>26.0</td>
      <td>30.4</td>
      <td>44.3</td>
      <td>26.0</td>
      <td>36.1</td>
      <td>24.7</td>
      <td>72.4</td>
      <td>0.0</td>
      <td>12.5</td>
      <td>188.0</td>
      <td>1009.1</td>
      <td>89.0</td>
      <td>9.1</td>
      <td>221.2</td>
      <td>19.0</td>
      <td>8</td>
      <td>2016-07-04T07:08:44</td>
      <td>2016-07-04T19:26:36</td>
      <td>0.00</td>
      <td>2016</td>
      <td>7</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-07-05</th>
      <td>33.9</td>
      <td>26.1</td>
      <td>30.0</td>
      <td>43.2</td>
      <td>26.1</td>
      <td>36.1</td>
      <td>25.3</td>
      <td>76.6</td>
      <td>0.0</td>
      <td>18.4</td>
      <td>182.1</td>
      <td>1008.4</td>
      <td>89.0</td>
      <td>8.4</td>
      <td>139.2</td>
      <td>11.9</td>
      <td>5</td>
      <td>2016-07-05T07:08:56</td>
      <td>2016-07-05T19:26:45</td>
      <td>0.02</td>
      <td>2016</td>
      <td>7</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</pre>
</details>

## Important: Prediction Based on Historical Information
An important thing to note here is the next day's Rainfall should be predicted based on data that is available on the day before, and NOT data on the current day (as data on the current day can only be obtained at the end of the day). Therefore, we will bring the weather data 1 day forward.

<details>
<summary>View Code</summary>

```python
df=df.assign(**{
    f'{col} (lag_1)': df[col].shift(1)
    for col in df.drop(['Year','Month','Day'], axis=1)
    })
df['Rainfall'] = df['precip'].apply(lambda x: 1 if x > 0 else x)
df['precipitation'] = df['precip']
df = df.iloc[1:,20:].drop(['rain_dawn','rain_morning','rain_afternoon','rain_evening'], axis=1)
df.head()
```

### Output
</details>
<pre>
<table border="0" class="dataframe">  <tbody>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
      <th>tempmax (lag_1)</th>
      <th>tempmin (lag_1)</th>
      <th>temp (lag_1)</th>
      <th>feelslikemax (lag_1)</th>
      <th>feelslikemin (lag_1)</th>
      <th>feelslike (lag_1)</th>
      <th>dew (lag_1)</th>
      <th>humidity (lag_1)</th>
      <th>precip (lag_1)</th>
      <th>windspeed (lag_1)</th>
      <th>winddir (lag_1)</th>
      <th>sealevelpressure (lag_1)</th>
      <th>cloudcover (lag_1)</th>
      <th>visibility (lag_1)</th>
      <th>solarradiation (lag_1)</th>
      <th>solarenergy (lag_1)</th>
      <th>uvindex (lag_1)</th>
      <th>sunrise (lag_1)</th>
      <th>sunset (lag_1)</th>
      <th>moonphase (lag_1)</th>
      <th>rain_dawn (lag_1)</th>
      <th>rain_morning (lag_1)</th>
      <th>rain_afternoon (lag_1)</th>
      <th>rain_evening (lag_1)</th>
      <th>Rainfall</th>
      <th>precipitation</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
    <tr>
      <th>2016-07-02</th>
      <td>2016</td>
      <td>7</td>
      <td>2</td>
      <td>34.5</td>
      <td>27.8</td>
      <td>30.6</td>
      <td>44.2</td>
      <td>32.1</td>
      <td>36.8</td>
      <td>25.0</td>
      <td>73.3</td>
      <td>0.0</td>
      <td>16.4</td>
      <td>154.6</td>
      <td>1009.1</td>
      <td>89.0</td>
      <td>9.0</td>
      <td>174.0</td>
      <td>15.0</td>
      <td>6.0</td>
      <td>2016-07-01T07:08:07</td>
      <td>2016-07-01T19:26:07</td>
      <td>0.88</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>2016-07-03</th>
      <td>2016</td>
      <td>7</td>
      <td>3</td>
      <td>34.8</td>
      <td>25.9</td>
      <td>30.3</td>
      <td>42.8</td>
      <td>25.9</td>
      <td>35.0</td>
      <td>24.4</td>
      <td>71.9</td>
      <td>0.0</td>
      <td>14.8</td>
      <td>147.9</td>
      <td>1009.8</td>
      <td>89.0</td>
      <td>9.1</td>
      <td>193.3</td>
      <td>16.6</td>
      <td>7.0</td>
      <td>2016-07-02T07:08:20</td>
      <td>2016-07-02T19:26:17</td>
      <td>0.92</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>2016-07-04</th>
      <td>2016</td>
      <td>7</td>
      <td>4</td>
      <td>35.8</td>
      <td>25.4</td>
      <td>30.7</td>
      <td>44.0</td>
      <td>25.4</td>
      <td>35.4</td>
      <td>23.9</td>
      <td>68.5</td>
      <td>0.0</td>
      <td>15.7</td>
      <td>151.6</td>
      <td>1009.9</td>
      <td>87.9</td>
      <td>9.8</td>
      <td>196.4</td>
      <td>17.1</td>
      <td>7.0</td>
      <td>2016-07-03T07:08:32</td>
      <td>2016-07-03T19:26:27</td>
      <td>0.95</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>2016-07-05</th>
      <td>2016</td>
      <td>7</td>
      <td>5</td>
      <td>34.8</td>
      <td>26.0</td>
      <td>30.4</td>
      <td>44.3</td>
      <td>26.0</td>
      <td>36.1</td>
      <td>24.7</td>
      <td>72.4</td>
      <td>0.0</td>
      <td>12.5</td>
      <td>188.0</td>
      <td>1009.1</td>
      <td>89.0</td>
      <td>9.1</td>
      <td>221.2</td>
      <td>19.0</td>
      <td>8.0</td>
      <td>2016-07-04T07:08:44</td>
      <td>2016-07-04T19:26:36</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>2016-07-06</th>
      <td>2016</td>
      <td>7</td>
      <td>6</td>
      <td>33.9</td>
      <td>26.1</td>
      <td>30.0</td>
      <td>43.2</td>
      <td>26.1</td>
      <td>36.1</td>
      <td>25.3</td>
      <td>76.6</td>
      <td>0.0</td>
      <td>18.4</td>
      <td>182.1</td>
      <td>1008.4</td>
      <td>89.0</td>
      <td>8.4</td>
      <td>139.2</td>
      <td>11.9</td>
      <td>5.0</td>
      <td>2016-07-05T07:08:56</td>
      <td>2016-07-05T19:26:45</td>
      <td>0.02</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>15.913</td>
    </tr>
  </tbody>
</table>
</pre>

# EDA
## Actual daily temperature
As a city located strategically near the equator, the weather in Kuala Lumpur is relatively stable with mean temperatures around 28±1.0°C throughout the year, although it is slightly hotter from April to June.

<details>
<summary>View Code</summary>

```python
window=30
plt.figure(figsize=(8, 4))
sns.lineplot(x=df['2023-01-01':'2023-12-31'].index, y='tempmax (lag_1)', data=df['2023-01-01':'2023-12-31'], color='sandybrown', label='max_temp')
sns.lineplot(x=df['2023-01-01':'2023-12-31'].index, y=df['2023-01-01':'2023-12-31']['tempmax (lag_1)'].rolling(window=window, center=True).mean(), data=df['2023-01-01':'2023-12-31'], color='sienna')

sns.lineplot(x=df['2023-01-01':'2023-12-31'].index, y='temp (lag_1)', data=df['2023-01-01':'2023-12-31'], color='orchid', label='temp')
sns.lineplot(x=df['2023-01-01':'2023-12-31'].index, y=df['2023-01-01':'2023-12-31']['temp (lag_1)'].rolling(window=window, center=True).mean(), data=df['2023-01-01':'2023-12-31'], color='purple')

sns.lineplot(x=df['2023-01-01':'2023-12-31'].index, y='tempmin (lag_1)', data=df['2023-01-01':'2023-12-31'], color='skyblue', label='min_temp')
sns.lineplot(x=df['2023-01-01':'2023-12-31'].index, y=df['2023-01-01':'2023-12-31']['tempmin (lag_1)'].rolling(window=window, center=True).mean(), data=df['2023-01-01':'2023-12-31'], color='navy')

plt.legend(loc='lower right')
plt.show()
```

### Output
</details>
![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/kuala-lumpur-rain-prediction/images/1.jpg?raw=true)

## Comparison between actual and felt temperature

Interesting enough, the people living here tend to feel much hotter than usual at all times, and the difference get more obvious as the actual temperature gets higher.

However, as the actual temperature approaches the minimum of 24°C, the felt temperature has no difference from the actual temperature.

<details>
<summary>View Code</summary>

```python
window=30
fig, axs = plt.subplots(3, 1, figsize=(8, 10))
sns.lineplot(x=df['2023-01-01':'2023-12-31'].index, y='feelslikemax (lag_1)', data=df['2023-01-01':'2023-12-31'], color='lightsalmon', ax=axs[0])
sns.lineplot(x=df['2023-01-01':'2023-12-31'].index, y='tempmax (lag_1)', data=df['2023-01-01':'2023-12-31'], color='mediumspringgreen', ax=axs[0])
sns.lineplot(x=df['2023-01-01':'2023-12-31'].index, y=df['2023-01-01':'2023-12-31']['feelslikemax (lag_1)'].rolling(window=window, center=True).mean(), data=df['2023-01-01':'2023-12-31'], color='brown', label='feel_max', ax=axs[0])
sns.lineplot(x=df['2023-01-01':'2023-12-31'].index, y=df['2023-01-01':'2023-12-31']['tempmax (lag_1)'].rolling(window=window, center=True).mean(), data=df['2023-01-01':'2023-12-31'], color='seagreen', label='actual_max', ax=axs[0])
axs[0].set(ylabel='Max Temperature', xlabel='', title='Actual vs Felt Temperature')
sns.lineplot(x=df['2023-01-01':'2023-12-31'].index, y='feelslike (lag_1)', data=df['2023-01-01':'2023-12-31'], color='lightsalmon', ax=axs[1])
sns.lineplot(x=df['2023-01-01':'2023-12-31'].index, y='temp (lag_1)', data=df['2023-01-01':'2023-12-31'], color='mediumspringgreen', ax=axs[1])
sns.lineplot(x=df['2023-01-01':'2023-12-31'].index, y=df['2023-01-01':'2023-12-31']['feelslike (lag_1)'].rolling(window=window, center=True).mean(), data=df['2023-01-01':'2023-12-31'], color='brown', label='feel', ax=axs[1])
sns.lineplot(x=df['2023-01-01':'2023-12-31'].index, y=df['2023-01-01':'2023-12-31']['temp (lag_1)'].rolling(window=window, center=True).mean(), data=df['2023-01-01':'2023-12-31'], color='seagreen', label='actual', ax=axs[1])
axs[1].set(ylabel='Mean Temperature', xlabel='')
sns.lineplot(x=df['2023-01-01':'2023-12-31'].index, y='feelslikemin (lag_1)', data=df['2023-01-01':'2023-12-31'], color='lightsalmon', ax=axs[2])
sns.lineplot(x=df['2023-01-01':'2023-12-31'].index, y='tempmin (lag_1)', data=df['2023-01-01':'2023-12-31'], color='mediumspringgreen', ax=axs[2])
sns.lineplot(x=df['2023-01-01':'2023-12-31'].index, y=df['2023-01-01':'2023-12-31']['feelslikemin (lag_1)'].rolling(window=window, center=True).mean(), data=df['2023-01-01':'2023-12-31'], color='brown', label='feel_min', ax=axs[2])
sns.lineplot(x=df['2023-01-01':'2023-12-31'].index, y=df['2023-01-01':'2023-12-31']['tempmin (lag_1)'].rolling(window=window, center=True).mean(), data=df['2023-01-01':'2023-12-31'], color='seagreen', label='actual_min', ax=axs[2])
axs[2].set(ylabel='Min Temperature', xlabel='Date')
plt.show()
```

### Output
</details>
![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/kuala-lumpur-rain-prediction/images/2.png?raw=true)

## Distribution of Rainfall through the year

Rainfalls are generally higher around mid-March and November to December, this is expected as those months are the Monsoon season of Malaysia.

<details>
<summary>View Code</summary>

```python
plt.figure(figsize=(16,6))
df[(df['Year']>2020) & (df['Year']<2024)].groupby(['Month','Year'])['precipitation'].sum().unstack().plot(kind='area', stacked=True)
plt.xlabel('Month')
plt.ylabel('Rainfall')
plt.legend(loc='upper left', bbox_to_anchor=(1,1)) 
plt.title('Monthly Rainfall by Year')
plt.show()
```

### Output
</details>
![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/kuala-lumpur-rain-prediction/images/3.png?raw=true)

## Univariate Analysis
By looking at the distribution of individual features, we would be able to identify if there is any skewness or outliers present.

<details>
<summary>View Code</summary>

```python
fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(12,20))
for i, column in enumerate(df.drop(['Rainfall','sunset (lag_1)','sunrise (lag_1)','Year','Month','Day'], axis=1).columns):
    sns.histplot(df[column],ax=axes[i//4,i%4])
```

### Output
</details>
![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/kuala-lumpur-rain-prediction/images/4.png?raw=true)

## Bivariate Analysis
### Scatterplots
Now we look at how each of the features relate to the daily precipitation.

There are no obvious relationship found between these features and precipitation, but we do notice at least 3 outliers in the precipitation which have values above 100.

<details>
<summary>View Code</summary>

```python
fig, axes = plt.subplots(nrows=7, ncols=4, figsize=(12,25))
for i, column in enumerate(df.drop(['Rainfall','sunrise (lag_1)','sunset (lag_1)'], axis=1).columns):
    sns.scatterplot(x=df[column], y=df['precipitation'],ax=axes[i//4,i%4])
```

### Output
</details>
![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/kuala-lumpur-rain-prediction/images/5.png?raw=true)

### Correlation Analysis

An analysis into the correlation between pairwise variables helps us identify variables which are highly correlated to each other and leave only one of them in our analysis.

From the heatmap below, it is obvious that the temperature components are pretty much correlated, the same goes for the solar components and uv index. We will pick only one of each group.

<details>
<summary>View Code</summary>

```python
plt.figure(figsize=(10, 10))
sns.heatmap(df.drop(['sunrise (lag_1)','sunset (lag_1)'],axis=1).corr(), annot=True, cmap='coolwarm', fmt=".1f", linewidths=0.5)
plt.show()
```

### Output
</details>
![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/kuala-lumpur-rain-prediction/images/6.png?raw=true)

```python
df.drop(['solarenergy (lag_1)','uvindex (lag_1)','tempmin (lag_1)','temp (lag_1)','tempmax (lag_1)','feelslikemin (lag_1)','feelslikemax (lag_1)','feelslike (lag_1)','sunrise (lag_1)','sunset (lag_1)'], axis=1, inplace=True)
df['monsoon_month']=df['Month'].isin([3,4, 11, 12]).astype(int)
df.head()
```

### Output

<pre><table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
      <th>dew (lag_1)</th>
      <th>humidity (lag_1)</th>
      <th>precip (lag_1)</th>
      <th>windspeed (lag_1)</th>
      <th>winddir (lag_1)</th>
      <th>sealevelpressure (lag_1)</th>
      <th>cloudcover (lag_1)</th>
      <th>visibility (lag_1)</th>
      <th>solarradiation (lag_1)</th>
      <th>moonphase (lag_1)</th>
      <th>rain_dawn (lag_1)</th>
      <th>rain_morning (lag_1)</th>
      <th>rain_afternoon (lag_1)</th>
      <th>rain_evening (lag_1)</th>
      <th>Rainfall</th>
      <th>precipitation</th>
      <th>monsoon_month</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
    <tr>
      <th>2016-07-02</th>
      <td>2016</td>
      <td>7</td>
      <td>2</td>
      <td>25.0</td>
      <td>73.3</td>
      <td>0.0</td>
      <td>16.4</td>
      <td>154.6</td>
      <td>1009.1</td>
      <td>89.0</td>
      <td>9.0</td>
      <td>174.0</td>
      <td>0.88</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-07-03</th>
      <td>2016</td>
      <td>7</td>
      <td>3</td>
      <td>24.4</td>
      <td>71.9</td>
      <td>0.0</td>
      <td>14.8</td>
      <td>147.9</td>
      <td>1009.8</td>
      <td>89.0</td>
      <td>9.1</td>
      <td>193.3</td>
      <td>0.92</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-07-04</th>
      <td>2016</td>
      <td>7</td>
      <td>4</td>
      <td>23.9</td>
      <td>68.5</td>
      <td>0.0</td>
      <td>15.7</td>
      <td>151.6</td>
      <td>1009.9</td>
      <td>87.9</td>
      <td>9.8</td>
      <td>196.4</td>
      <td>0.95</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-07-05</th>
      <td>2016</td>
      <td>7</td>
      <td>5</td>
      <td>24.7</td>
      <td>72.4</td>
      <td>0.0</td>
      <td>12.5</td>
      <td>188.0</td>
      <td>1009.1</td>
      <td>89.0</td>
      <td>9.1</td>
      <td>221.2</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-07-06</th>
      <td>2016</td>
      <td>7</td>
      <td>6</td>
      <td>25.3</td>
      <td>76.6</td>
      <td>0.0</td>
      <td>18.4</td>
      <td>182.1</td>
      <td>1008.4</td>
      <td>89.0</td>
      <td>8.4</td>
      <td>139.2</td>
      <td>0.02</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>15.913</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</pre>

### Outlier & Missing Value Treatment

<details>
<summary>View Code</summary>

```python
df['visibility (lag_1)']=np.clip(df['visibility (lag_1)'], a_max=None, a_min=df['visibility (lag_1)'].quantile(0.005))
df['sealevelpressure (lag_1)']=df['sealevelpressure (lag_1)'].fillna(np.mean(df['sealevelpressure (lag_1)']))
summary(df)
```

### Output
</details>
<pre>
                             Type     NaN       Zeros
Year                        int32  0 (0%)      0 (0%)
Month                       int32  0 (0%)      0 (0%)
Day                         int32  0 (0%)      0 (0%)
dew (lag_1)               float64  0 (0%)      0 (0%)
humidity (lag_1)          float64  0 (0%)      0 (0%)
precip (lag_1)            float64  0 (0%)   903 (32%)
windspeed (lag_1)         float64  0 (0%)      0 (0%)
winddir (lag_1)           float64  0 (0%)      0 (0%)
sealevelpressure (lag_1)  float64  0 (0%)      0 (0%)
cloudcover (lag_1)        float64  0 (0%)      0 (0%)
visibility (lag_1)        float64  0 (0%)      0 (0%)
solarradiation (lag_1)    float64  0 (0%)      0 (0%)
moonphase (lag_1)         float64  0 (0%)     93 (3%)
rain_dawn (lag_1)         float64  0 (0%)  1850 (67%)
rain_morning (lag_1)      float64  0 (0%)  1805 (65%)
rain_afternoon (lag_1)    float64  0 (0%)  1821 (66%)
rain_evening (lag_1)      float64  0 (0%)  1611 (58%)
Rainfall                  float64  0 (0%)   902 (32%)
precipitation             float64  0 (0%)   902 (32%)
monsoon_month               int64  0 (0%)  1831 (66%)
</pre>

# Feature Scaling
by scaling each feature to the range (0,1) using SKlearn's MinMaxSaler, all numerical features will fall within the same range. This will improve the performance of distance-based algorithms and speed up the convergence of gradient descent algorithms in the next part.

<details>
<summary>View Code</summary>

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_bin=df.iloc[:,-7:]
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df.iloc[:,-7:]=df_bin
df.drop('precipitation', axis=1, inplace=True)
df.head()
```

### Output
</details>
<pre>
<table border="0" class="dataframe"><tbody>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
      <th>dew (lag_1)</th>
      <th>humidity (lag_1)</th>
      <th>precip (lag_1)</th>
      <th>windspeed (lag_1)</th>
      <th>winddir (lag_1)</th>
      <th>sealevelpressure (lag_1)</th>
      <th>cloudcover (lag_1)</th>
      <th>visibility (lag_1)</th>
      <th>solarradiation (lag_1)</th>
      <th>moonphase (lag_1)</th>
      <th>rain_dawn (lag_1)</th>
      <th>rain_morning (lag_1)</th>
      <th>rain_afternoon (lag_1)</th>
      <th>rain_evening (lag_1)</th>
      <th>Rainfall</th>
      <th>monsoon_month</th>
    </tr>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.545455</td>
      <td>0.033333</td>
      <td>0.698630</td>
      <td>0.350725</td>
      <td>0.0</td>
      <td>0.337423</td>
      <td>0.428651</td>
      <td>0.360465</td>
      <td>0.914616</td>
      <td>0.690909</td>
      <td>0.512873</td>
      <td>0.897959</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.545455</td>
      <td>0.066667</td>
      <td>0.616438</td>
      <td>0.310145</td>
      <td>0.0</td>
      <td>0.288344</td>
      <td>0.409978</td>
      <td>0.441860</td>
      <td>0.914616</td>
      <td>0.709091</td>
      <td>0.578252</td>
      <td>0.938776</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.545455</td>
      <td>0.100000</td>
      <td>0.547945</td>
      <td>0.211594</td>
      <td>0.0</td>
      <td>0.315951</td>
      <td>0.420290</td>
      <td>0.453488</td>
      <td>0.898698</td>
      <td>0.836364</td>
      <td>0.588753</td>
      <td>0.969388</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.545455</td>
      <td>0.133333</td>
      <td>0.657534</td>
      <td>0.324638</td>
      <td>0.0</td>
      <td>0.217791</td>
      <td>0.521739</td>
      <td>0.360465</td>
      <td>0.914616</td>
      <td>0.709091</td>
      <td>0.672764</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.545455</td>
      <td>0.166667</td>
      <td>0.739726</td>
      <td>0.446377</td>
      <td>0.0</td>
      <td>0.398773</td>
      <td>0.505295</td>
      <td>0.279070</td>
      <td>0.914616</td>
      <td>0.581818</td>
      <td>0.394986</td>
      <td>0.020408</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</pre>

# Model Building
For a binary classification problem like this, the **Logistic Regression** model would be the baseline model due to it's simple yet highly interpretable nature. 

As for the split, we opt for a 70% train, 20% validation and 10% test split, where the train data is cross-validated on 8 folds with the validation data before evaluating it using the test data.

Also, for any classification problems, we should always check if there is any serious imbalance in the outcomes, otherwise the model might not learn well the parameters for each outcome.

```python
df['Rainfall'].value_counts()
```

### Output
<pre>
Rainfall
1.0    1844
0.0     902
Name: count, dtype: int64
</pre>

The ratio of 1 to 0 here is approximately 2:1, obviously this data is imbalanced. In the upcoming part, we will utilize SMOTE to perform downsampling and upsampling on the outcomes respectively so the model will learn both outcomes equally well.

Let's import the Machine Learning libraries conveniently provided by SKlearn:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, make_scorer, roc_curve, rac_scorer
import optuna
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
optuna.logging.set_verbosity(optuna.logging.WARNING)
```

### Baseline Model - Logistic Regression

<details>
<summary>View Code</summary>

```python
X=df.drop('Rainfall', axis=1)
y=df['Rainfall']

x_train, x_testval, y_train, y_testval = train_test_split(X, y, test_size=0.30, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_testval, y_testval, test_size=0.33, random_state=42)

over = SMOTE(sampling_strategy=0.6, random_state=42)
under = RandomUnderSampler(sampling_strategy=0.6, random_state=42)
smt = Pipeline(steps=[('o', over), ('u', under)])
x_train, y_train=smt.fit_resample(x_train, y_train)

LR=LogisticRegression(random_state=42)
LR.fit(x_train, y_train)
y_pred = LR.predict(x_val)
y_pred_proba = LR.predict_proba(x_val)[::,1]
LR_fp, LR_tp, _ = roc_curve(y_val,  y_pred_proba)
print("Accuracy: ", "%.4f" % accuracy_score(y_val, y_pred))
print("Precision: ", "%.4f" % precision_score(y_val, y_pred))
print("Recall: ", "%.4f" % recall_score(y_val, y_pred))
print("F1-Score: ", "%.4f" % f1_score(y_val, y_pred))
confusion_matrix(y_val, y_pred)
```

### Output
</details>
<pre>
Accuracy:  0.6667
Precision:  0.7367
Recall:  0.7844
F1-Score:  0.7598
array([[ 77, 104],
       [ 80, 291]])
</pre>

Although the accuracy isn't really satisfactory, however the advantage of using this algorithm is we are able to find out which features play a more important role in predicting the final outcome.

By plotting the feature importances:

<details>
<summary>View Code</summary>

```python
coeff=pd.DataFrame(zip(x_train.columns, np.transpose(LR.coef_.flatten())), columns=['features', 'coef']).sort_values(by='coef')
plt.barh(coeff['features'], coeff['coef'], color='skyblue')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Logistic Regression Feature Importances')
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.show()
```

### Output
</details>
![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/kuala-lumpur-rain-prediction/images/7.png?raw=true)
