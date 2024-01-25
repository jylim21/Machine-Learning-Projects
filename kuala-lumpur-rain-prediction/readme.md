<h1 align="center">Rain Prediction in Kuala Lumpur</h1>

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/kuala-lumpur-rain-prediction/images/kl-rain.jpg?raw=true)

Kuala Lumpur has a tropical rainforest climate with abundant rainfall throughout the year. On average it receives around 2,400 mm of rain annually, with elevated precipitation during monsoon seasons. 

Flash floods frequently disrupt traffic and public transport during torrential downpours. Meanwhile, long dry spells threaten water supply and can hamper agriculture. Reliable rainfall forecasting can help this city prepare for these events through improved stormwater management and water resource planning. Furthermore, sectors like tourism, construction and renewable energy can optimize operations using accurate precipitation predictions.

This project aims to build a machine learning model to forecast rainfall levels in Kuala Lumpur based on historical meteorological data. The precipitation outlook generated can aid urban planning and climate resilience. Government agencies can leverage the forecasts to develop flood mitigation infrastructure and ensure continuous water supply during hot seasons. Businesses can also adjust plans based on wet and dry period forecasts. Overall, enhanced rainfall prediction stands to provide economic and societal benefits for Malaysia's vibrant capital.

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

def dir(bearing):
    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    index = int((bearing + 22.5) % 360 / 45)
    return dirs[index]
df['winddir'] = df['winddir'].apply(dir)
winddir=pd.get_dummies(df['winddir'])
df = pd.concat([df, winddir.set_index(df.index)], axis = 1)
df.iloc[:,-8:]=df.iloc[:,-8:].astype(int)
df.drop('winddir', axis=1, inplace=True)

df['Year'] = df['datetime'].dt.year
df['Month'] = df['datetime'].dt.month
df['Day'] = df['datetime'].dt.day
df = df.set_index('datetime').asfreq('D')

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
      <th>sealevelpressure</th>
      <th>cloudcover</th>
      <th>visibility</th>
      <th>solarradiation</th>
      <th>solarenergy</th>
      <th>uvindex</th>
      <th>sunrise</th>
      <th>sunset</th>
      <th>moonphase</th>
      <th>rain_dawn</th>
      <th>rain_morning</th>
      <th>rain_afternoon</th>
      <th>rain_evening</th>
      <th>E</th>
      <th>N</th>
      <th>NE</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SW</th>
      <th>W</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
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
      <td>1009.1</td>
      <td>89.0</td>
      <td>9.0</td>
      <td>174.0</td>
      <td>15.0</td>
      <td>6</td>
      <td>2016-07-01T07:08:07</td>
      <td>2016-07-01T19:26:07</td>
      <td>0.88</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2016</td>
      <td>7</td>
      <td>1</td>
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
      <td>1009.8</td>
      <td>89.0</td>
      <td>9.1</td>
      <td>193.3</td>
      <td>16.6</td>
      <td>7</td>
      <td>2016-07-02T07:08:20</td>
      <td>2016-07-02T19:26:17</td>
      <td>0.92</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2016</td>
      <td>7</td>
      <td>2</td>
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
      <td>1009.9</td>
      <td>87.9</td>
      <td>9.8</td>
      <td>196.4</td>
      <td>17.1</td>
      <td>7</td>
      <td>2016-07-03T07:08:32</td>
      <td>2016-07-03T19:26:27</td>
      <td>0.95</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2016</td>
      <td>7</td>
      <td>3</td>
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
      <td>1009.1</td>
      <td>89.0</td>
      <td>9.1</td>
      <td>221.2</td>
      <td>19.0</td>
      <td>8</td>
      <td>2016-07-04T07:08:44</td>
      <td>2016-07-04T19:26:36</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2016</td>
      <td>7</td>
      <td>4</td>
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
      <td>1008.4</td>
      <td>89.0</td>
      <td>8.4</td>
      <td>139.2</td>
      <td>11.9</td>
      <td>5</td>
      <td>2016-07-05T07:08:56</td>
      <td>2016-07-05T19:26:45</td>
      <td>0.02</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2016</td>
      <td>7</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</pre>


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
df = df.iloc[1:,31:]
df.head()
```

### Output
</details>
<pre><table border="0" class="dataframe"><tbody><tr style="text-align: right;">
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
      <th>E (lag_1)</th>
      <th>N (lag_1)</th>
      <th>NE (lag_1)</th>
      <th>NW (lag_1)</th>
      <th>S (lag_1)</th>
      <th>SE (lag_1)</th>
      <th>SW (lag_1)</th>
      <th>W (lag_1)</th>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>15.913</td></tr></tbody>
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
fig, axes = plt.subplots(nrows=8, ncols=4, figsize=(12,20))
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
fig, axes = plt.subplots(nrows=9, ncols=4, figsize=(12,25))
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
plt.figure(figsize=(15, 10))
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

<pre><table border="0" class="dataframe"><tbody><tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
      <th>dew (lag_1)</th>
      <th>humidity (lag_1)</th>
      <th>precip (lag_1)</th>
      <th>windspeed (lag_1)</th>
      <th>sealevelpressure (lag_1)</th>
      <th>cloudcover (lag_1)</th>
      <th>visibility (lag_1)</th>
      <th>solarradiation (lag_1)</th>
      <th>moonphase (lag_1)</th>
      <th>rain_dawn (lag_1)</th>
      <th>rain_morning (lag_1)</th>
      <th>rain_afternoon (lag_1)</th>
      <th>rain_evening (lag_1)</th>
      <th>E (lag_1)</th>
      <th>N (lag_1)</th>
      <th>NE (lag_1)</th>
      <th>NW (lag_1)</th>
      <th>S (lag_1)</th>
      <th>SE (lag_1)</th>
      <th>SW (lag_1)</th>
      <th>W (lag_1)</th>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <td>1008.4</td>
      <td>89.0</td>
      <td>8.4</td>
      <td>139.2</td>
      <td>0.02</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
sealevelpressure (lag_1)  float64  0 (0%)      0 (0%)
cloudcover (lag_1)        float64  0 (0%)      0 (0%)
visibility (lag_1)        float64  0 (0%)      0 (0%)
solarradiation (lag_1)    float64  0 (0%)      0 (0%)
moonphase (lag_1)         float64  0 (0%)     93 (3%)
rain_dawn (lag_1)         float64  0 (0%)  1850 (67%)
rain_morning (lag_1)      float64  0 (0%)  1805 (65%)
rain_afternoon (lag_1)    float64  0 (0%)  1821 (66%)
rain_evening (lag_1)      float64  0 (0%)  1611 (58%)
E (lag_1)                 float64  0 (0%)  2668 (97%)
N (lag_1)                 float64  0 (0%)  2623 (95%)
NE (lag_1)                float64  0 (0%)  2622 (95%)
NW (lag_1)                float64  0 (0%)  1921 (69%)
S (lag_1)                 float64  0 (0%)  2236 (81%)
SE (lag_1)                float64  0 (0%)  2405 (87%)
SW (lag_1)                float64  0 (0%)  2473 (90%)
W (lag_1)                 float64  0 (0%)  2274 (82%)
Rainfall                  float64  0 (0%)   902 (32%)
precipitation             float64  0 (0%)   902 (32%)
monsoon_month               int64  0 (0%)  1831 (66%)
</pre>

# Feature Scaling
By scaling each feature to the range (0,1) using SKlearn's **MinMaxSaler**, all numerical features will fall within the same range. This will improve the performance of distance-based algorithms and speed up the convergence of gradient descent algorithms in the next part.

<details>
<summary>View Code</summary>

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_bin=df.iloc[:,-14:]
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df.iloc[:,-14:]=df_bin
df.drop('precipitation', axis=1, inplace=True)
df.head()
```

### Output
</details>
<pre><table border="0" class="dataframe">  <thead>    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
      <th>dew (lag_1)</th>
      <th>humidity (lag_1)</th>
      <th>precip (lag_1)</th>
      <th>windspeed (lag_1)</th>
      <th>sealevelpressure (lag_1)</th>
      <th>cloudcover (lag_1)</th>
      <th>visibility (lag_1)</th>
      <th>solarradiation (lag_1)</th>
      <th>moonphase (lag_1)</th>
      <th>rain_dawn (lag_1)</th>
      <th>rain_morning (lag_1)</th>
      <th>rain_afternoon (lag_1)</th>
      <th>rain_evening (lag_1)</th>
      <th>E (lag_1)</th>
      <th>N (lag_1)</th>
      <th>NE (lag_1)</th>
      <th>NW (lag_1)</th>
      <th>S (lag_1)</th>
      <th>SE (lag_1)</th>
      <th>SW (lag_1)</th>
      <th>W (lag_1)</th>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <td>0.279070</td>
      <td>0.914616</td>
      <td>0.581818</td>
      <td>0.394986</td>
      <td>0.020408</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
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

As for the split, we opt for a 70% train, 15% validation and 15% test split, where the train data is cross-validated on 8 folds with the validation data before evaluating it using the test data.

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

## Baseline Model - Logistic Regression

<details>
<summary>View Code</summary>

```python
X=df.drop('Rainfall', axis=1)
y=df['Rainfall']

x_train, x_testval, y_train, y_testval = train_test_split(X, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_testval, y_testval, test_size=0.5, random_state=24)

# over = SMOTE(sampling_strategy=0.6, random_state=42)
# under = RandomUnderSampler(sampling_strategy=0.6, random_state=42)
# smt = Pipeline(steps=[('o', over), ('u', under)])
# x_train, y_train=smt.fit_resample(x_train, y_train)

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
Accuracy:  0.7233
Precision:  0.7530
Recall:  0.8908
F1-Score:  0.8161
array([[ 45,  83],
       [ 31, 253]])
</pre>

Although the accuracy isn't really excellent for this algorithm, the advantage of using this algorithm is we are able to find out which features play a more important role in predicting the final outcome.

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

Some of the important previous day variables are the humidity, total precipitation amount and also the previous evening rain indicator.

We will try to enhance this Logistic Regression model using only some of the more important features:

### Validation Performance

<details>
<summary>View Code</summary>

```python
X=df[['humidity (lag_1)','precip (lag_1)','rain_evening (lag_1)','monsoon_month','NW (lag_1)',#'cloudcover (lag_1)','windspeed (lag_1)','Month',
     'sealevelpressure (lag_1)','SE (lag_1)','rain_dawn (lag_1)']]
y=df['Rainfall']

x_train, x_testval, y_train, y_testval = train_test_split(X, y, test_size=0.30, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_testval, y_testval, test_size=0.5, random_state=24)

# over = SMOTE(sampling_strategy=0.5, random_state=42)
# under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
# smt = Pipeline(steps=[('o', over), ('u', under)])
# x_train, y_train=smt.fit_resample(x_train, y_train)

def objective(trial):
    model=LogisticRegression(solver='liblinear',
                                penalty='l1',
                                 C=2,
                                 max_iter=5000,
                                 #class_weight={0: 0.5,1: 0.5},
                                 random_state=42
                    )

    # Perform K-fold cross-validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    ac_scorer = make_scorer(accuracy_score)
    scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring=ac_scorer)
    return np.mean(scores)

# Create and run the Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
best_params = study.best_params

LR=LogisticRegression(**best_params)
LR.fit(x_train, y_train)
y_pred = LR.predict(x_val)
print("Accuracy: ", "%.4f" % accuracy_score(y_val, y_pred))
print("Precision: ", "%.4f" % precision_score(y_val, y_pred))
print("Recall: ", "%.4f" % recall_score(y_val, y_pred))
print("F1-Score: ", "%.4f" % f1_score(y_val, y_pred))
confusion_matrix(y_val, y_pred)
```

### Output
</details>
<pre>
Accuracy:  0.7282
Precision:  0.7606
Recall:  0.8838
F1-Score:  0.8176
array([[ 49,  79],
       [ 33, 251]])
</pre>

Fortunately, we are able to replicate the performance using much less features available. And by validating it against unseen test data,

### Test Performance

<details>
<summary>View Code</summary>

```python
y_pred = LR.predict(x_test)
y_pred_proba = LR.predict_proba(x_test)[::,1]
LR_fp, LR_tp, _ = roc_curve(y_test,  y_pred_proba)
print("Accuracy: ", "%.4f" % accuracy_score(y_test, y_pred))
print("Precision: ", "%.4f" % precision_score(y_test, y_pred))
print("Recall: ", "%.4f" % recall_score(y_test, y_pred))
print("F1-Score: ", "%.4f" % f1_score(y_test, y_pred))
confusion_matrix(y_test, y_pred)
```

### Output
</details>
<pre>
Accuracy:  0.7039
Precision:  0.7448
Recall:  0.8746
F1-Score:  0.8045
array([[ 39,  86],
       [ 36, 251]])
</pre>

## Naive Bayes
### Validation Performance

<details>
<summary>View Code</summary>

```python
X=df[['monsoon_month','Month','humidity (lag_1)','solarradiation (lag_1)', 'rain_evening (lag_1)']]#'Year','Month','Week','temp (lag_1)','cloudcover (lag_1)','windspeed (lag_1)', 
y=df['Rainfall']

x_train, x_testval, y_train, y_testval = train_test_split(X, y, test_size=0.30, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_testval, y_testval, test_size=0.5, random_state=24)

# over = SMOTE(sampling_strategy=0.6, random_state=42)
# under = RandomUnderSampler(sampling_strategy=0.7, random_state=42)
# smt = Pipeline(steps=[('o', over), ('u', under)])
# x_train, y_train=smt.fit_resample(x_train, y_train)

NB=GaussianNB()
NB.fit(x_train,y_train)
y_pred = NB.predict(x_val)
print("Accuracy: ", "%.4f" % accuracy_score(y_val, y_pred))
print("Precision: ", "%.4f" % precision_score(y_val, y_pred))
print("Recall: ", "%.4f" % recall_score(y_val, y_pred))
print("F1-Score: ", "%.4f" % f1_score(y_val, y_pred))
confusion_matrix(y_val, y_pred)
```

### Output
</details>
<pre>
Accuracy:  0.6772
Precision:  0.7807
Recall:  0.7394
F1-Score:  0.7595
array([[ 69,  59],
       [ 74, 210]])
</pre>

### Test Performance

<details>
<summary>View Code</summary>

```python
y_pred = NB.predict(x_test)
y_pred_proba = NB.predict_proba(x_test)[::,1]
NB_fp, NB_tp, _ = roc_curve(y_test,  y_pred_proba)
print("Accuracy: ", "%.4f" % accuracy_score(y_test, y_pred))
print("Precision: ", "%.4f" % precision_score(y_test, y_pred))
print("Recall: ", "%.4f" % recall_score(y_test, y_pred))
print("F1-Score: ", "%.4f" % f1_score(y_test, y_pred))
confusion_matrix(y_test, y_pred)
```

### Output
</details>
<pre>
Accuracy:  0.6796
Precision:  0.7881
Recall:  0.7387
F1-Score:  0.7626
array([[ 68,  57],
       [ 75, 212]])
</pre>

## Decision Tree
### Validation Performance

<details>
<summary>View Code</summary>

```python
X=df[['monsoon_month','humidity (lag_1)','rain_evening (lag_1)','NW (lag_1)','sealevelpressure (lag_1)']]
y=df['Rainfall']

x_train, x_testval, y_train, y_testval = train_test_split(X, y, test_size=0.30, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_testval, y_testval, test_size=0.5, random_state=42)

over = SMOTE(sampling_strategy=0.5, random_state=42)
under = RandomUnderSampler(sampling_strategy=0.6, random_state=42)
smt = Pipeline(steps=[('o', over), ('u', under)])
x_train, y_train=smt.fit_resample(x_train, y_train)

def objective(trial):
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy','log_loss'])
    splitter = trial.suggest_categorical('splitter', ['best','random'])
    max_depth = trial.suggest_int('max_depth', 2, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 30)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 20)
    max_features = trial.suggest_categorical('max_features', ['sqrt','log2'])
    #random_state = trial.suggest_int('random_state', 1, 30)

    # Create a decision tree regressor with the suggested hyperparameters
    model=DecisionTreeClassifier(criterion=criterion,
                                 splitter=splitter,
                                 max_depth=max_depth,
                                 min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf,
                                 max_features=max_features,
                                 random_state=42
                    )

    # Perform K-fold cross-validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    ac_scorer = make_scorer(accuracy_score)
    scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring=ac_scorer)
    return np.mean(scores)

# Create and run the Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)
best_params = study.best_params

DT=DecisionTreeClassifier(**best_params)
DT.fit(x_train, y_train)
y_pred = DT.predict(x_val)
print("Accuracy: ", "%.4f" % accuracy_score(y_val, y_pred))
print("Precision: ", "%.4f" % precision_score(y_val, y_pred))
print("Recall: ", "%.4f" % recall_score(y_val, y_pred))
print("F1-Score: ", "%.4f" % f1_score(y_val, y_pred))
confusion_matrix(y_val, y_pred)
```

### Output
</details>
<pre>
Accuracy:  0.6255
Precision:  0.7139
Recall:  0.7446
F1-Score:  0.7289
array([[ 67, 111],
       [ 95, 277]])
</pre>

### Test Performance

<details>
<summary>View Code</summary>

```python
DT.fit(x_train, y_train)
y_pred = DT.predict(x_test)
y_pred_proba = DT.predict_proba(x_test)[::,1]
DT_fp, DT_tp, _ = roc_curve(y_test,  y_pred_proba)
print("Accuracy: ", "%.4f" % accuracy_score(y_test, y_pred))
print("Precision: ", "%.4f" % precision_score(y_test, y_pred))
print("Recall: ", "%.4f" % recall_score(y_test, y_pred))
print("F1-Score: ", "%.4f" % f1_score(y_test, y_pred))
confusion_matrix(y_test, y_pred)
```

### Output
</details>
<pre>
Accuracy:  0.6460
Precision:  0.7485
Recall:  0.7033
F1-Score:  0.7252
array([[ 49,  43],
       [ 54, 128]])
</pre>

## K-Neighbors
### Validation Performance

<details>
<summary>View Code</summary>

```python
X=df[['humidity (lag_1)','precip (lag_1)','rain_evening (lag_1)','monsoon_month','NW (lag_1)',#'cloudcover (lag_1)','windspeed (lag_1)','Month',
      'sealevelpressure (lag_1)','SE (lag_1)','rain_dawn (lag_1)']]
y=df['Rainfall']

x_train, x_testval, y_train, y_testval = train_test_split(X, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_testval, y_testval, test_size=0.5, random_state=24)

# over = SMOTE(sampling_strategy=0.5, random_state=42)
# under = RandomUnderSampler(sampling_strategy=0.6, random_state=42)
# smt = Pipeline(steps=[('o', over), ('u', under)])
# x_train, y_train=smt.fit_resample(x_train, y_train)


def objective(trial):
    n_neighbors = trial.suggest_int('n_neighbors', 1, 20)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
    p = trial.suggest_int('p', 1, 2)
    leaf_size = trial.suggest_int('leaf_size', 1, 50)

    # Create a decision tree regressor with the suggested hyperparameters
    model = KNeighborsClassifier(n_neighbors=n_neighbors, 
                                weights=weights, 
                                algorithm=algorithm,
                                p=p,
                                leaf_size=leaf_size
                               )

    # Perform K-fold cross-validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    ac_scorer = make_scorer(accuracy_score)
    scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring=ac_scorer)
    return np.mean(scores)

# Create and run the Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
best_params = study.best_params

KN=KNeighborsClassifier(**best_params)
KN.fit(x_train, y_train)
y_pred = KN.predict(x_val)
print("Accuracy: ", "%.4f" % accuracy_score(y_val, y_pred))
print("Precision: ", "%.4f" % precision_score(y_val, y_pred))
print("Recall: ", "%.4f" % recall_score(y_val, y_pred))
print("F1-Score: ", "%.4f" % f1_score(y_val, y_pred))
confusion_matrix(y_val, y_pred)
```

### Output
</details>
<pre>
Accuracy:  0.6869
Precision:  0.7370
Recall:  0.8486
F1-Score:  0.7889
array([[ 42,  86],
       [ 43, 241]])
</pre>

### Test Performance

<details>
<summary>View Code</summary>

```python
y_pred = KN.predict(x_test)
y_pred_proba = KN.predict_proba(x_test)[::,1]
KN_fp, KN_tp, _ = roc_curve(y_test,  y_pred_proba)
print("Accuracy: ", "%.4f" % accuracy_score(y_test, y_pred))
print("Precision: ", "%.4f" % precision_score(y_test, y_pred))
print("Recall: ", "%.4f" % recall_score(y_test, y_pred))
print("F1-Score: ", "%.4f" % f1_score(y_test, y_pred))
confusion_matrix(y_test, y_pred)
```

### Output
</details>
<pre>
Accuracy:  0.6626
Precision:  0.7312
Recall:  0.8153
F1-Score:  0.7710
array([[ 39,  86],
       [ 53, 234]])
</pre>

## Random Forest
### Validation Performance

<details>
<summary>View Code</summary>

```python
X=df[['Year','Month','Day','monsoon_month','moonphase (lag_1)','solarradiation (lag_1)','cloudcover (lag_1)','windspeed (lag_1)','visibility (lag_1)','humidity (lag_1)','precip (lag_1)','rain_dawn (lag_1)','rain_morning (lag_1)','rain_afternoon (lag_1)','rain_evening (lag_1)']]
y=df['Rainfall']

x_train, x_testval, y_train, y_testval = train_test_split(X, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_testval, y_testval, test_size=0.5, random_state=24)

# over = SMOTE(sampling_strategy=0.5, random_state=42)
# under = RandomUnderSampler(sampling_strategy=0.6, random_state=42)
# smt = Pipeline(steps=[('o', over), ('u', under)])
# x_train, y_train=smt.fit_resample(x_train, y_train)


def objective(trial):
    # Define the hyperparameters to optimize
    criterion = trial.suggest_categorical('criterion', ['gini','entropy','log_loss'])#, 'poisson'
    max_depth = trial.suggest_int('max_depth', 2, 15)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 20)
    max_features = trial.suggest_categorical('max_features', ['sqrt','log2'])
    #random_state = trial.suggest_int('random_state', 1, 30)
    model = RandomForestClassifier(criterion=criterion, 
                                  max_depth=max_depth, 
                                  min_samples_split=min_samples_split, 
                                  min_samples_leaf=min_samples_leaf, 
                                  max_features=max_features,
                                  random_state=42
                                 )

    # Perform K-fold cross-validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    ac_scorer = make_scorer(accuracy_score)
    scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring=ac_scorer)
    return np.mean(scores)

# Create and run the Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)
best_params = study.best_params

RF=RandomForestClassifier(**best_params)
RF.fit(x_train, y_train)
y_pred = RF.predict(x_val)
print("Accuracy: ", "%.4f" % accuracy_score(y_val, y_pred))
print("Precision: ", "%.4f" % precision_score(y_val, y_pred))
print("Recall: ", "%.4f" % recall_score(y_val, y_pred))
print("F1-Score: ", "%.4f" % f1_score(y_val, y_pred))
confusion_matrix(y_val, y_pred)
```

### Output
</details>
<pre>
Accuracy:  0.7184
Precision:  0.7471
Recall:  0.8944
F1-Score:  0.8141
array([[ 42,  86],
       [ 30, 254]])
</pre>

### Test Performance

<details>
<summary>View Code</summary>

```python
y_pred = RF.predict(x_test)
y_pred_proba = RF.predict_proba(x_test)[::,1]
RF_fp, RF_tp, _ = roc_curve(y_test,  y_pred_proba)
print("Accuracy: ", "%.4f" % accuracy_score(y_test, y_pred))
print("Precision: ", "%.4f" % precision_score(y_test, y_pred))
print("Recall: ", "%.4f" % recall_score(y_test, y_pred))
print("F1-Score: ", "%.4f" % f1_score(y_test, y_pred))
confusion_matrix(y_test, y_pred)
```

### Output
</details>
<pre>
Accuracy:  0.7257
Precision:  0.7500
Recall:  0.9094
F1-Score:  0.8220
array([[ 38,  87],
       [ 26, 261]])
</pre>

## MLP Classifier
### Validation Performance

<details>
<summary>View Code</summary>

```python
from sklearn.neural_network import MLPClassifier

X=df[['Month','humidity (lag_1)','precip (lag_1)','rain_evening (lag_1)','monsoon_month','NW (lag_1)',#'cloudcover (lag_1)','windspeed (lag_1)','Month',
     'sealevelpressure (lag_1)','SE (lag_1)','rain_dawn (lag_1)']]
y=df['Rainfall']

x_train, x_testval, y_train, y_testval = train_test_split(X, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_testval, y_testval, test_size=0.5, random_state=24)

# over = SMOTE(sampling_strategy=0.5, random_state=42)
# under = RandomUnderSampler(sampling_strategy=0.6, random_state=42)
# smt = Pipeline(steps=[('o', over), ('u', under)])
# x_train, y_train=smt.fit_resample(x_train, y_train)


MLP = MLPClassifier(hidden_layer_sizes=(),activation='logistic', solver='lbfgs')
MLP.fit(x_train, y_train)
y_pred = MLP.predict(x_val)
y_pred_proba = MLP.predict_proba(x_val)[::,1]
MLP_fp, MLP_tp, _ = roc_curve(y_val,  y_pred_proba)
print("Accuracy: ", "%.4f" % accuracy_score(y_val, y_pred))
print("Precision: ", "%.4f" % precision_score(y_val, y_pred))
print("Recall: ", "%.4f" % recall_score(y_val, y_pred))
print("F1-Score: ", "%.4f" % f1_score(y_val, y_pred))
confusion_matrix(y_val, y_pred)
```

### Output
</details>
<pre>
Accuracy:  0.7209
Precision:  0.7568
Recall:  0.8768
F1-Score:  0.8124
array([[ 48,  80],
       [ 35, 249]])
</pre>

### Test Performance

<details>
<summary>View Code</summary>

```python
y_pred = MLP.predict(x_test)
y_pred_proba = MLP.predict_proba(x_test)[::,1]
MLP_fp, MLP_tp, _ = roc_curve(y_test,  y_pred_proba)
print("Accuracy: ", "%.4f" % accuracy_score(y_test, y_pred))
print("Precision: ", "%.4f" % precision_score(y_test, y_pred))
print("Recall: ", "%.4f" % recall_score(y_test, y_pred))
print("F1-Score: ", "%.4f" % f1_score(y_test, y_pred))
confusion_matrix(y_test, y_pred)
```

### Output
</details>
<pre>
Accuracy:  0.7039
Precision:  0.7448
Recall:  0.8746
F1-Score:  0.8045
array([[ 39,  86],
       [ 36, 251]])
</pre>

## Sequential NN
### Validation Performance

<details>
<summary>View Code</summary>

```python
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import BinaryCrossentropy

X=df[['Year','Month','Day','monsoon_month','moonphase (lag_1)','solarradiation (lag_1)','cloudcover (lag_1)','windspeed (lag_1)','visibility (lag_1)','humidity (lag_1)','precip (lag_1)','rain_dawn (lag_1)','rain_morning (lag_1)','rain_afternoon (lag_1)','rain_evening (lag_1)']]
y=df['Rainfall']

x_train, x_testval, y_train, y_testval = train_test_split(X, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_testval, y_testval, test_size=0.5, random_state=24)

# Creating model using the Sequential in tensorflow
NN = Sequential()
NN.add(Dense(512, activation='relu', input_dim=15))
NN.add(Dropout(0.2))
NN.add(Dense(1, activation='sigmoid'))
NN.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
hist = NN.fit(x_train, y_train, epochs=50, batch_size=10, validation_split=0.2)
NN.summary()
```

    
### Output
</details>

<details>
<summary>View Epochs</summary>
<pre>
Epoch 1/50
154/154 [==============================] - 1s 3ms/step - loss: 0.6305 - accuracy: 0.6636 - val_loss: 0.6245 - val_accuracy: 0.6519
Epoch 2/50
154/154 [==============================] - 0s 2ms/step - loss: 0.6101 - accuracy: 0.6649 - val_loss: 0.6188 - val_accuracy: 0.6519
Epoch 3/50
154/154 [==============================] - 0s 2ms/step - loss: 0.6055 - accuracy: 0.6649 - val_loss: 0.6152 - val_accuracy: 0.6519
Epoch 4/50
154/154 [==============================] - 0s 2ms/step - loss: 0.6017 - accuracy: 0.6649 - val_loss: 0.6127 - val_accuracy: 0.6519
Epoch 5/50
154/154 [==============================] - 0s 2ms/step - loss: 0.6010 - accuracy: 0.6649 - val_loss: 0.6106 - val_accuracy: 0.6519
Epoch 6/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5971 - accuracy: 0.6662 - val_loss: 0.6091 - val_accuracy: 0.6571
Epoch 7/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5960 - accuracy: 0.6695 - val_loss: 0.6081 - val_accuracy: 0.6519
Epoch 8/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5934 - accuracy: 0.6682 - val_loss: 0.6077 - val_accuracy: 0.6519
Epoch 9/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5948 - accuracy: 0.6721 - val_loss: 0.6070 - val_accuracy: 0.6545
Epoch 10/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5939 - accuracy: 0.6721 - val_loss: 0.6058 - val_accuracy: 0.6701
Epoch 11/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5922 - accuracy: 0.6708 - val_loss: 0.6052 - val_accuracy: 0.6675
Epoch 12/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5912 - accuracy: 0.6747 - val_loss: 0.6052 - val_accuracy: 0.6727
Epoch 13/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5894 - accuracy: 0.6773 - val_loss: 0.6042 - val_accuracy: 0.6675
Epoch 14/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5898 - accuracy: 0.6825 - val_loss: 0.6042 - val_accuracy: 0.6753
Epoch 15/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5917 - accuracy: 0.6786 - val_loss: 0.6033 - val_accuracy: 0.6571
Epoch 16/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5898 - accuracy: 0.6734 - val_loss: 0.6033 - val_accuracy: 0.6727
Epoch 17/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5875 - accuracy: 0.6845 - val_loss: 0.6029 - val_accuracy: 0.6701
Epoch 18/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5897 - accuracy: 0.6858 - val_loss: 0.6027 - val_accuracy: 0.6519
Epoch 19/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5872 - accuracy: 0.6799 - val_loss: 0.6028 - val_accuracy: 0.6519
Epoch 20/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5907 - accuracy: 0.6760 - val_loss: 0.6024 - val_accuracy: 0.6519
Epoch 21/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5873 - accuracy: 0.6851 - val_loss: 0.6020 - val_accuracy: 0.6623
Epoch 22/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5854 - accuracy: 0.6903 - val_loss: 0.6019 - val_accuracy: 0.6883
Epoch 23/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5859 - accuracy: 0.6890 - val_loss: 0.6017 - val_accuracy: 0.6597
Epoch 24/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5856 - accuracy: 0.6845 - val_loss: 0.6018 - val_accuracy: 0.6519
Epoch 25/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5870 - accuracy: 0.6877 - val_loss: 0.6021 - val_accuracy: 0.6831
Epoch 26/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5860 - accuracy: 0.6890 - val_loss: 0.6012 - val_accuracy: 0.6597
Epoch 27/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5863 - accuracy: 0.6916 - val_loss: 0.6009 - val_accuracy: 0.6727
Epoch 28/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5870 - accuracy: 0.6897 - val_loss: 0.6008 - val_accuracy: 0.6857
Epoch 29/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5854 - accuracy: 0.6871 - val_loss: 0.6006 - val_accuracy: 0.6831
Epoch 30/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5858 - accuracy: 0.6897 - val_loss: 0.6007 - val_accuracy: 0.6701
Epoch 31/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5863 - accuracy: 0.6747 - val_loss: 0.6004 - val_accuracy: 0.6857
Epoch 32/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5845 - accuracy: 0.6936 - val_loss: 0.6003 - val_accuracy: 0.6831
Epoch 33/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5831 - accuracy: 0.6825 - val_loss: 0.6003 - val_accuracy: 0.6857
Epoch 34/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5854 - accuracy: 0.6929 - val_loss: 0.6001 - val_accuracy: 0.6831
Epoch 35/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5831 - accuracy: 0.6897 - val_loss: 0.6004 - val_accuracy: 0.6883
Epoch 36/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5828 - accuracy: 0.6968 - val_loss: 0.6004 - val_accuracy: 0.6883
Epoch 37/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5820 - accuracy: 0.6871 - val_loss: 0.5997 - val_accuracy: 0.6857
Epoch 38/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5824 - accuracy: 0.6884 - val_loss: 0.5996 - val_accuracy: 0.6883
Epoch 39/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5834 - accuracy: 0.6916 - val_loss: 0.5996 - val_accuracy: 0.6909
Epoch 40/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5830 - accuracy: 0.6936 - val_loss: 0.5993 - val_accuracy: 0.6935
Epoch 41/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5811 - accuracy: 0.6929 - val_loss: 0.5992 - val_accuracy: 0.6935
Epoch 42/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5828 - accuracy: 0.6955 - val_loss: 0.5991 - val_accuracy: 0.6961
Epoch 43/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5821 - accuracy: 0.6910 - val_loss: 0.5995 - val_accuracy: 0.6701
Epoch 44/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5814 - accuracy: 0.6916 - val_loss: 0.5989 - val_accuracy: 0.6935
Epoch 45/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5817 - accuracy: 0.6877 - val_loss: 0.5989 - val_accuracy: 0.6987
Epoch 46/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5784 - accuracy: 0.6988 - val_loss: 0.5988 - val_accuracy: 0.6961
Epoch 47/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5798 - accuracy: 0.6923 - val_loss: 0.5990 - val_accuracy: 0.6857
Epoch 48/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5795 - accuracy: 0.6994 - val_loss: 0.5986 - val_accuracy: 0.6961
Epoch 49/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5798 - accuracy: 0.6962 - val_loss: 0.5987 - val_accuracy: 0.6883
Epoch 50/50
154/154 [==============================] - 0s 2ms/step - loss: 0.5793 - accuracy: 0.6994 - val_loss: 0.5984 - val_accuracy: 0.6961
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 512)               8192      
                                                                 
 dropout (Dropout)           (None, 512)               0         
                                                                 
 dense_1 (Dense)             (None, 1)                 513       
                                                                 
=================================================================
Total params: 8705 (34.00 KB)
Trainable params: 8705 (34.00 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
</pre>
</details>

### Test Performance

```python
y_pred = NN.predict(x_test) >0.5
NN_fp, NN_tp, _ = roc_curve(y_test,  y_pred_proba)
print("Accuracy: ", "%.4f" % accuracy_score(y_test, y_pred))
print("Precision: ", "%.4f" % precision_score(y_test, y_pred))
print("Recall: ", "%.4f" % recall_score(y_test, y_pred))
print("F1-Score: ", "%.4f" % f1_score(y_test, y_pred))
confusion_matrix(y_test, y_pred)
```

### Output
</details>
<pre>
13/13 [==============================] - 0s 1ms/step
Accuracy:  0.7306
Precision:  0.7500
Recall:  0.9199
F1-Score:  0.8263
array([[ 37,  88],
       [ 23, 264]])
</pre>

# Performance Comparison

<details>
<summary>View Code</summary>

```python
plt.plot(LR_fp,LR_tp, color='blue', label='Logistic Regression')
plt.plot(NB_fp,NB_tp, color='red', label='Naive Bayes')
plt.plot(DT_fp,DT_tp, color='cyan', label='Decision Tree')
plt.plot(KN_fp,KN_tp, color='green', label='KNN')
#plt.plot(SV_fp,SV_tp, color='grey', Support Vector)
plt.plot(RF_fp,RF_tp, color='pink', label='Random Forest')
plt.plot(MLP_fp,MLP_tp, color='orange', label='MLP')
plt.plot(NN_fp,NN_tp, color='magenta', label='Sequential NN')
plt.plot([0,1],[0,1],'--', color='black')
plt.title('ROC Curve Comparison')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()
```

### Output
</details>

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/kuala-lumpur-rain-prediction/images/9.png?raw=true)

Besides KNN and Decision Trees which are performing poorly than the rest, the performance of other models are hardly distinguishable, the Logistic Regression and Naive Bayes algorithms were even performing on par with complex models such as the Random Forest and Sequential NN.

However, if we were to take a more prudent approach on predicting rainfall (predicting more false positives than false negatives), the model with the highest precision would be preferred, in this case it would be the **MLP Classifier** with **76%** Precision. 
