<h1 align="center">Data Visualization: Uber Fares Analysis</h1>

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/uber-fares-analysis/images/uber.jpg?raw=true)

In 2023, *Uber* has a reported active drivers reaching 1.5 million in the United States alone, occupying up to 74% of the global e-hailing market.

A core part of *Uber*'s business is dynamic pricing - also known as surge pricing - where fares fluctuate based on supply and demand principles. When demand for rides is high, such as during rush hour or special events when the demand for rides are high, fares increase automatically to entice more drivers to get on the road.

As *Uber* has grown exponentially in recent years to become the dominant e-hailing service globally, analysis on its pricing model and decisions has become increasingly important. This webpage will take a data-driven look at how *Uber*'s fares are set across different situations. The goal is to provide transparency and insight into a key aspect of a company that has disrupted transportation enormously in a short span of time.

## THE PROJECT
The dataset used here is the [Uber Fares Dataset](https://www.kaggle.com/datasets/yasserh/uber-fares-dataset) by *M YASSER H* which can be found on Kaggle, it contains data of 500k trips which are mainly concentrated in the suburbs of New York City, a pretty big one I would say!

To begin, we will import the following libraries to our kernel:
* **Pandas** - A staple for reading and manipulating structured data.
* **Numpy** - For array-based operations.
* **Matplotlib** and **Seaborn** - Used together to produce stunning visuals.
* **Datetime** - Used to parse datetimes into their respective components such as days, hours, and minutes.
* **Folium** - An interactive map to visualize pickup and dropoff locations with provided coordinates
* **Nominatim** by Geopy - Used to calculate the Haversine distance between 2 coordinates

<details>
<summary>View Code</summary>
	
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import folium
from folium.plugins import MarkerCluster, MousePosition
from folium.features import DivIcon
from geopy.geocoders import Nominatim
```
</details>

I have also defined a custom function to check on empty and zero values within the dataset as follow, it returns the number and % of NaNs and zeros for each column in the dataframe:

<details>
<summary>View Code</summary>

```python
def summary(dtf):
  sumary=pd.concat([dtf.isna().sum(),((dtf == 0.sum())/dtf.shape[0],dtf.dtypes], axis=1)
  sumary=sumary.rename(columns={sumary.columns[0]: 'NaN'})
  sumary=sumary.rename(columns={sumary.columns[1]: 'Zeros'})
  sumary=sumary.rename(columns={sumary.columns[2]: 'Type'})
  sumary['NaN']=sumary['NaN'].astype(str)+' ('+((sumary['NaN']*100/dtf.shape[0]).astype(int)).astype(str)+'%)'
  sumary['Zeros']=(sumary['Zeros']*100).astype(int)
  sumary['Zeros']=(dtf == 0).sum().astype(str)+' ('+sumary['Zeros'].astype(str)+'%)'
  sumary=sumary[['Type','NaN','Zeros']]
  return
```
</details>

This is how the data looks in the first place.

```python
df = pd.read_csv("/kaggle/input/uber-fares-dataset/uber.csv")
print(summary(df))
df.head()
```
### Output
<pre>
                      Type     NaN      Zeros
Unnamed: 0           int64  0 (0%)     0 (0%)
key                 object  0 (0%)     0 (0%)
fare_amount        float64  0 (0%)     5 (0%)
pickup_datetime     object  0 (0%)     0 (0%)
pickup_longitude   float64  0 (0%)  3786 (1%)
pickup_latitude    float64  0 (0%)  3782 (1%)
dropoff_longitude  float64  1 (0%)  3764 (1%)
dropoff_latitude   float64  1 (0%)  3758 (1%)
passenger_count      int64  0 (0%)   709 (0%)

<table border="0" class="dataframe"><tbody>
    <tr style="text-align: right;">
      <th></th>
      <th>trip</th>
      <th>fare_amount</th>
      <th>pickup_datetime</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>passenger_count</th>
    </tr>
    <tr>
      <th>0</th>
      <td>24238194</td>
      <td>7.5</td>
      <td>2015-05-07 19:52:06 UTC</td>
      <td>-73.999817</td>
      <td>40.738354</td>
      <td>-73.999512</td>
      <td>40.723217</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27835199</td>
      <td>7.7</td>
      <td>2009-07-17 20:04:56 UTC</td>
      <td>-73.994355</td>
      <td>40.728225</td>
      <td>-73.994710</td>
      <td>40.750325</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>44984355</td>
      <td>12.9</td>
      <td>2009-08-24 21:45:00 UTC</td>
      <td>-74.005043</td>
      <td>40.740770</td>
      <td>-73.962565</td>
      <td>40.772647</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25894730</td>
      <td>5.3</td>
      <td>2009-06-26 08:22:21 UTC</td>
      <td>-73.976124</td>
      <td>40.790844</td>
      <td>-73.965316</td>
      <td>40.803349</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17610152</td>
      <td>16.0</td>
      <td>2014-08-28 17:47:00 UTC</td>
      <td>-73.925023</td>
      <td>40.744085</td>
      <td>-73.973082</td>
      <td>40.761247</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</pre>

Looks like we have at least 3k entries having nil latitudes and longitudes which does not seem to be valid here. Though it is generally not advisable to drop empty rows, this case is an exception as the empty entries only make up 1% of the dataset, we shouldn't waste time figuring how to impute these empty values as they could introduce biases to the data.

We will do the same approach for zero values in fare_amount and the NaNs in dropoff_longitude and dropoff_latitude.

<details>
<summary>View Code</summary>
	
```python
df=df[df['pickup_longitude']!=0]
df=df[df['pickup_latitude']!=0]
df=df[df['dropoff_longitude']!=0]
df=df[df['dropoff_latitude']!=0]
df=df[df['passenger_count']!=0]
df=df[df['fare_amount']!=0]
df.reset_index(inplace=True)
df.drop(['index','key'], axis=1, inplace=True)
df.rename(columns={"Unnamed: 0": "trip"}, inplace=True)

summary(df)
```
</details>
	
### Output
<pre>
                      Type     NaN      Zeros
trip	             int64  0 (0%)     0 (0%)
fare_amount        float64  0 (0%)     0 (0%)
pickup_datetime     object  0 (0%)     0 (0%)
pickup_longitude   float64  0 (0%)     0 (0%)
pickup_latitude    float64  0 (0%)     0 (0%)
dropoff_longitude  float64  0 (0%)     0 (0%)
dropoff_latitude   float64  0 (0%)     0 (0%)
passenger_count      int64  0 (0%)     0 (0%)
</pre>

## Mapping of Pickup and Dropoff Locations

<details>
<summary>View Code</summary>
	
```python
mymap = folium.Map(location=[40.6970193,-74.3093268], zoom_start=4)
marker_cluster = MarkerCluster(name='Pickups')
marker_cluster2 = MarkerCluster(name='Dropoffs')
marker_cluster.add_to(mymap)
marker_cluster2.add_to(mymap)
for index, row in df1.sample(n=5000, random_state=123).iterrows():
  lb1, lb2 = row['trip'], row['fare_amount']
  folium.Marker(location=[row['pickup_latitude'], row['pickup_longitude']], icon=folium.Icon(color='blue'), popup=row['trip']).add_to(marker_cluster)
  folium.Marker(location=[row['dropoff_latitude'], row['dropoff_longitude']], icon=folium.Icon(color='red'), popup=row['trip']).add_to(marker_cluster2)
  folium.LayerControl(collapsed=False).add_to(mymap)
												
formatter = "function(num) {return L.Util.formatNum(num, 5);};"
mouse_position = MousePosition(
  position='topright',
  separator=' Long: ',
  empty_string='NaN',
  lng_first=False,
  num_digits=20,
  prefix='Lat:',
  lat_formatter=formatter,
  lng_formatter=formatter,
  )
												
mymap.add_child(mouse_position)
display(mymap)
```
</details>

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/uber-fares-analysis/images/3.jpg?raw=true)

By plotting the coordinates on a map, we know for sure that this driver is based in New York (Manhattan to be precise). Unfortunately, this map also exposes many other problematic coordinates provided to us, there are a few coordinates which are in the sea, Europe, Africa, and even Antartica!

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/uber-fares-analysis/images/5.jpg?raw=true)

The zero entries removed earlier were just the tip of an iceberg, we need to screen through the coordinates and eliminate any anomalies such as:

### 1. Invalid coordinates
latitudes and longitudes can only take on the following range of values, any value which falls outside these ranges are 100% invalid:
* Latitudes: -90 to 90
* Longitudes: -180 to 180
```python
df[(abs(df['pickup_latitude'])>90)|(abs(df['dropoff_latitude'])>90)|(abs(df['dropoff_longitude'])>180)|(abs(df['pickup_longitude'])>180)].head(10)
```
### Output
<pre>
<table border="0" class="dataframe"><tbody>
    <tr style="text-align: right;">
      <th></th>
      <th>trip</th>
      <th>fare_amount</th>
      <th>pickup_datetime</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>passenger_count</th>
    </tr>
    <tr>
      <th>4852</th>
      <td>42931506</td>
      <td>4.9</td>
      <td>2012-04-28 00:58:00 UTC</td>
      <td>-748.016667</td>
      <td>40.739957</td>
      <td>-74.003570</td>
      <td>40.734192</td>
      <td>1</td>
    </tr>
    <tr>
      <th>31806</th>
      <td>5634081</td>
      <td>15.7</td>
      <td>2012-06-16 10:04:00 UTC</td>
      <td>-74.016055</td>
      <td>40.715155</td>
      <td>-737.916665</td>
      <td>40.697862</td>
      <td>2</td>
    </tr>
    <tr>
      <th>47421</th>
      <td>1055960</td>
      <td>33.7</td>
      <td>2011-11-05 23:26:00 UTC</td>
      <td>-735.200000</td>
      <td>40.770092</td>
      <td>-73.980187</td>
      <td>40.765530</td>
      <td>1</td>
    </tr>
    <tr>
      <th>55313</th>
      <td>14257861</td>
      <td>8.1</td>
      <td>2012-03-11 07:24:00 UTC</td>
      <td>-73.960828</td>
      <td>404.433332</td>
      <td>-73.988357</td>
      <td>40.769037</td>
      <td>1</td>
    </tr>
    <tr>
      <th>60372</th>
      <td>2849369</td>
      <td>8.5</td>
      <td>2012-06-13 05:45:00 UTC</td>
      <td>-73.951385</td>
      <td>401.066667</td>
      <td>-73.982110</td>
      <td>40.754117</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</pre>

### 2. Swapped Latitude and Longitude Values 
Some rides had their latitude and longitude pairs reversed by plotting the coordinates, this is where the Antartica coordinates came from:
```python
df[(df['pickup_latitude']<-70) | (df['dropoff_latitude']<-70)].head()
```		
### Output
<pre>
<table border="0" class="dataframe"><tbody>
    <tr style="text-align: right;">
      <th></th>
      <th>trip</th>
      <th>fare_amount</th>
      <th>pickup_datetime</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>passenger_count</th>
    </tr>
    <tr>
      <th>2441</th>
      <td>6452891</td>
      <td>6.0</td>
      <td>2013-05-22 10:54:00 UTC</td>
      <td>40.746760</td>
      <td>-73.982127</td>
      <td>40.757287</td>
      <td>-73.974800</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4287</th>
      <td>44648183</td>
      <td>12.5</td>
      <td>2013-05-24 00:43:00 UTC</td>
      <td>40.751797</td>
      <td>-73.970777</td>
      <td>40.719787</td>
      <td>-73.992137</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4335</th>
      <td>3826665</td>
      <td>9.5</td>
      <td>2013-05-25 01:16:00 UTC</td>
      <td>40.732897</td>
      <td>-73.997740</td>
      <td>40.747532</td>
      <td>-73.972540</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4653</th>
      <td>52635142</td>
      <td>6.5</td>
      <td>2013-05-22 12:38:00 UTC</td>
      <td>40.770667</td>
      <td>-73.961957</td>
      <td>40.761672</td>
      <td>-73.967237</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7219</th>
      <td>45851743</td>
      <td>10.5</td>
      <td>2013-05-23 00:29:00 UTC</td>
      <td>40.714897</td>
      <td>-74.009697</td>
      <td>40.726197</td>
      <td>-73.994370</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</pre>

### 3. Intercontinential Travel
There are also trips which spans across continents, and amazingly they all took less than $30, what a bang for a buck!								
```python
df[abs(df['pickup_longitude']-df['dropoff_longitude'])>40].head()
```
### Output
<pre>
<table border="0" class="dataframe"><tbody>
    <tr style="text-align: right;">
      <th></th>
      <th>trip</th>
      <th>fare_amount</th>
      <th>pickup_datetime</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>passenger_count</th>
    </tr>
    <tr>
      <th>1900</th>
      <td>915515</td>
      <td>7.0</td>
      <td>2013-02-10 16:18:00 UTC</td>
      <td>-0.131667</td>
      <td>40.757063</td>
      <td>-73.991593</td>
      <td>40.749953</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2118</th>
      <td>7976070</td>
      <td>5.7</td>
      <td>2012-07-21 12:16:00 UTC</td>
      <td>-1.216667</td>
      <td>40.748597</td>
      <td>-74.004822</td>
      <td>40.734670</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4852</th>
      <td>42931506</td>
      <td>4.9</td>
      <td>2012-04-28 00:58:00 UTC</td>
      <td>-748.016667</td>
      <td>40.739957</td>
      <td>-74.003570</td>
      <td>40.734192</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6468</th>
      <td>32571365</td>
      <td>9.3</td>
      <td>2012-06-05 19:05:00 UTC</td>
      <td>-1.866667</td>
      <td>40.765987</td>
      <td>-73.972280</td>
      <td>40.793807</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11631</th>
      <td>16955954</td>
      <td>14.9</td>
      <td>2012-02-26 00:01:00 UTC</td>
      <td>-0.007712</td>
      <td>40.725602</td>
      <td>-73.967487</td>
      <td>40.766410</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</pre>
	 
### 4. Trips in the Ocean
Obviously we can't have coordinates in the ocean:

```python
df[(abs(df['pickup_longitude'])<1) | (abs(df['dropoff_longitude'])<1)].head()
```
### Output
<pre>
<table border="0" class="dataframe"><tbody>
    <tr style="text-align: right;">
      <th></th>
      <th>trip</th>
      <th>fare_amount</th>
      <th>pickup_datetime</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>passenger_count</th>
    </tr>
    <tr>
      <th>163</th>
      <td>17269533</td>
      <td>14.5</td>
      <td>2010-07-14 17:38:00 UTC</td>
      <td>0.001782</td>
      <td>0.007380</td>
      <td>0.000875</td>
      <td>0.005670</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1900</th>
      <td>915515</td>
      <td>7.0</td>
      <td>2013-02-10 16:18:00 UTC</td>
      <td>-0.131667</td>
      <td>40.757063</td>
      <td>-73.991593</td>
      <td>40.749953</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11631</th>
      <td>16955954</td>
      <td>14.9</td>
      <td>2012-02-26 00:01:00 UTC</td>
      <td>-0.007712</td>
      <td>40.725602</td>
      <td>-73.967487</td>
      <td>40.766410</td>
      <td>5</td>
    </tr>
    <tr>
      <th>22602</th>
      <td>10365124</td>
      <td>10.9</td>
      <td>2010-10-09 22:53:00 UTC</td>
      <td>-0.076468</td>
      <td>0.087237</td>
      <td>-0.079742</td>
      <td>0.097257</td>
      <td>1</td>
    </tr>
    <tr>
      <th>35835</th>
      <td>24351230</td>
      <td>11.7</td>
      <td>2010-06-12 11:55:00 UTC</td>
      <td>0.013518</td>
      <td>0.001857</td>
      <td>0.010920</td>
      <td>0.010308</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</pre>
  
#### 5. Identical Dropoff & Pickup Locations
Seriously, why am I paying if your car didn't even budge? And surprisingly there were 1,316 of such trips!
```python
df[(df['pickup_latitude'] == df['dropoff_latitude']) & (df['dropoff_longitude'] == df['pickup_longitude'])].head()
```
### Output
<pre>
<table border="0" class="dataframe"><tbody>
    <tr style="text-align: right;">
      <th></th>
      <th>trip</th>
      <th>fare_amount</th>
      <th>pickup_datetime</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>passenger_count</th>
    </tr>
    <tr>
      <th>5</th>
      <td>44470845</td>
      <td>4.90</td>
      <td>2011-02-12 02:27:09 UTC</td>
      <td>-73.969019</td>
      <td>40.755910</td>
      <td>-73.969019</td>
      <td>40.755910</td>
      <td>1</td>
    </tr>
    <tr>
      <th>46</th>
      <td>22405517</td>
      <td>56.80</td>
      <td>2013-01-03 22:24:41 UTC</td>
      <td>-73.993498</td>
      <td>40.764686</td>
      <td>-73.993498</td>
      <td>40.764686</td>
      <td>1</td>
    </tr>
    <tr>
      <th>81</th>
      <td>25485719</td>
      <td>49.57</td>
      <td>2009-08-07 10:43:07 UTC</td>
      <td>-73.975058</td>
      <td>40.788820</td>
      <td>-73.975058</td>
      <td>40.788820</td>
      <td>1</td>
    </tr>
    <tr>
      <th>160</th>
      <td>54642873</td>
      <td>4.50</td>
      <td>2014-01-22 21:01:18 UTC</td>
      <td>-73.992937</td>
      <td>40.757780</td>
      <td>-73.992937</td>
      <td>40.757780</td>
      <td>1</td>
    </tr>
    <tr>
      <th>351</th>
      <td>11876316</td>
      <td>10.10</td>
      <td>2009-08-24 17:25:00 UTC</td>
      <td>-73.928705</td>
      <td>40.753475</td>
      <td>-73.928705</td>
      <td>40.753475</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</pre>

## Date & Time Feature Extraction
The pickup date & time could tell us a lot more on the fare evolution such as the following:
* Year-to-Year comparison to observe if there is a price hike due to inflation
* Day of week comparison to visualize if there are significant difference in fares on weekdays and weekends.
* Hourly comparison to see if there are any significant price hikes during peak hours.

```python
df['pickup_datetime'].head()
```
### Output
<pre>
0    2015-05-07 19:52:06 UTC
1    2009-07-17 20:04:56 UTC
2    2009-08-24 21:45:00 UTC
3    2009-06-26 08:22:21 UTC
4    2014-08-28 17:47:00 UTC
Name: pickup_datetime, dtype: object
</pre>
However, looking back at the pickup_datetimes, there is another step required before we parse the datetimes into their respective components.
Notice that the data type is object instead of a datetime format, and the times are in the UTC timezone instead of the Eastern Time used by New York (UTC-4/-5).

we can use the `pd.to_datetime()` and `dt.tz_convert()` to convert it into a datetime data type with its timezone adjusted to reflect Eastern Time (with Daylight Savings taken into account).

<details>
<summary>View Code</summary>
	
```python
df.reset_index(inplace=True)
df.drop(['index'], axis=1, inplace=True)
df['pickup_datetime']=pd.to_datetime(df['pickup_datetime'])
df['pickup_datetime']=df['pickup_datetime'].dt.tz_convert('America/New_York')
df['Date'] = df['pickup_datetime'].dt.date
df['Time'] = df['pickup_datetime'].dt.strftime('%H:%M')
df['Year'] = df['pickup_datetime'].dt.year
df['Month'] = df['pickup_datetime'].dt.month
df['Day'] = df['pickup_datetime'].dt.day
df['WeekDay'] = df['pickup_datetime'].dt.weekday
df['Hour'] = df['pickup_datetime'].dt.hour
df.head()
```
</details>

### Output
<pre>
<table border="0" class="dataframe"><tbody>
    <tr style="text-align: right;">
      <th></th>
      <th>trip</th>
      <th>fare_amount</th>
      <th>pickup_datetime</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>passenger_count</th>
      <th>Date</th>
      <th>Time</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
      <th>WeekDay</th>
      <th>Hour</th>
    </tr>
    <tr>
      <th>0</th>
      <td>24238194</td>
      <td>7.5</td>
      <td>2015-05-07 15:52:06-04:00</td>
      <td>-73.999817</td>
      <td>40.738354</td>
      <td>-73.999512</td>
      <td>40.723217</td>
      <td>1</td>
      <td>2015-05-07</td>
      <td>15:52</td>
      <td>2015</td>
      <td>5</td>
      <td>7</td>
      <td>3</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27835199</td>
      <td>7.7</td>
      <td>2009-07-17 16:04:56-04:00</td>
      <td>-73.994355</td>
      <td>40.728225</td>
      <td>-73.994710</td>
      <td>40.750325</td>
      <td>1</td>
      <td>2009-07-17</td>
      <td>16:04</td>
      <td>2009</td>
      <td>7</td>
      <td>17</td>
      <td>4</td>
      <td>16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>44984355</td>
      <td>12.9</td>
      <td>2009-08-24 17:45:00-04:00</td>
      <td>-74.005043</td>
      <td>40.740770</td>
      <td>-73.962565</td>
      <td>40.772647</td>
      <td>1</td>
      <td>2009-08-24</td>
      <td>17:45</td>
      <td>2009</td>
      <td>8</td>
      <td>24</td>
      <td>0</td>
      <td>17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25894730</td>
      <td>5.3</td>
      <td>2009-06-26 04:22:21-04:00</td>
      <td>-73.976124</td>
      <td>40.790844</td>
      <td>-73.965316</td>
      <td>40.803349</td>
      <td>3</td>
      <td>2009-06-26</td>
      <td>04:22</td>
      <td>2009</td>
      <td>6</td>
      <td>26</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17610152</td>
      <td>16.0</td>
      <td>2014-08-28 13:47:00-04:00</td>
      <td>-73.925023</td>
      <td>40.744085</td>
      <td>-73.973082</td>
      <td>40.761247</td>
      <td>5</td>
      <td>2014-08-28</td>
      <td>13:47</td>
      <td>2014</td>
      <td>8</td>
      <td>28</td>
      <td>3</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</pre>

## Trip Distance Calculation
Admit it, when we talk about factors influencing transportation fares, distance is probably the first thing that comes into our mind.

Although there are many APIs out there to calculate the optimal driving distance between 2 points, we will refrain from using them this time due to the following limitations:
* Performing requests on each row of this massive dataset is too time consuming
* They are mostly paid services, and
* Some of these APIs only return the shortest distance and doesn't take traffic congestion into account.

We can still opt for an alternative to calculating the distance between pickup and dropoff points, which is using the **Haversine distance**.
>**Haversine distance** is defined as the great-circle distance between two points on a sphere given their longitudes and latitudes, this assumes that the Earth is a perfect sphere (sorry flat-earthers!)
Although the Haversine distance is less superior compared to the driving distance, it generalises the distance for each trip and is less prone to overfitting the dataset.

<details>
<summary>View Code</summary>
	
```python
def hvs(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    a = np.sin(d_lat/2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon/2) ** 2
    distance = 6371* 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return distance

df['distance'] = df.apply(lambda row: hvs(row['pickup_latitude'], row['pickup_longitude'], row['dropoff_latitude'], row['dropoff_longitude']), axis=1)
```
</details>

```python
df['distance'].head()
```

### Output

<pre>
0    1.683323
1    2.457590
2    5.036377
3    1.661683
4    4.475450
Name: distance, dtype: float64
</pre>

## Exploratory Data Analysis
### Univariate Analysis
We haven't touch on the numerical features yet, we should probably have a look on each variable:

```python
fig, axs = plt.subplots(1, 2, figsize=(10, 1))
sns.boxplot(data=df['fare_amount'],orient='h', ax=axs[0]).set(title='fare_amount')
sns.boxplot(data=df['passenger_count'], orient='h', ax=axs[1]).set(title='passenger_count')
plt.show()
```
### Output
<img src='https://github.com/jylim21/bear-with-data.github.io/blob/main/uber-fares-analysis/images/6.png'>
There might seem a little too many extreme values in the fare_amount, but don't forget this is a really massive dataset so it is acceptable. 
As for passenger_count, it looks like most of the trips have 1 to 2 passengers which is reasonable since Uber cars are mostly 4-seaters.

### Bivariate Analysis
```python
sns.boxplot(data=df, y='passenger_count', x='fare_amount', orient='h')
```
### Output
<img src='https://github.com/jylim21/bear-with-data.github.io/blob/main/uber-fares-analysis/images/9.png'>

Although at first glance it seems that fares for 1 passenger are higher, but if we look closely, they are all outliers and in fact, they all have the same quartiles. This means fares are not determined by number of passengers.

Don't believe? Let's zoom in onto the quartiles:
```python
sns.boxplot(data=df[df['fare_amount']<30], y='passenger_count', x='fare_amount', orient='h')
df.groupby('passenger_count')['fare_amount'].describe()
```
### Output
<img src='https://github.com/jylim21/bear-with-data.github.io/blob/main/uber-fares-analysis/images/10.png'>

<pre>
                    count       mean        std   min  25%  50%   75%     max
passenger_count                                                              
1                134075.0  11.229843   9.601384  0.01  6.0  8.5  12.5  220.00
2                 28555.0  11.778504  10.129457  2.50  6.1  8.5  13.0  230.00
3                  8624.0  11.449652   9.490586  2.50  6.1  8.5  12.9  120.30
4                  4146.0  11.666331   9.942237  2.50  6.1  8.5  12.9  132.33
5                 13547.0  11.245369   9.288177  2.50  6.0  8.5  12.5  100.50
6                  4145.0  12.185884  10.148115  2.50  6.5  9.0  13.7  152.83
</pre>

From above, it sseems that the interquartile range (75% percentile minus 25% percentile) for fare_amounts are approximately 6.7 ± 0.2 for all passenger counts, with an exception of 6 passengers which is slightly higher than the rest, however the difference isn't too significant here.

## Hourly Comparison


Based on the figure, 
Fares are exceptionally higher past midnight from 12am to 2am, probably due to lesser drivers available.
In the daytime, fares do not differ much in gereral from 5am to 4pm, but a steep increase is observed after 4pm until midnight.
