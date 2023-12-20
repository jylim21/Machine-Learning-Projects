<h1 align="center">Laptop Price Prediction</h1>

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/Uber/uber.jpg?raw=true)

In 2023, *Uber* has a reported active drivers reaching 1.5 million in the United States alone, occupying up to 74% of the global e-hailing market.

A core part of *Uber*'s business is dynamic pricing - also known as surge pricing - where fares fluctuate based on supply and demand principles. When demand for rides is high, such as during rush hour or special events when the demand for rides are high, fares increase automatically to entice more drivers to get on the road.

As *Uber* has grown exponentially in recent years to become the dominant e-hailing service globally, analysis on its pricing model and decisions has become increasingly important. This webpage will take a data-driven look at how *Uber*'s fares are set across different situations. The goal is to provide transparency and insight into a key aspect of a company that has disrupted transportation enormously in a short span of time.

## THE PROJECT
This analysis will be conducted using Python, with the following libraries:

* **Pandas** - A staple for reading and manipulating structured data.
* **Numpy** - For array-based operations.
* **Matplotlib** and **Seaborn** - Used together to produce stunning visuals.
* **Datetime** - Used to parse datetimes into their respective components such as days, hours, and minutes.
* **Folium** - An interactive map to visualize pickup and dropoff locations with provided coordinates
* **Nominatim** by Geopy - Used to calculate the Haversine distance between 2 coordinates

And we will be using the [Uber Fares Dataset](https://www.kaggle.com/datasets/yasserh/uber-fares-dataset) by *M YASSER H* which can be found on Kaggle, it contains data of 500k trips which are mainly concentrated in the suburbs of New York City, a pretty big one I would say!

To begin, let's import all the relevant libraries to our kernel.

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
![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/Uber/1.jpg?raw=true)

Looks like we have at least 3k entries having nil latitudes and longitudes which does not seem to be valid here. Though it is generally not advisable to drop empty rows, this case is an exception as the empty entries only make up 1% of the dataset, we shouldn't waste time figuring how to impute these empty values as they could introduce biases to the data.

We will do the same approach for zero values in fare_amount and the NaNs in dropoff_longitude and dropoff_latitude.

```python
df=df[df['pickup_longitude']!=0]
df=df[df['pickup_latitude']!=0 | df['dropoff_longitude']!=0 | df['dropoff_latitude']!=0 | df['passenger_count']!=0]
df.reset_index(inplace=True)
df.drop(['index','key'], axis=1, inplace=True)
df.rename(columns={"Unnamed: 0": "trip"}, inplace=True)

print(summary(df))
```
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

By plotting the coordinates on a map, we know for sure that this driver is based in New York City. Unfortunately, this map also exposes many other problematic coordinates provided to us, there are a few coordinates which are in the sea, Europe, Africa, and even Antartica!

The zero entries removed earlier were just the tip of an iceberg, we need to screen through the coordinates and eliminate any anomalies such as:

#### 1. Invalid coordinates
latitudes and longitudes can only take on the following range of values, any value which falls outside these ranges are 100% invalid:
* Latitudes: -90 to 90
* Longitudes: -180 to 180
```python
df[(abs(df['pickup_latitude'])>90)|(abs(df['dropoff_latitude'])>90)|(abs(df['dropoff_longitude'])>180)|(abs(df['pickup_longitude'])>180)].head(10)
```

#### 2. Swapped Latitude and Longitude Values 
Some rides had their latitude and longitude pairs reversed by plotting the coordinates, this is where the Antartica coordinates came from:
```python
df[(df['pickup_latitude']<-70) | (df['dropoff_latitude']<-70)].head()
```		
									
#### 3. Intercontinential Travel
There are also trips which spans across continents, and amazingly they all took less than $30, what a bang for a buck!								
```python
df=df[abs(df['pickup_longitude']-df['dropoff_longitude'])<40]
```
									
#### 4. Trips in the Ocean
Obviously we can't have coordinates in the ocean:

```python
df=df[abs(df['pickup_longitude'])>1]
df=df[~(((df['pickup_latitude']<40.52) & (df['pickup_longitude']>-73.96)) | ((df['dropoff_latitude']<40.52) & (df['dropoff_longitude']>-73.96)))]
```
									
#### 5. Identical Dropoff & Pickup Locations
Seriously, why am I paying if your car didn't even budge? And surprisingly there were 1,316 of such trips!
```python
df=df[(df['pickup_longitude']!= df['dropoff_longitude']) | (df['pickup_latitude']!= df['dropoff_latitude'])]
```
