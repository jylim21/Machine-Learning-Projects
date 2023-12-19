<h1 align="center">Data Visualization: Uber Fare Analysis</h1>

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
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import folium
from folium.plugins import MarkerCluster, MousePosition
from folium.features import DivIcon
from geopy.geocoders import Nominatim

df = pd.read_csv("/kaggle/input/uber-fares-dataset/uber.csv")
```
I have also defined a custom function to check on empty and zero values within the dataset as follow, it returns the number and % of NaNs and zeros for each column in the dataframe:

```
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

print(summary(df))
```
### Output
![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/Uber/1.jpg?raw=true)

