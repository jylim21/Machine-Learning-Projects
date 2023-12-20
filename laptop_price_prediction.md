<h1 align="center">Laptop Price Prediction</h1>

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/laptop-price-prediction/laptop.jpg?raw=true)

Laptop processors have undergone an intriguing competitive shift in recent years. *Intel* dominated for over a decade, but began facing renewed competition recently - not just from long-time rival *AMD* making a comeback, but also *Apple* bringing proprietary M1 and M2 ARM-based silicon chips to MacBooks.

*AMD* re-entered with more affordable x86 CPUs matching and exceeding Intel's computing power, which boosted AMD's market share from around 10% in 2016 to over 20% by 2019, degrading Intel's dominant position. Meanwhile Apple's new in-house MacBook chipsets boast industry-leading power efficiency that competes on overall performance. This three-way battle between Intel, AMD and Apple has meaningful implications on laptop pricing and performance-per-dollar.

As *AMD* and *Apple* chips become more popular in laptops, did that translate to lower average prices for equivalent performance? Does *Intel* still command a premium due to long-standing market leadership? This analysis investigates recent pricing trends using laptop specs like CPU, Monitor, RAM, storage and GPU etc.

## THE PROJECT
The dataset used here is the [Laptop Price Prediction Dataset](https://www.kaggle.com/datasets/arnabchaki/laptop-price-prediction) by *RANDOMARNAB* which can be found on Kaggle, it is comprised of 977 laptops from brands such as Dell, HP, Apple, Acer etc. along with their relevant hardware specifications, not a big dataset I would say.

As usual, we will begin with importing the common libraries such as pandas, numpy, matplotlib, and seaborn, along with my custom function which counts the occurrences of NaNs / Zeroes:
	
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

If you are not familiar with this custom function, here is a demonstration of it which I always use as the first step:

```python
df_train = pd.read_csv("/kaggle/input/laptop-price-prediction/laptops_train.csv")
summary(df_train)
```

### Output
<pre>
                             Type        NaN   Zeros
Manufacturer               object     0 (0%)  0 (0%)
Model Name                 object     0 (0%)  0 (0%)
Category                   object     0 (0%)  0 (0%)
Screen Size                object     0 (0%)  0 (0%)
Screen                     object     0 (0%)  0 (0%)
CPU                        object     0 (0%)  0 (0%)
RAM                        object     0 (0%)  0 (0%)
 Storage                   object     0 (0%)  0 (0%)
GPU                        object     0 (0%)  0 (0%)
Operating System           object     0 (0%)  0 (0%)
Operating System Version   object  136 (13%)  0 (0%)
Weight                     object     0 (0%)  0 (0%)
Price                     float64     0 (0%)  0 (0%)

<table border="1" class="dataframe">
  <tbody>
    <tr style="text-align:right">
      <th></th>
      <th>Manufacturer</th>
      <th>Model Name</th>
      <th>Category</th>
      <th>Screen Size</th>
      <th>Screen</th>
      <th>CPU</th>
      <th>RAM</th>
      <th>Storage</th>
      <th>GPU</th>
      <th>Operating System</th>
      <th>Operating System Version</th>
      <th>Weight</th>
      <th>Price</th>
    </tr>
    <tr>
      <th>0</th>
      <td>Apple</td>
      <td>MacBook Pro</td>
      <td>Ultrabook</td>
      <td>13.3"</td>
      <td>IPS Panel Retina Display 2560x1600</td>
      <td>Intel Core i5 2.3GHz</td>
      <td>8GB</td>
      <td>128GB SSD</td>
      <td>Intel Iris Plus Graphics 640</td>
      <td>macOS</td>
      <td>NaN</td>
      <td>1.37kg</td>
      <td>11912523.48</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Apple</td>
      <td>Macbook Air</td>
      <td>Ultrabook</td>
      <td>13.3"</td>
      <td>1440x900</td>
      <td>Intel Core i5 1.8GHz</td>
      <td>8GB</td>
      <td>128GB Flash Storage</td>
      <td>Intel HD Graphics 6000</td>
      <td>macOS</td>
      <td>NaN</td>
      <td>1.34kg</td>
      <td>7993374.48</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HP</td>
      <td>250 G6</td>
      <td>Notebook</td>
      <td>15.6"</td>
      <td>Full HD 1920x1080</td>
      <td>Intel Core i5 7200U 2.5GHz</td>
      <td>8GB</td>
      <td>256GB SSD</td>
      <td>Intel HD Graphics 620</td>
      <td>No OS</td>
      <td>NaN</td>
      <td>1.86kg</td>
      <td>5112900.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Apple</td>
      <td>MacBook Pro</td>
      <td>Ultrabook</td>
      <td>15.4"</td>
      <td>IPS Panel Retina Display 2880x1800</td>
      <td>Intel Core i7 2.7GHz</td>
      <td>16GB</td>
      <td>512GB SSD</td>
      <td>AMD Radeon Pro 455</td>
      <td>macOS</td>
      <td>NaN</td>
      <td>1.83kg</td>
      <td>22563005.40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Apple</td>
      <td>MacBook Pro</td>
      <td>Ultrabook</td>
      <td>13.3"</td>
      <td>IPS Panel Retina Display 2560x1600</td>
      <td>Intel Core i5 3.1GHz</td>
      <td>8GB</td>
      <td>256GB SSD</td>
      <td>Intel Iris Plus Graphics 650</td>
      <td>macOS</td>
      <td>NaN</td>
      <td>1.37kg</td>
      <td>16037611.20</td>
    </tr>
  </tbody>
</table>
</pre>

Nothing looks unusual except for Operating System Version, which has 136 (13%) NaN values. It might be tempting to remove them right away but since we have only 900+ data, it's better to understand the data further before dealing with it.

By looking at the data types and dataframe, we have an idea on what should be done next:
* Features such as Screen Size, RAM, Storage, and Weight are all formatted as objects instead of numerical data types, due to the presence of measurement units (eg. GB, kg) which are strings. These features have to be converted to numerical form before we can visualize it.
* The Screen feature provides us with the following info, and should be broken down into:
  1. Panel Type (Eg. IPS)
  2. Display Type (Eg. Retina Display)
  3. Screen Resolution (Eg. 2560x1600)
* The CPU feature also provides us with 3 different info, namely
  1. CPU brand (eg. Intel, AMD)
  2. CPU model (eg. Core i5, Ryzen 5)
  3. CPU speed (eg. 2.3GHz)
* Besides Storage Capacity, we are also able to deduce the Storage type (eg. SSD, HDD) from the Storage feature.

