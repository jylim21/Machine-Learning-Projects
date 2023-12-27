<h1 align="center">Laptop Price Prediction</h1>

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/laptop-price-prediction/images/laptop.jpg?raw=true)

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
import warnings
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

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

# Distribution plot
def dist(dtf,coln, title):
    sns.set(style="white")
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
    sns.boxplot(data=dtf, x=coln, ax=ax_box).set(title=title, xlabel='')
    sns.histplot(data=dtf, x=coln, ax=ax_hist, kde=True, color='blue').set(xlabel='')
    return

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

If you are not familiar with this custom function, here is a demonstration of it which I always use as the first step:

```python
df_train = pd.read_csv("/kaggle/input/laptop-price-prediction/laptops_train.csv")
df_test  = pd.read_csv("/kaggle/input/laptop-price-prediction/laptops_test.csv")
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

## Data Cleansing
By looking at the data types and dataframe, we have an idea on what should be done next:
* Features such as **Screen Size**, **RAM**, **Storage**, and **Weight** are all formatted as objects instead of numerical data types, due to the presence of measurement units (eg. GB, kg) which are strings. These features have to be converted to numerical form before we can visualize it.
* The *Screen* feature provides us with the following info, and should be broken down into:
  1. **Display Type** (Eg. IPS, Retina Display)
  2. **Screen Resolution** (Eg. 2560x1600)
* The CPU feature also provides us with 3 different info, namely
  1. **CPU brand** (eg. Intel, AMD)
  2. **CPU model** (eg. Core i5, Ryzen 5)
  3. **CPU speed** (eg. 2.3GHz)
* Besides Storage Capacity, we are also able to deduce the Storage type (eg. SSD, HDD) from the Storage feature.

As a computer geek who is in touch with the latest tech releases, these features should definitely have an influence on the laptop prices, or could I be wrong? Let's find out in a moment.

I combined the train and test data before cleansing to ensure the same transformation is applied to both data, we will split them again after the cleansing process.

```python
df=pd.concat([df_train,df_test])
```

And now we can proceed.

**Warning**: Lot's of RegeX ahead, proceed with a calm mind!

<details>
<summary>View Code</summary>
	
```python
# Splitting "GPU" column
df['GPU'] = df['GPU'].str.replace(r"Nvidia G\D+\s*", "Nvidia GeForce ", regex=True)
df['GeForce_ver'] = df['GPU'].str.extract(r"(?<=Nvidia GeForce\s)(\d+)").astype(float)
df['GPU'] = df['GPU'].str.replace(r"Quadro M", "Quadro ", regex=True)
df['Quadro_ver'] = df['GPU'].str.extract(r"(?<=Nvidia Quadro\s)(\d+)").astype(float)
#df['IntelGPU_ver'] = df['GPU'].str.extract(r"\bIntel\b.*?(\d+)")
df['IntelGPU_ver'] = df['GPU'].str.extract(r"Intel\s+\D*(\d+)").astype(float)
df['GPU'] = df['GPU'].str.replace(r"AMD Radeon RX", "AMD Radeon R10", regex=True)
df['Radeon_Gen'] = df['GPU'].str.extract(r"(?<=Radeon R)(\d+)").astype(float)
df['Radeon_ver'] = df['GPU'].str.extract(r"Radeon.+([\d]{3})") .astype(float)
df['FirePro_ver'] = df['GPU'].str.extract(r"(?<=AMD FirePro W)(\d+)") .astype(float)

# Splitting "CPU" column
df['AMD_A_Gen']=df['CPU'].str.extract(r'(?<=AMD A)(\d+)').astype(float)
df['AMD_A_ver']=df['CPU'].str.extract(r'(?<=AMD A).*(\d{4})').astype(float)
df['AMD_E_ver']=df['CPU'].str.extract(r'(?<=AMD E).*(\d{4})').astype(float)
df['AMD_FX_ver']=df['CPU'].str.extract(r'(?<=AMD FX).*(\d{4})').astype(float)
df['AMD_Ryzen_ver']=df['CPU'].str.extract(r'(?<=AMD Ryzen).*(\d{4})').astype(float)
df['Intel_Atom_ver']=df['CPU'].str.extract(r'(?<=Intel Atom).*(\d{4})').astype(float)
df['Intel_Celeron_ver']=df['CPU'].str.extract(r'(?<=Intel Celeron).*(\d{4})').astype(float)
df['Intel_Celeron_Cores']=df['CPU'].str.extract(r'(?<=Celeron\s)(\w*)').replace({'Quad': 4, 'Dual':2})
df['Intel_Core_i_Gen']=df['CPU'].str.extract(r'(?<=Intel Core [iI])(\d+)').astype(float)
df['Intel_Core_i_ver']=df['CPU'].str.extract(r'(?<=Intel Core [iI]).*(\d{4})').astype(float)
df['Intel_Pentium_ver']=df['CPU'].str.extract(r'(?<=Intel Pentium).*(\d{4})').astype(float)
df['Intel_Pentium_Cores']=df['CPU'].str.extract(r'(?<=Pentium\s)(\w*)').replace({'Quad': 4, 'Dual':2})
df['Intel_Xeon_ver']=df['CPU'].str.extract(r'(?<=Intel Xeon).*(\d{4})').astype(float)
df['CPU_Speed']=df['CPU'].str.rsplit(n=1).str.get(-1).str.replace('GHz', '').astype(float)

# Splitting "Screen" column
df['Screen Size']=df['Screen Size'].str.replace('"', '').astype(float)
df['Resolution_1']=df['Screen'].str.rsplit('x', n=1).str[0].str[-4:].astype(int)
df['Resolution_2']=df['Screen'].str.rsplit('x', n=1).str[1].str[-4:].astype(int)
df['IPS Panel'] = (df['Screen'].str.contains('IPS Panel')).astype(int)
df['Touchscreen'] = (df['Screen'].str.contains('Touchscreen')).astype(int)
df['Retina Display'] = (df['Screen'].str.contains('Retina Display')).astype(int)

# Splitting "Storage" into HDD, SSD, Flash Storage & hybrid storage volumes in GB
df['SSD']=df[' Storage'].str.extract(r'\b(\w+)\s+SSD\b')
df['HDD']=df[' Storage'].str.extract(r'\b(\w+)\s+HDD\b')
df['Flash_Storage']=df[' Storage'].str.extract(r'\b(\w+)\s+Flash\sStorage\b')
df['Hybrid']=df[' Storage'].str.extract(r'\b(\w+)\s+Hybrid\b')
df['HDD']=df['HDD'].str.replace('TB','000GB').str.replace('GB','').fillna(0).astype(int)
df['SSD']=df['SSD'].str.replace('TB','000GB').str.replace('GB','').fillna(0).astype(int)
df['Flash_Storage']=df['Flash_Storage'].str.replace('TB','000GB').str.replace('GB','').fillna(0).astype(int)
df['Hybrid']=df['Hybrid'].str.replace('TB','000GB').str.replace('GB','').fillna(0).astype(int)

# Combine Operating System and its version
df['OS'] = df['Operating System'] + ' ' + df['Operating System Version'].fillna(' ')

# Other misc. cleansing steps
df['RAM']=df['RAM'].str.replace('GB', '').astype(int)
df['Weight']=df['Weight'].str.replace('kgs', '').str.replace('kg', '').astype(float)

# Creating new Total_Storage feature
df['Total_Storage']=df['SSD']+df['HDD']+df['Flash_Storage']+df['Hybrid']

# Filling all NaNs
df.fillna(0, inplace=True)
```
</details>

We then split the data into its original train and test entries.
```
df_train=df.iloc[0:len(df_train)]
df_test=df.iloc[len(df_train):]
```

Looking at our final form:

```
summary(df_train)
df_train.head(5)
```
### Output

<pre>
                             Type     NaN      Zeros
Manufacturer               object  0 (0%)     0 (0%)
Model Name                 object  0 (0%)     0 (0%)
Category                   object  0 (0%)     0 (0%)
Screen Size               float64  0 (0%)     0 (0%)
Screen                     object  0 (0%)     0 (0%)
CPU                        object  0 (0%)     0 (0%)
RAM                         int64  0 (0%)     0 (0%)
 Storage                   object  0 (0%)     0 (0%)
GPU                        object  0 (0%)     0 (0%)
Operating System           object  0 (0%)     0 (0%)
Operating System Version   object  0 (0%)  136 (13%)
Weight                    float64  0 (0%)     0 (0%)
Price                     float64  0 (0%)     0 (0%)
GeForce_ver               float64  0 (0%)  693 (70%)
Quadro_ver                float64  0 (0%)  951 (97%)
IntelGPU_ver              float64  0 (0%)  466 (47%)
Radeon_Gen                float64  0 (0%)  896 (91%)
Radeon_ver                float64  0 (0%)  864 (88%)
FirePro_ver               float64  0 (0%)  974 (99%)
AMD_A_Gen                 float64  0 (0%)  942 (96%)
AMD_A_ver                 float64  0 (0%)  942 (96%)
AMD_E_ver                 float64  0 (0%)  970 (99%)
AMD_FX_ver                float64  0 (0%)  975 (99%)
AMD_Ryzen_ver             float64  0 (0%)  973 (99%)
Intel_Atom_ver            float64  0 (0%)  969 (99%)
Intel_Celeron_ver         float64  0 (0%)  914 (93%)
Intel_Celeron_Cores       float64  0 (0%)  914 (93%)
Intel_Core_i_Gen          float64  0 (0%)  156 (15%)
Intel_Core_i_ver          float64  0 (0%)  179 (18%)
Intel_Pentium_ver         float64  0 (0%)  954 (97%)
Intel_Pentium_Cores       float64  0 (0%)  954 (97%)
Intel_Xeon_ver            float64  0 (0%)  973 (99%)
CPU_Speed                 float64  0 (0%)     0 (0%)
Resolution_1                int64  0 (0%)     0 (0%)
Resolution_2                int64  0 (0%)     0 (0%)
IPS Panel                   int64  0 (0%)  697 (71%)
Touchscreen                 int64  0 (0%)  836 (85%)
Retina Display              int64  0 (0%)  963 (98%)
SSD                         int64  0 (0%)  324 (33%)
HDD                         int64  0 (0%)  556 (56%)
Flash_Storage               int64  0 (0%)  922 (94%)
Hybrid                      int64  0 (0%)  975 (99%)
OS                         object  0 (0%)     0 (0%)
Total_Storage               int64  0 (0%)     0 (0%)
</pre>
<pre>
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
      <th>...</th>
      <th>Resolution_2</th>
      <th>IPS Panel</th>
      <th>Touchscreen</th>
      <th>Retina Display</th>
      <th>SSD</th>
      <th>HDD</th>
      <th>Flash_Storage</th>
      <th>Hybrid</th>
      <th>OS</th>
      <th>Total_Storage</th>
    </tr>
    <tr>
      <th>0</th>
      <td>Apple</td>
      <td>MacBook Pro</td>
      <td>Ultrabook</td>
      <td>13.3</td>
      <td>IPS Panel Retina Display 2560x1600</td>
      <td>Intel Core i5 2.3GHz</td>
      <td>8</td>
      <td>128GB SSD</td>
      <td>Intel Iris Plus Graphics 640</td>
      <td>macOS</td>
      <td>...</td>
      <td>1600</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>128</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>macOS</td>
      <td>128</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Apple</td>
      <td>Macbook Air</td>
      <td>Ultrabook</td>
      <td>13.3</td>
      <td>1440x900</td>
      <td>Intel Core i5 1.8GHz</td>
      <td>8</td>
      <td>128GB Flash Storage</td>
      <td>Intel HD Graphics 6000</td>
      <td>macOS</td>
      <td>...</td>
      <td>900</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>128</td>
      <td>0</td>
      <td>macOS</td>
      <td>128</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HP</td>
      <td>250 G6</td>
      <td>Notebook</td>
      <td>15.6</td>
      <td>Full HD 1920x1080</td>
      <td>Intel Core i5 7200U 2.5GHz</td>
      <td>8</td>
      <td>256GB SSD</td>
      <td>Intel HD Graphics 620</td>
      <td>No OS</td>
      <td>...</td>
      <td>1080</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>256</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No OS</td>
      <td>256</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Apple</td>
      <td>MacBook Pro</td>
      <td>Ultrabook</td>
      <td>15.4</td>
      <td>IPS Panel Retina Display 2880x1800</td>
      <td>Intel Core i7 2.7GHz</td>
      <td>16</td>
      <td>512GB SSD</td>
      <td>AMD Radeon Pro 455</td>
      <td>macOS</td>
      <td>...</td>
      <td>1800</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>512</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>macOS</td>
      <td>512</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Apple</td>
      <td>MacBook Pro</td>
      <td>Ultrabook</td>
      <td>13.3</td>
      <td>IPS Panel Retina Display 2560x1600</td>
      <td>Intel Core i5 3.1GHz</td>
      <td>8</td>
      <td>256GB SSD</td>
      <td>Intel Iris Plus Graphics 650</td>
      <td>macOS</td>
      <td>...</td>
      <td>1600</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>256</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>macOS</td>
      <td>256</td>
    </tr>
  </tbody>
</table>
</pre>

# EDA
**Breakdown of Laptops by Brand**

In this dataset, the greatest quantity of laptops comes from the usual Manufacturers we see on the market:
1. Dell
2. Lenovo
3. Hewlett-Packard (HP)
4. Asus
5. Acer

```python
ax=sns.countplot(y=df_train['Manufacturer'], order=df_train['Manufacturer'].value_counts(ascending=False).index)
sns.set(rc={'figure.figsize':(15,8.27)})
print(ax.bar_label(container=ax.containers[0], labels=df_train['Manufacturer'].value_counts(ascending=False).values))
```
### Output
![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/laptop-price-prediction/images/1.png?raw=true)
