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
<img src='https://github.com/jylim21/bear-with-data.github.io/blob/main/laptop-price-prediction/images/1.png' width='800'>

**Price Breakdown by Brand:**

By observing the price distribution of different laptop brands, we can divide the brands into at least 4 different price tiers according to their price range and median price:
* Tier 1   : Vero, Mediacom, Chuwi
* Tier 1.5 : Acer, Fujitsu
* Tier 2   : HP, Dell, Lenovo, Asus, Xiaomi
* Tier 2.5 : Toshiba
* Tier 3   : Huawei, Apple, Microsoft, MSI, Samsung , Google
* Tier 3.5 : LG
* Tier 4   : Razer

```python
print(sns.boxplot(x=df_train['Price'], y=df_train['Manufacturer'],orient='h', order=["Vero","Mediacom","Chuwi","Acer","Fujitsu","HP","Asus","Lenovo","Dell","Xiaomi","Toshiba","Huawei","Apple","Google","Microsoft","Samsung","MSI","LG","Razer"]))
```

### Output
<img src='https://github.com/jylim21/bear-with-data.github.io/blob/main/laptop-price-prediction/images/2.png' width='800'>

**Price Breakdown by Laptop Category**

The price difference is obvious between categories, it is observed that *Netbooks* are the cheapest, this is followed by *Notebook* and eventually the costliest laptops are from the *Workstation* Category. 

Besides, the price distribution *Ultrabook* is fairly close to *2 in 1 Convertible* and *Gaming Laptops* are fairly close in terms of either price range or median price.

```python
print(sns.boxplot(x=df_train['Price'], y=df_train['Category'],orient='h', order=["Netbook","Notebook","2 in 1 Convertible","Ultrabook","Gaming","Workstation"]))
```

### Output
<img src='https://github.com/jylim21/bear-with-data.github.io/blob/main/laptop-price-prediction/images/3.png' width='800'>

**Price Breakdown by Operating System Version**

```python
sns.boxplot(x=df_train['Price'], y=df_train['OS'],orient='h', order=["Android  ","Chrome OS  ","No OS  ","Linux  ","Windows 10","Mac OS X","Windows 10 S","Windows 7","macOS  "])
```

### Output
<img src='https://github.com/jylim21/bear-with-data.github.io/blob/main/laptop-price-prediction/images/4.png' width='800'>

**Qualitative Variable EDA (Correlation)**

We analyse the relationship between numerical variables and the price using a correlation heatmap. 

```python
sns.set (rc = {'figure.figsize':(12, 12)})
sns.heatmap(df_train.corr())#, annot=True, fmt=".1f")
```

### Output
<img src='https://github.com/jylim21/bear-with-data.github.io/blob/main/laptop-price-prediction/images/5.png' width='800'>

By focusing only on variables which are highly correlated with Price, we can deduce the following:

1. **Resolution_1** and **Resolution_2** are perfectly correlated, we will only keep either one (I chose **Resolution_1**, no specific reason here but it's fine if you chose Resolution_2 either).
2. Price is highly positively correlated with **RAM** (r=0.7), **Resolution** (r=0.6) and **SSD** (r=0.7), so we would expect to see laptops with more RAM, higher resolution and greater SSD storage space to be costlier.

```python
sns.heatmap(df_train[['Price','RAM','SSD','Category','Resolution_1','Intel_Core_i_Gen','CPU_Speed','Quadro_ver','GeForce_ver','Manufacturer_Razer','IPS Panel']].corr(), annot=True, fmt=".2f")
```

### Output
<img src='https://github.com/jylim21/bear-with-data.github.io/blob/main/laptop-price-prediction/images/6.png' width='800'>

# Data Transformation

Distribution of Weight is observed to be positively skewed. By taking a log-transformation approach, we manage to normalize it despite having a few stubborn outliers.

```python
fig, axs = plt.subplots(2, 2, figsize=(10, 5), gridspec_kw={'height_ratios': [1, 5]})
sns.boxplot(x=df_train['Weight'], orient='h', ax=axs[0, 0])
axs[0, 0].set_xlabel('')
axs[0, 0].set_title('Weight')
sns.histplot(x=df_train['Weight'], ax=axs[1, 0])
axs[1, 0].set_xlabel('')
axs[1, 0].set_ylabel('')
sns.boxplot(x=np.log(df_train['Weight']), orient='h', ax=axs[0, 1])
axs[0, 1].set_xlabel('')
axs[0, 1].set_title('log(Weight)')
sns.histplot(x=np.log(df_train['Weight']), ax=axs[1, 1])
axs[1, 1].set_xlabel('')
axs[1, 1].set_ylabel('')
plt.show()

# Log-transforming Weight feature in both train & test set
df_train['Weight']=np.log(df_train['Weight'])
df_test['Weight']=np.log(df_test['Weight'])
```

### Output

<img src='https://github.com/jylim21/bear-with-data.github.io/blob/main/laptop-price-prediction/images/7.png' width='800'>

# Feature Engineering
We will now perform **One-Hot Encoding** on the string columns 'Manufacturer' and 'OS' to assign a binary indicator column for each unique value, and keep only n-1 of those values.

As for the 'Category' feature, it does not have many different values and the EDA earlier showed us that certain laptop categories are higher than each other, so we will do a **Ordinal Encoding** on Category instead.

```python
df = pd.concat([df_train,df_test])

df = pd.get_dummies(df, columns = ['Manufacturer','OS'], drop_first=True)
df['Manufacturer']=df['Manufacturer'].replace(['Apple','Huawei','Microsoft','MSI','Samsung','Google'],'3').replace(['Vero','Mediacom','Chuwi'],'1').replace(['Acer','Fujitsu'],'1.5').replace(['HP','Dell','Lenovo','Asus','Xiaomi'],'2').replace(['Toshiba'],'2.5').replace(['LG'],'3.5').replace(['Razer'],'4').astype(float)
df['Category']=df['Category'].replace({ 'Netbook' : 1, 'Notebook' : 2, '2 in 1 Convertible' : 3, 'Ultrabook' : 3.5, 'Gaming' : 4, 'Workstation' : 5 })

# Splitting back into train and test data
df_train=df.iloc[0:len(df_train)]
df_test=df.iloc[len(df_train):]
df_train.head(5)
```

### Output

<pre>
<table border="1" class="dataframe">
  <tbody>
    <tr style="text-align:right">
      <th></th>
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
      <th>...</th>
      <th>Manufacturer_Vero</th>
      <th>Manufacturer_Xiaomi</th>
      <th>OS_Chrome OS</th>
      <th>OS_Linux</th>
      <th>OS_Mac OS X</th>
      <th>OS_No OS</th>
      <th>OS_Windows 10</th>
      <th>OS_Windows 10 S</th>
      <th>OS_Windows 7</th>
      <th>OS_macOS</th>
    </tr>
    <tr>
      <th>0</th>
      <td>MacBook Pro</td>
      <td>3.5</td>
      <td>13.3</td>
      <td>IPS Panel Retina Display 2560x1600</td>
      <td>Intel Core i5 2.3GHz</td>
      <td>8</td>
      <td>128GB SSD</td>
      <td>Intel Iris Plus Graphics 640</td>
      <td>macOS</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>1</th>
      <td>Macbook Air</td>
      <td>3.5</td>
      <td>13.3</td>
      <td>1440x900</td>
      <td>Intel Core i5 1.8GHz</td>
      <td>8</td>
      <td>128GB Flash Storage</td>
      <td>Intel HD Graphics 6000</td>
      <td>macOS</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>2</th>
      <td>250 G6</td>
      <td>2.0</td>
      <td>15.6</td>
      <td>Full HD 1920x1080</td>
      <td>Intel Core i5 7200U 2.5GHz</td>
      <td>8</td>
      <td>256GB SSD</td>
      <td>Intel HD Graphics 620</td>
      <td>No OS</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MacBook Pro</td>
      <td>3.5</td>
      <td>15.4</td>
      <td>IPS Panel Retina Display 2880x1800</td>
      <td>Intel Core i7 2.7GHz</td>
      <td>16</td>
      <td>512GB SSD</td>
      <td>AMD Radeon Pro 455</td>
      <td>macOS</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>4</th>
      <td>MacBook Pro</td>
      <td>3.5</td>
      <td>13.3</td>
      <td>IPS Panel Retina Display 2560x1600</td>
      <td>Intel Core i5 3.1GHz</td>
      <td>8</td>
      <td>256GB SSD</td>
      <td>Intel Iris Plus Graphics 650</td>
      <td>macOS</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
  </tbody>
</table>
</pre>

And in the last step before we train our model, we remove all non-numeric columns.

```python
# Removing string columns
categ_col=['Model Name','Screen','CPU',' Storage','GPU','Operating System','Operating System Version']
df_train = df_train.drop(categ_col, axis=1)
df_test = df_test.drop(categ_col, axis=1)
df_train.head()
```

### Output

<table border="1" class="dataframe">
  <tbody>
    <tr style="text-align:right">
      <th></th>
      <th>Category</th>
      <th>Screen Size</th>
      <th>RAM</th>
      <th>Weight</th>
      <th>Price</th>
      <th>GeForce_ver</th>
      <th>Quadro_ver</th>
      <th>IntelGPU_ver</th>
      <th>Radeon_Gen</th>
      <th>Radeon_ver</th>
      <th>...</th>
      <th>Manufacturer_Vero</th>
      <th>Manufacturer_Xiaomi</th>
      <th>OS_Chrome OS</th>
      <th>OS_Linux</th>
      <th>OS_Mac OS X</th>
      <th>OS_No OS</th>
      <th>OS_Windows 10</th>
      <th>OS_Windows 10 S</th>
      <th>OS_Windows 7</th>
      <th>OS_macOS</th>
    </tr>
    <tr>
      <th>0</th>
      <td>3.5</td>
      <td>13.3</td>
      <td>8</td>
      <td>0.314811</td>
      <td>11912523.48</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>640.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>1</th>
      <td>3.5</td>
      <td>13.3</td>
      <td>8</td>
      <td>0.292670</td>
      <td>7993374.48</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>15.6</td>
      <td>8</td>
      <td>0.620576</td>
      <td>5112900.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>620.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.5</td>
      <td>15.4</td>
      <td>16</td>
      <td>0.604316</td>
      <td>22563005.40</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>455.0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>4</th>
      <td>3.5</td>
      <td>13.3</td>
      <td>8</td>
      <td>0.314811</td>
      <td>16037611.20</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>650.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
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
    </tr>
  </tbody>
</table>

# Model Training
With the dataset ready to be trained, we proceed to split the test and train data which were combined earlier for data cleansing. The algorithms will have their hyperparameters tuned using Optuna.

For a training dataset with just 977 entries, we would opt for algorithms which have a higher bias to prevent overfitting. 

Some of the algorithms used here are:
* Linear Regression
* Ridge Regression
* Decision Tree
* Gradient Boost
* Random Forest
* XGBoost
* ElasticNet

```python
import optuna
from sklearn.metrics import r2_score, make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn import utils

y_train=df_train['Price']
x_train=df_train.drop(['Price'],axis=1)
y_test=df_test['Price']
x_test=df_test.drop(['Price'],axis=1)

optuna.logging.set_verbosity(optuna.logging.WARNING)
```

### Baseline Algorithm - Linear Regression
Ideally we would start off by using the simplest **Linear Regression** algorithm as it is the easiest to interpret, although it might not perform as well as other more complex algorithms.

A **10-fold cross validation** will be performed on the training set before experimenting it on the test set to evaluate the model performance, hopefully the performance on the test set would not differ too much from the cross-validation performance otherwise it would be a sign of overfitting.

```python
def objective(trial):
    model = LinearRegression()
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    r2_scorer = make_scorer(r2_score)
    scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring=r2_scorer)
    return np.mean(scores)

# Create and run the Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
best_params = study.best_params

LR=LinearRegression(**best_params)
LR.fit(x_train, y_train)
y_pred = LR.predict(x_test)
print("Linear Regression Results")
print("R-squared: ", "%.4f" % r2_score(y_test, y_pred))
print("Mean Absolute Error: ", "{0:,.2f}".format(mean_absolute_error(y_test, y_pred)))
print("Root Mean Squared Error: ", "{0:,.2f}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
```

### Output
<pre>
Linear Regression Results
R-squared:  0.7318
Mean Absolute Error:  2,146,615.12
Root Mean Squared Error:  3,073,002.35
</pre>

The Linear Regression model was able to fit with 73% R-squared, although not too impressive but it could help us identify the main drivers of the laptop prices.

### An extension to Linear Regression - Ridge Regression (L2)
Since we have 60 features above, the Linear Regression model could be prone to overfitting. This could be mitigated by introducing a L2 regularization factor to surpress less important features.

```python
def objective(trial):
    alpha = trial.suggest_float('alpha', 0.0001, 10.0, log=True)
    
    model = Ridge(alpha=alpha)
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    r2_scorer = make_scorer(r2_score)
    scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring=r2_scorer)
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
best_params = study.best_params

RR=Ridge(**best_params)
RR.fit(x_train, y_train)
y_pred = RR.predict(x_test)
print("Ridge Regression Results")
print("R-squared: ", "%.4f" % r2_score(y_test, y_pred))
print("Mean Absolute Error: ", "{0:,.2f}".format(mean_absolute_error(y_test, y_pred)))
print("Root Mean Squared Error: ", "{0:,.2f}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
```

### Output
<pre>
Ridge Regression Results
R-squared:  0.7367
Mean Absolute Error:  2,134,474.55
Root Mean Squared Error:  3,045,033.43
</pre>

Despite having a R-squared of 74%, there is no significant improvement from the Linear Regression model.

### Decision Tree
The **Decision Tree** is another interpretable algorithm which tells us how it arrives at its predictions starting from the top of the tree, all the way down to it's node.

However there are many parameters here to optimize for the best performance. To handle all these dirty work, we will utilize **Optuna** to find the optimal parameters which yields the **highest R-squared** metric.

```python
def objective(trial):
    criterion = trial.suggest_categorical('criterion', ['squared_error', 'friedman_mse','absolute_error', 'poisson'])
    #splitter = trial.suggest_categorical('splitter', ['best','random'])
    max_depth = trial.suggest_int('max_depth', 6, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 50)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 20)
    #max_features = trial.suggest_categorical('max_features', ['auto','sqrt','log2'])
    #random_state = trial.suggest_int('random_state', 1, 30)

    model = DecisionTreeRegressor(criterion=criterion,
                                  splitter='best',
                                  max_depth=max_depth,
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,
                                  max_features='auto',
                                  random_state=20
                                 )

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    r2_scorer = make_scorer(r2_score)
    scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring=r2_scorer)
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
best_params = study.best_params

DT=DecisionTreeRegressor(**best_params)
DT.fit(x_train, y_train)
y_pred = DT.predict(x_test)
print("Decision Tree Results")
print("R-squared: ", "%.4f" % r2_score(y_test, y_pred))
print("Mean Absolute Error: ", "{0:,.2f}".format(mean_absolute_error(y_test, y_pred)))
print("Root Mean Squared Error: ", "{0:,.2f}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
```

### Output
<pre>
Decision Tree Results
R-squared:  0.7179
Mean Absolute Error:  2,091,154.57
Root Mean Squared Error:  3,151,694.65
</pre>

### Distance based algorithms - SVM and KNN
For algorithms such as Support Vector Regressor and K-Nearest Neighbors which are sensitive to distance between points, we have to scale them to a fixed ranged before training them.

```python
# Scaling of features
numeric_col=list(x_train.columns)
x_train_scaled=MinMaxScaler().fit_transform(x_train[numeric_col])
x_test_scaled=MinMaxScaler().fit_transform(x_test[numeric_col])
```

## Support Vector Machine

```python
def objective(trial):
    loss=trial.suggest_categorical('loss',['epsilon_insensitive','squared_epsilon_insensitive'])
    tol=trial.suggest_float('tol',0.001,0.1)
    C=trial.suggest_float('C',0.001,1)
    epsilon=trial.suggest_float('epsilon',0.01,0.1)
    fit_intercept=trial.suggest_categorical('fit_intercept',['True','False'])
    intercept_scaling=trial.suggest_float('intercept_scaling',0.001,10)
    max_iter=trial.suggest_int('max_iter',2,1000)
    random_state = trial.suggest_int('random_state', 1, 30)

    model = LinearSVR(loss=loss, 
                      tol=tol, 
                      C=C, 
                      epsilon=epsilon, 
                      fit_intercept=fit_intercept, 
                      intercept_scaling=intercept_scaling, 
                      max_iter=max_iter,
                      random_state=random_state)

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    r2_scorer = make_scorer(r2_score)
    scores = cross_val_score(model, x_train_scaled, y_train, cv=kfold, scoring=r2_scorer)
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
best_params = study.best_params

SVR=LinearSVR(**best_params)
SVR.fit(x_train_scaled, y_train)
y_pred = SVR.predict(x_test_scaled)
print("Linear SVM Results")
print("R-squared: ", "%.4f" % r2_score(y_test, y_pred))
print("Mean Absolute Error: ", "{0:,.2f}".format(mean_absolute_error(y_test, y_pred)))
print("Root Mean Squared Error: ", "{0:,.2f}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
```

### Output
<pre>
Linear SVM Results
R-squared:  0.6861
Mean Absolute Error:  2,285,553.15
Root Mean Squared Error:  3,324,633.25
</pre>

## K-Nearest Neighbors

```python
def objective(trial):
    n_neighbors = trial.suggest_int('n_neighbors', 1, 20)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
    p = trial.suggest_int('p', 1, 2)
    leaf_size = trial.suggest_int('leaf_size', 1, 50)

    model = KNeighborsRegressor(n_neighbors=n_neighbors, 
                                weights=weights, 
                                algorithm=algorithm,
                                p=p,
                                leaf_size=leaf_size,
                               )

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    r2_scorer = make_scorer(r2_score)
    scores = cross_val_score(model, x_train_scaled, y_train, cv=kfold, scoring=r2_scorer)
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
best_params = study.best_params

KNN=KNeighborsRegressor(**best_params)
KNN.fit(x_train_scaled, y_train)
y_pred = KNN.predict(x_test_scaled)
print("K-Nearest Neighbors Result")
print("R-squared: ", "%.4f" % r2_score(y_test, y_pred))
print("Mean Absolute Error: ", "{0:,.2f}".format(mean_absolute_error(y_test, y_pred)))
print("Root Mean Squared Error: ", "{0:,.2f}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
```

### Output
<pre>
K-Nearest Neighbors Result
R-squared:  0.6969
Mean Absolute Error:  2,118,331.90
Root Mean Squared Error:  3,267,188.23
</pre>

**It seems that the Linear/Ridge Regression model is the ideal model here with the highest R-squared and lowest MAE/RMSE. So we will use it to interpret our results later.**

### Feature Importances
* Contrary to 'our' prior belief, hardware specs are **not** the main factors in determining the laptop prices, but it is instead determined by the **Manufacturer** brands only. 
* From the chart below, we can deduce that the **costliest laptops** are often manufactured by **Razer**, and this is followed by laptops manufactured by **Microsoft**, **Google**, **LG**, and **Huawei**.
* However, this does not mean hardware specifications do not matter at all, as **CPU Speed**, **Category**, and **Weight** still has a minor impact on the prices. 
* **CPU, GPU, Screen**, and **Storage** features do not have a place in determining the prices.

```python
# Plotting
sorted_feature_coefficients = sorted(zip(x_train.columns, LR.coef_), key=lambda x: abs(x[1]), reverse=False)
sorted_feature_names, sorted_coefficients = zip(*sorted_feature_coefficients)
plt.figure(figsize=(10, 15))
plt.barh(sorted_feature_names, sorted_coefficients, color='skyblue')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature Names')
plt.title('Ridge Regression Feature Importances')
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.show()
```

### Output

<img src='https://github.com/jylim21/bear-with-data.github.io/blob/main/laptop-price-prediction/images/8.png' width='800'>

## Ensemble - Random Forest, GBM, LGBM, XGB
Since the previous algorithms does not seem to meet our expectations, if we are merely interested in improving prediction accuracy, we could try out some Ensemble algorithms to improve our predictions:

## Random Forest

```python
def objective(trial):
    # Define the hyperparameters to optimize
    criterion = trial.suggest_categorical('criterion', ['squared_error', 'friedman_mse','absolute_error'])#, 'poisson'
    max_depth = trial.suggest_int('max_depth', 5, 15)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 20)
    #max_features = trial.suggest_categorical('max_features', ['sqrt','log2'])
    random_state = trial.suggest_int('random_state', 1, 30)
    model = RandomForestRegressor(criterion=criterion, 
                                  max_depth=max_depth, 
                                  min_samples_split=min_samples_split, 
                                  min_samples_leaf=min_samples_leaf, 
                                  max_features='sqrt',
                                  random_state=random_state
                                 )

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    r2_scorer = make_scorer(r2_score)
    scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring=r2_scorer)
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5)
best_params = study.best_params

RF=RandomForestRegressor(**best_params)
RF.fit(x_train, y_train)
y_pred = RF.predict(x_test)
print("Random Forest Results")
print("R-squared: ", "%.4f" % r2_score(y_test, y_pred))
print("Mean Absolute Error: ", "{0:,.2f}".format(mean_absolute_error(y_test, y_pred)))
print("Root Mean Squared Error: ", "{0:,.2f}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
```

### Output
<pre>
Random Forest Results
R-squared:  0.7349
Mean Absolute Error:  1,953,317.51
Root Mean Squared Error:  3,055,142.42
</pre>

# Gradient Boost

```python
def objective(trial):
    loss = trial.suggest_categorical('loss', ['squared_error', 'absolute_error', 'huber'])#, 'quantile'
    criterion = trial.suggest_categorical('criterion', ['squared_error', 'friedman_mse'])
    n_estimators = trial.suggest_int('n_estimators', 5, 50)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 15)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 20)
    max_features = trial.suggest_categorical('max_features', ['auto','sqrt','log2'])
    #random_state = trial.suggest_int('random_state', 1, 30)

    model = GradientBoostingRegressor(loss=loss, 
                                      criterion=criterion, 
                                      max_depth=max_depth, 
                                      min_samples_split=min_samples_split, 
                                      min_samples_leaf=min_samples_leaf, 
                                      max_features=max_features,
                                      random_state=14
                                     )

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    r2_scorer = make_scorer(r2_score)
    scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring=r2_scorer)
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5)
best_params = study.best_params

GB=GradientBoostingRegressor(**best_params)
GB.fit(x_train, y_train)
y_pred = GB.predict(x_test)
print("Gradient Boosting Results")
print("R-squared: ", "%.4f" % r2_score(y_test, y_pred))
print("Mean Absolute Error: ", "{0:,.2f}".format(mean_absolute_error(y_test, y_pred)))
print("Root Mean Squared Error: ", "{0:,.2f}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
```

### Output
<pre>
Gradient Boosting Results
R-squared:  0.7208
Mean Absolute Error:  1,917,422.00
Root Mean Squared Error:  3,135,782.16
</pre>

## Light GBM

```python
def objective(trial):
    num_leaves= trial.suggest_int('num_leaves', 2, 50)
    learning_rate= trial.suggest_float('learning_rate', 0.001, 0.1)
    #feature_fraction= trial.suggest_float('feature_fraction', 0.1, 1.0)
    #bagging_fraction= trial.suggest_float('bagging_fraction', 0.1, 1.0)
    reg_alpha= trial.suggest_float('reg_alpha', 0.0, 1.0)
    reg_lambda= trial.suggest_float('reg_lambda', 0.0, 1.0)
    random_state= trial.suggest_int('random_state', 0, 42)
        
    model = LGBMRegressor(num_leaves=num_leaves, 
                         learning_rate=learning_rate,
                         #feature_fraction=feature_fraction, 
                         #bagging_fraction=bagging_fraction, 
                         reg_alpha=reg_alpha,
                         reg_lambda=reg_lambda,
                         random_state=random_state
                        )

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    r2_scorer = make_scorer(r2_score)
    scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring=r2_scorer)
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5)
best_params = study.best_params

LG=LGBMRegressor(**best_params)
LG.fit(x_train, y_train)
y_pred = LG.predict(x_test)
print("LightGBM Result")
print("R-squared: ", "%.4f" % r2_score(y_test, y_pred))
print("Mean Absolute Error: ", "{0:,.2f}".format(mean_absolute_error(y_test, y_pred)))
print("Root Mean Squared Error: ", "{0:,.2f}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
```

### Output
<pre>
LightGBM Result
R-squared:  0.7384
Mean Absolute Error:  1,909,419.70
Root Mean Squared Error:  3,035,286.42
</pre>

## XGBoost

```python
def objective(trial):
    eta = trial.suggest_float('eta', 0.01,0.9)
    #min_child_weights = trial.suggest_float('min_child_weights', 0.1, 1)
    max_depth = trial.suggest_int('max_depth', 5, 20)
    gamma=trial.suggest_float('gamma',0.5,1)
    subsample = trial.suggest_float('subsample', 0.2,1)
    #random_state = trial.suggest_int('random_state', 1, 30)
    model = XGBRegressor(eta=eta, 
                         #min_child_weights=min_child_weights,
                         max_depth=max_depth, 
                         gamma=gamma, 
                         subsample=subsample,
                         reg_lambda=2,
                         random_state=23
                        )

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    r2_scorer = make_scorer(r2_score)
    scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring=r2_scorer)
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5)
best_params = study.best_params

XG=XGBRegressor(**best_params)
XG.fit(x_train, y_train)
y_pred = XG.predict(x_test)
print("XGBoost Results")
print("R-squared: ", "%.4f" % r2_score(y_test, y_pred))
print("Mean Absolute Error: ", "{0:,.2f}".format(mean_absolute_error(y_test, y_pred)))
print("Root Mean Squared Error: ", "{0:,.2f}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
```

### Output
<pre>
XGBoost Results
R-squared:  0.7771
Mean Absolute Error:  1,805,740.25
Root Mean Squared Error:  2,801,565.01
</pre>

All ensemble algorithms indeed yielded a better performance than the previous models, with **XGBoost** being the best performer here with **77% R-squared** and **2.8E6 RMSE**.
