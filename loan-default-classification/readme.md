<h1 align="center">Loan Default Classification</h1>

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/loan-default-classification/images/loan-default.JPG?raw=true)

Managing loan defaults is a major risk concern for financial institutions, with defaults leading to loss of capital and strained balance sheets. Identifying potentially delinquent accounts early allows preventive action to avoid full defaults. However, picking these high-default-risk accounts from millions of loans is challenging without robust analytical capabilities.

This project aims to leverage machine learning techniques to analyze past loan data and build predictive models for loan default risk. By examining attributes like loan amount, term, debt-to-income, and repayment history patterns, the models can predict defaults and prioritize accounts requiring intervention.

Operationalizing these data-driven default predictions can help banks substantially lower their overall default rates and maintain healthy loan books. The models can also provide insights into the core drivers of delinquency to guide policy reforms and prevent future high-risk lending.

## THE PROJECT

The dataset used here is the [Loan Classification Dataset](https://www.kaggle.com/datasets/abhishek14398/loan-dataset) by *ALLENA VENKATA SAI ABY*, the free version allows users to export daily data on temperature, humidity, wind, solar, and precipitation etc. for different location with a daily limit of 1000 rows. Since this dataset contains data from July 2016 to Dec 2023, it took me 3 days to get the data required.

It consists of a portfolio of approved loans which were either Fully Paid, Charged Off, or still ongoing (current) with no certain outcome. Our objective is to predict the outcomes of these current loans which would give the Loan Commissioner a probabilistic estimate of recoveries on the current portfolio, and hopefully, a way to screen borrowers based on their initial details. 
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
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

# First Glance
This dataset contains a mixture of closed loan cases (both fully paid and defaulted) and open (current) loans. We will separate the closed loans from the current ongoing loans and perform analysis on the closed cases beforehand.

```python
df= pd.read_csv("/kaggle/input/loan-dataset/loan.csv")
df_test=df[df['loan_status'] == 'Current']
df=df[df['loan_status'] != 'Current']
df.head(5)
```

### Output
<pre>
<table border="0" class="dataframe">
  <tbody>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>member_id</th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>...</th>
      <th>num_tl_90g_dpd_24m</th>
      <th>num_tl_op_past_12m</th>
      <th>pct_tl_nvr_dlq</th>
      <th>percent_bc_gt_75</th>
      <th>pub_rec_bankruptcies</th>
      <th>tax_liens</th>
      <th>tot_hi_cred_lim</th>
      <th>total_bal_ex_mort</th>
      <th>total_bc_limit</th>
      <th>total_il_high_credit_limit</th>
    </tr>
    <tr>
      <th>0</th>
      <td>1077501</td>
      <td>1296599</td>
      <td>5000</td>
      <td>5000</td>
      <td>4975.0</td>
      <td>36 months</td>
      <td>10.65%</td>
      <td>162.87</td>
      <td>B</td>
      <td>B2</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1077430</td>
      <td>1314167</td>
      <td>2500</td>
      <td>2500</td>
      <td>2500.0</td>
      <td>60 months</td>
      <td>15.27%</td>
      <td>59.83</td>
      <td>C</td>
      <td>C4</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1077175</td>
      <td>1313524</td>
      <td>2400</td>
      <td>2400</td>
      <td>2400.0</td>
      <td>36 months</td>
      <td>15.96%</td>
      <td>84.33</td>
      <td>C</td>
      <td>C5</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1076863</td>
      <td>1277178</td>
      <td>10000</td>
      <td>10000</td>
      <td>10000.0</td>
      <td>36 months</td>
      <td>13.49%</td>
      <td>339.31</td>
      <td>C</td>
      <td>C1</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1075269</td>
      <td>1311441</td>
      <td>5000</td>
      <td>5000</td>
      <td>5000.0</td>
      <td>36 months</td>
      <td>7.90%</td>
      <td>156.46</td>
      <td>A</td>
      <td>A4</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
32950 rows × 5 columns
</pre>

## Labels
There are relatively more cases of loan been paid off in full (32950) than loan defaults (5627), this dataset is **imbalanced**. 

Although it is good news for the LC but it is no good for me as a Data Scientist for this project as ML algorithms would have a hard time identifying the default cases. We will have to upsample the default cases later on using SMOTE.

```python
df['loan_status'].value_counts()
```

### Output
<pre>
Fully Paid     32950
Charged Off     5627
Name: loan_status, dtype: int64
</pre>

# Data Cleansing
There are way too many features (111) to analyse this dataset effectively, but we will begin with 
<ol>
<li> Identifying and removing vague columns which presents little to no value to our analysis:</li>
  <ul>
  <li> <b>Totally-unique columns</b> eg. id, member_id, url </li>
  <li> <b>Columns having only 1 unique value / all zero and NaNs</b> </li>
  <li> <b>'Recovery' columns</b> because it would introduce data leakage eg. <i>recoveries,collection_recovery_fee,total_rec_late_fee</i> </li>
</ul>
<li> Converting numerical columns which were wrongly identified as 'object' data type</li>

<li> Creating new loan credit ratio metrics from existing features:</li>
<ul>
<li> <b>Loan duration</b> = last payment date - issue date <i>(not to be confused with 'term' as defaulted loans should be closed before the term ends)</i> </li>
<li> <b>Applied amount vs actual amount</b> = funded amount / loan amount</li>
<li> <b>Investor Participation ratios</b> </li>
<li> <b>Principal to Interest Ratio</b> = total received principal / total received interest </li>
<li> <b>Open Credit Account percentage</b> = open accounts / total accounts </li>
<li> <b>Duration between credit pulls</b> = last credit pull date - first credit pull date </li>
</ul>
</ol>
