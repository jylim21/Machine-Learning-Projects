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

<details>
<summary>View Code</summary>

```python
# Removing columns having only 1 unique value
df.drop(df.columns[49:105],axis=1, inplace=True)
df.drop(df.columns[50:55],axis=1, inplace=True)
df.drop(['out_prncp','out_prncp_inv'], axis=1, inplace=True)

# Remove recovery columns which would introduce data leakage (because only default loans will have recoveries)
df.drop(['recoveries','collection_recovery_fee','total_rec_late_fee'], axis=1, inplace=True)

# Removing columns which are either duplicates of other columns, or a waste of time to process
df.drop(['id','member_id','emp_title','url','pymnt_plan','desc','title','addr_state','zip_code','initial_list_status','mths_since_last_delinq','mths_since_last_record','next_pymnt_d'], axis=1, inplace=True)

# Transforming string/object columns into a numerical format
df['term']=df['term'].replace({' 36 months':36, ' 60 months':60}).astype(int)
df['grade']=df['grade'].replace({'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6})
df['sub_grade']=df['sub_grade'].str[1].astype(int)
df['emp_length']=df['emp_length'].str.replace(r' year[s]?','', regex=True)
df['emp_length']=df['emp_length'].replace({'10+':10,'< 1':0.5,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10+':10})
df['source_verified'] = df['verification_status'].apply(lambda x: 1 if x == 'Source Verified' else 0)
df['income_verified'] = df['verification_status'].apply(lambda x: 0 if x == 'Not Verified' else 1)
df['int_rate']=df['int_rate'].str.replace('%','').astype(float)
df['revol_util']=df['revol_util'].str.replace('%','').astype(float)
df['loan_status']=df['loan_status'].replace({'Fully Paid':0, 'Charged Off':1}).astype(int)
df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%y')
df['last_credit_pull_d'] = pd.to_datetime(df['last_credit_pull_d'], format='%b-%y')
df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%y')
df['last_pymnt_d'] = pd.to_datetime(df['last_pymnt_d'], format='%b-%y')

# Treating missing values and generating new features
df['last_credit_pull_d']=df.apply(lambda row: row['issue_d'] + pd.DateOffset(months=36) if pd.isna(row['last_credit_pull_d']) else row['last_credit_pull_d'], axis=1)
df['loan_duration']=round((df['last_pymnt_d']-df['issue_d'])/np.timedelta64(1,'M'))
df['loan_duration']=df.apply(lambda row: (df[df['loan_status']==1]['loan_duration']).median() if pd.isna(row['loan_duration']) else row['loan_duration'], axis=1)
df['applied_funded_perc']=df['funded_amnt']/df['loan_amnt']
df['inv_funded_perc']=df['funded_amnt_inv']/df['funded_amnt']
df['inv_total_perc']=df['total_pymnt_inv']/df['total_pymnt']
df['loan_prncp_perc']=df['total_rec_prncp']/(df['total_rec_prncp']+df['total_rec_int'])
df['loan_prncp_perc']=df.apply(lambda row: (df[df['loan_status']==1]['loan_prncp_perc']).mean() if pd.isna(row['loan_prncp_perc']) else row['loan_prncp_perc'], axis=1)
df['open_acc_perc']=df['open_acc']/df['total_acc']
df['issue_to_credit_pull']=round((df['last_credit_pull_d']-df['issue_d'])/np.timedelta64(1,'M'))
df['pub_rec_bankruptcies']=df.apply(lambda row: 0 if row['pub_rec'] == 0 else 1 if pd.isna(row['pub_rec_bankruptcies']) else row['pub_rec_bankruptcies'], axis=1)
df['revol_util']=df['revol_util'].fillna(0)
df['inv_total_perc']=df.apply(lambda row: row['inv_funded_perc'] if pd.isna(row['inv_total_perc']) else row['inv_total_perc'], axis=1)
df.drop(['emp_length','verification_status','issue_d','earliest_cr_line','last_pymnt_d','last_credit_pull_d'], axis=1, inplace=True)
```
</details>

# EDA

## Univariate Analysis

<details>
<summary>View Code</summary>

```python
fig, axes = plt.subplots(nrows=8, ncols=4, figsize=(12,20))
fig.tight_layout()
for i, column in enumerate(df.drop(['home_ownership','purpose','loan_status'], axis=1).columns):
    sns.histplot(df[column],ax=axes[i//4,i%4])
```
### Output
</details>

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/loan-default-classification/images/1.png?raw=true)

## Loan Duration Comparison

<details>
<summary>View Code</summary>

```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
sns.boxplot(data=df[df['term']==36], x='loan_duration',y='loan_status', orient='h',ax=axes[0]).set(title='Loan Duration (36 months)')
sns.boxplot(data=df[df['term']==60], x='loan_duration',y='loan_status', orient='h',ax=axes[1]).set(title='Loan Duration (60 months)')
```
### Output
</details>

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/loan-default-classification/images/2.png?raw=true)

# BONUS: Dimensionality Reduction using PCA
The dimensionality problem caused by the enormous number of features (111 again) can be effectively mitigated by using Principal Component Analysis (PCA). But first, all of the features were normalized using sklearn's StandardScaler because distance algorithms such as the PCA can only work well with scaled features.

<details>
<summary>View Code</summary>

```python
scaler = StandardScaler()
df_norm=pd.DataFrame(scaler.fit_transform(df.drop(['term','loan_status','source_verified','income_verified','home_ownership','purpose'], axis=1)))
nums = np.arange(20)
var_ratio = []
for num in nums:
  pca = PCA(n_components=num)
  pca.fit(df_norm)
  var_ratio.append(np.sum(pca.explained_variance_ratio_))

plt.figure(figsize=(6,5))
plt.grid()
plt.plot(nums,var_ratio,marker='o')
plt.xlabel('n_components')
plt.ylabel('Explained variance ratio')
plt.title('n_components vs. Explained Variance Ratio')
```
### Output
</details>

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/loan-default-classification/images/3.png?raw=true)

According to the graph, the greatest increase in variance was explained by the first principal component (pc1), although it is still increasing with each PCs but the effect gets smaller with increasing PCs.

We should probably take the first 6 principal components and see what each of them represents, using a **correlation** plot.

<details>
<summary>View Code</summary>

```python
pca=PCA(n_components=6)
pc = pca.fit_transform(df_norm)
pcadf = pd.DataFrame(data = pc, columns = ['pc1', 'pc2','pc3','pc4','pc5','pc6'])
df = pd.concat([df, pcadf.set_index(df.index)], axis = 1)

plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".1f", linewidths=0.5)
plt.show()
```
### Output
</details>

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/loan-default-classification/images/4.png?raw=true)

And by zooming into the 6 principal components:

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/loan-default-classification/images/8.png?raw=true)

According to the correlation plot, each principal component represents the following:
* **PC1** - loan repayment amounts
* **PC2** - loan attributes (interest, principle, grade)
* **PC3** - % of investor contributions (just 'inv_funded_perc' & 'inv_total_perc')
* **PC4** - loan accounts
* **PC5** - public deragotary records (just 'pub_rec' and 'pub_rec_bankruptcies')

The first principal component (PC1) did a great job by combining at least 8 features into 1, PC2 also grouped the loan attributes well.

However, everything after PC2 is probably not so useful as they represent just about 2 features in each component, therefore we should probably just take the first 2 principal components **PC1** & **PC2** if necessary. However in this project we will not be using these principal components.

# Feature Scaling & Encoding
Scaling the dataset usually gives us a boost in model performance, this can be achieved using the MinMaxScaler from sklearn.

Categorical variables will also be one-hot encoded, which is assigning a binary numerical value for each data attribute present.

*NOTE: I have created a copy of the Scaler to be applied to the list of current/open loans in the further part of this project.*

```python
scaler_copy = MinMaxScaler()
df_copy = pd.DataFrame(scaler_copy.fit_transform(df[['revol_util','int_rate','funded_amnt','grade','sub_grade','issue_to_credit_pull','total_acc','open_acc','dti','loan_prncp_perc','term']]), columns=df[['revol_util','int_rate','funded_amnt','grade','sub_grade','issue_to_credit_pull','total_acc','open_acc','dti','loan_prncp_perc','term']].columns)

df = pd.get_dummies(df, columns = ['home_ownership','purpose'])
df.drop(['purpose_other','home_ownership_OTHER'], axis=1, inplace=True)
df.drop(['loan_amnt','funded_amnt_inv','installment','total_pymnt_inv','total_rec_prncp','total_rec_int'], axis=1, inplace=True)
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df.drop(['inv_total_perc','pub_rec_bankruptcies','total_pymnt','last_pymnt_amnt','loan_duration'], axis=1, inplace=True)
df.head()
```

### Output
<pre>
<table border="0" class="dataframe">
  <tbody>
    <tr style="text-align: right;">
      <th></th>
      <th>funded_amnt</th>
      <th>term</th>
      <th>int_rate</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>annual_inc</th>
      <th>loan_status</th>
      <th>dti</th>
      <th>delinq_2yrs</th>
      <th>inq_last_6mths</th>
      <th>...</th>
      <th>purpose_educational</th>
      <th>purpose_home_improvement</th>
      <th>purpose_house</th>
      <th>purpose_major_purchase</th>
      <th>purpose_medical</th>
      <th>purpose_moving</th>
      <th>purpose_renewable_energy</th>
      <th>purpose_small_business</th>
      <th>purpose_vacation</th>
      <th>purpose_wedding</th>
    </tr>
    <tr>
      <th>0</th>
      <td>0.130435</td>
      <td>0.0</td>
      <td>0.275553</td>
      <td>0.166667</td>
      <td>0.25</td>
      <td>0.003336</td>
      <td>0.0</td>
      <td>0.921974</td>
      <td>0.0</td>
      <td>0.125</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.057971</td>
      <td>1.0</td>
      <td>0.518967</td>
      <td>0.333333</td>
      <td>0.75</td>
      <td>0.004336</td>
      <td>1.0</td>
      <td>0.033344</td>
      <td>0.0</td>
      <td>0.625</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.055072</td>
      <td>0.0</td>
      <td>0.555321</td>
      <td>0.333333</td>
      <td>1.00</td>
      <td>0.001376</td>
      <td>0.0</td>
      <td>0.290764</td>
      <td>0.0</td>
      <td>0.250</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>3</th>
      <td>0.275362</td>
      <td>0.0</td>
      <td>0.425184</td>
      <td>0.333333</td>
      <td>0.00</td>
      <td>0.007538</td>
      <td>0.0</td>
      <td>0.666889</td>
      <td>0.0</td>
      <td>0.125</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.130435</td>
      <td>0.0</td>
      <td>0.130664</td>
      <td>0.000000</td>
      <td>0.75</td>
      <td>0.005337</td>
      <td>0.0</td>
      <td>0.373458</td>
      <td>0.0</td>
      <td>0.375</td>
      <td>...</td>
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
    </tr>
  </tbody>
</table>
5 rows × 41 columns
</pre>

# Model Training

The data will be split into a 80% train & 20% test set.

Due to the imbalance in outcomes, additional samples will be generated for the minority class (outcome 1-default) using **SMOTE**.

The train set will be **10-fold cross-validated**, optimized for the best parameters using **Optuna** before finally evaluating against the test set.

For this binary classification problem, our baseline algorithm would be the **logistic regressor** where other subsequent algorithms will be benchmarked against this baseline algorithm.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, make_scorer, roc_curve#, rac_scorer
import optuna
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
optuna.logging.set_verbosity(optuna.logging.WARNING)
```

## Baseline Algorithm - Logistic Regression

```python
X=df.drop('loan_status', axis=1)
y=df['loan_status']

x_train, x_testval, y_train, y_testval = train_test_split(X, y, test_size=0.4, random_state=2)
x_val, x_test, y_val, y_test = train_test_split(x_testval, y_testval, test_size=0.5, random_state=2)

# over = SMOTE(sampling_strategy=0.8, random_state=42)
# under = RandomUnderSampler(sampling_strategy=0.9, random_state=42)
# smt = Pipeline(steps=[('o', over), ('u', under)])
# x_train, y_train=smt.fit_resample(x_train, y_train)

LR=LogisticRegression(random_state=42, penalty='l2', solver='newton-cholesky')
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
<pre>
Accuracy:  0.9655
Precision:  0.9957
Recall:  0.7813
F1-Score:  0.8756
array([[6513,    4],
       [ 262,  936]])
</pre>

The Logistic Regressor gave us an whooping **96.5%** prediction accuracy, seems quite impressive for this simple algorithm, or it isn't? 

Notice that we have a considerable number of false negatives (180) but no false positives at all, this could meant that the algo did **not** manage to capture the minority class outcome. The false negatives here represents cases where the loans default when we think it will be paid off, these wrong assumptions could incur great loss to the LC as compared to the opposite which is false positives.

Enough said, accuracy isn't the main consideration here, we should opt for models with a **high recall** as well.

### **Feature Importances**

The advantage of using the Logistic Regressor is that we are able to identify which feature has a greater influence in the final predicted outcome.

By plotting the coefficients, 

*(You should probably scroll right to the bottom of the image)*

<details>
<summary>View Code</summary>

```python
coeff=pd.DataFrame(zip(x_train.columns, np.transpose(LR.coef_.flatten())), columns=['features', 'coef']).sort_values(by='coef')
plt.figure(figsize=(10, 15))
plt.barh(coeff['features'], coeff['coef'], color='skyblue')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Logistic Regression Feature Importances')
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.show()
```
### Output
</details>

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/loan-default-classification/images/6.png?raw=true)

Recalling that *0 = paid in full* and *1 = default*, features with a more positive coefficient tend to increase the probability of loan default, whereas features with negative coefficients will decrease the likelihood of loan defaults.

According to the plot, there weren't any features which would increase the probability of default, but there were highly-negative coefficients such as **Principle to Interest %**, **interest rate**, **term** and **loan grade** which has a negative influence on the probability of default.

In layman terms, this model shows that loans which defaulted tend to have lower principal to interest %, lower interest rate, and a shorter term.

But is this the case? We can try to optimize this baseline model by trying different combination of features.

## Logistic Regression (Enhanced)

Besides the 4 main features mentioned above, another model was created with the addition of other credit-related features such as 
* debt-to-income ratio (dti)
* revolving utilization
* funded amount
* total & open accounts
* duration from issue date to last credit pull date in months

<details>
<summary>View Code</summary>
  
```python
X=df[['revol_util','int_rate','funded_amnt','grade','sub_grade',
    'issue_to_credit_pull','total_acc','open_acc','dti','loan_prncp_perc','term']]
y=df['loan_status']

x_train, x_testval, y_train, y_testval = train_test_split(X, y, test_size=0.4, random_state=2)
x_val, x_test, y_val, y_test = train_test_split(x_testval, y_testval, test_size=0.5, random_state=2)

over = SMOTE(sampling_strategy=0.8, random_state=42)
under = RandomUnderSampler(sampling_strategy=0.9, random_state=42)
smt = Pipeline(steps=[('o', over), ('u', under)])
x_train, y_train=smt.fit_resample(x_train, y_train)

LR=LogisticRegression(random_state=42, penalty='l2', solver='newton-cholesky')
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
Accuracy:  0.9857
Precision:  0.9936
Recall:  0.9140
F1-Score:  0.9522
array([[6510,    7],
       [ 103, 1095]])
</pre>

This model managed to improve the recall by **13%** which also brought up the F1-score by 7%, this is quite an improvement from the baseline model but it could also subject to overfitting. To find out if it really overfitted, let's check it's performance on the test set this time:

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
Accuracy:  0.9878
Precision:  0.9980
Recall:  0.9150
F1-Score:  0.9547
array([[6632,    2],
       [  92,  990]])
</pre>

The model retains its performance despite being used on unseen data, this is definitely an improvement over the baseline model.

Let's see how other classification algorithms perform:

## Gaussian Naive Bayes

<details>
<summary>View Code</summary>

```python
X=df[['revol_util','funded_amnt',
    'issue_to_credit_pull','dti','loan_prncp_perc']]#,'loan_duration','total_pymnt','last_pymnt_amnt'
y=df['loan_status']

x_train, x_testval, y_train, y_testval = train_test_split(X, y, test_size=0.4, random_state=2)
x_val, x_test, y_val, y_test = train_test_split(x_testval, y_testval, test_size=0.5, random_state=2)

over = SMOTE(sampling_strategy=0.9, random_state=42)
under = RandomUnderSampler(sampling_strategy=1, random_state=42)
smt = Pipeline(steps=[('o', over), ('u', under)])
x_train, y_train=smt.fit_resample(x_train, y_train)

NB=GaussianNB()
NB.fit(x_train,y_train)
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
Accuracy:  0.8747
Precision:  0.5381
Recall:  0.7514
F1-Score:  0.6271
array([[5936,  698],
       [ 269,  813]])
</pre>

## Decision Tree

<details>
<summary>View Code</summary>

```python
X=df[['revol_util','int_rate','grade','funded_amnt','term',
    'issue_to_credit_pull','dti','sub_grade','loan_prncp_perc']]#,'loan_duration','total_pymnt','last_pymnt_amnt'
y=df['loan_status']

x_train, x_testval, y_train, y_testval = train_test_split(X, y, test_size=0.4, random_state=2)
x_val, x_test, y_val, y_test = train_test_split(x_testval, y_testval, test_size=0.5, random_state=2)

over = SMOTE(sampling_strategy=0.6, random_state=42)
under = RandomUnderSampler(sampling_strategy=0.9, random_state=42)
smt = Pipeline(steps=[('o', over), ('u', under)])
x_train, y_train=smt.fit_resample(x_train, y_train)

def objective(trial):
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy','log_loss'])
    splitter = trial.suggest_categorical('splitter', ['best','random'])
    max_depth = trial.suggest_int('max_depth', 2, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 30)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 20)
    max_features = trial.suggest_categorical('max_features', ['sqrt','log2'])

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

DT=DecisionTreeClassifier(**best_params, random_state=42)
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
Accuracy:  0.9811
Precision:  0.9309
Recall:  0.9344
F1-Score:  0.9327
array([[6559,   75],
       [  71, 1011]])
</pre>

## K-Nearest Neighbors

<details>
<summary>View Code</summary>

```python
X=df[['revol_util','int_rate','grade','funded_amnt',
    'issue_to_credit_pull','total_acc','open_acc','dti','sub_grade','loan_prncp_perc']]#,'loan_duration','total_pymnt','last_pymnt_amnt'
y=df['loan_status']

x_train, x_testval, y_train, y_testval = train_test_split(X, y, test_size=0.4, random_state=2)
x_val, x_test, y_val, y_test = train_test_split(x_testval, y_testval, test_size=0.5, random_state=2)

# over = SMOTE(sampling_strategy=0.7, random_state=42)
# under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
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
Accuracy:  0.9158
Precision:  0.9675
Recall:  0.4131
F1-Score:  0.5790
array([[6619,   15],
       [ 635,  447]])
</pre>

## Multi-level Perceptron (MLP)

<details>
<summary>View Code</summary>
  
```python
X=df[['revol_util','int_rate','grade','funded_amnt','term',
    'issue_to_credit_pull','total_acc','open_acc','dti','sub_grade','loan_prncp_perc']]#,'loan_duration','total_pymnt','last_pymnt_amnt'
y=df['loan_status']

x_train, x_testval, y_train, y_testval = train_test_split(X, y, test_size=0.4, random_state=2)
x_val, x_test, y_val, y_test = train_test_split(x_testval, y_testval, test_size=0.5, random_state=2)

over = SMOTE(sampling_strategy=0.8, random_state=42)
under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
smt = Pipeline(steps=[('o', over), ('u', under)])
x_train, y_train=smt.fit_resample(x_train, y_train)

MLP = MLPClassifier(hidden_layer_sizes=(),activation='logistic', solver='lbfgs')
MLP.fit(x_train, y_train)
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
Accuracy:  0.9936
Precision:  0.9952
Recall:  0.9593
F1-Score:  0.9769
array([[6629,    5],
       [  44, 1038]])
</pre>

## Sequential NN

<details>
<summary>View Code</summary>
  
```python
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import BinaryCrossentropy

X=df[['revol_util','int_rate','grade','funded_amnt','term',
    'issue_to_credit_pull','total_acc','open_acc','dti','sub_grade','loan_prncp_perc']]#,'loan_duration','total_pymnt','last_pymnt_amnt'
y=df['loan_status']

x_train, x_testval, y_train, y_testval = train_test_split(X, y, test_size=0.4, random_state=2)
x_val, x_test, y_val, y_test = train_test_split(x_testval, y_testval, test_size=0.5, random_state=2)

over = SMOTE(sampling_strategy=0.9, random_state=42)
under = RandomUnderSampler(sampling_strategy=1, random_state=42)
smt = Pipeline(steps=[('o', over), ('u', under)])
x_train, y_train=smt.fit_resample(x_train, y_train)

# Creating model using the Sequential in tensorflow
NN = Sequential()
NN.add(Dense(512, activation='relu', input_dim=11))
NN.add(Dropout(0.2))
NN.add(Dense(1, activation='sigmoid'))
NN.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
hist = NN.fit(x_train, y_train, epochs=30, batch_size=10, validation_split=0.2)
NN.summary()
```
### Output

<pre>
Epoch 1/30
2851/2851 [==============================] - 7s 2ms/step - loss: 0.5784 - accuracy: 0.6988 - val_loss: 0.7621 - val_accuracy: 0.4673
Epoch 2/30
2851/2851 [==============================] - 6s 2ms/step - loss: 0.4982 - accuracy: 0.7643 - val_loss: 0.7374 - val_accuracy: 0.5171
Epoch 3/30
2851/2851 [==============================] - 5s 2ms/step - loss: 0.4183 - accuracy: 0.8295 - val_loss: 0.5825 - val_accuracy: 0.6580
Epoch 4/30
2851/2851 [==============================] - 6s 2ms/step - loss: 0.3356 - accuracy: 0.8846 - val_loss: 0.4345 - val_accuracy: 0.7709
Epoch 5/30
2851/2851 [==============================] - 6s 2ms/step - loss: 0.2699 - accuracy: 0.9189 - val_loss: 0.3378 - val_accuracy: 0.8357
Epoch 6/30
2851/2851 [==============================] - 6s 2ms/step - loss: 0.2243 - accuracy: 0.9358 - val_loss: 0.2436 - val_accuracy: 0.8956
Epoch 7/30
2851/2851 [==============================] - 6s 2ms/step - loss: 0.1928 - accuracy: 0.9473 - val_loss: 0.3451 - val_accuracy: 0.8210
Epoch 8/30
2851/2851 [==============================] - 6s 2ms/step - loss: 0.1693 - accuracy: 0.9535 - val_loss: 0.2476 - val_accuracy: 0.8850
Epoch 9/30
2851/2851 [==============================] - 5s 2ms/step - loss: 0.1523 - accuracy: 0.9594 - val_loss: 0.2331 - val_accuracy: 0.8902
Epoch 10/30
2851/2851 [==============================] - 6s 2ms/step - loss: 0.1396 - accuracy: 0.9625 - val_loss: 0.1917 - val_accuracy: 0.9144
Epoch 11/30
2851/2851 [==============================] - 6s 2ms/step - loss: 0.1288 - accuracy: 0.9655 - val_loss: 0.1791 - val_accuracy: 0.9205
Epoch 12/30
2851/2851 [==============================] - 6s 2ms/step - loss: 0.1216 - accuracy: 0.9674 - val_loss: 0.1262 - val_accuracy: 0.9487
Epoch 13/30
2851/2851 [==============================] - 6s 2ms/step - loss: 0.1146 - accuracy: 0.9697 - val_loss: 0.1588 - val_accuracy: 0.9276
Epoch 14/30
2851/2851 [==============================] - 6s 2ms/step - loss: 0.1085 - accuracy: 0.9709 - val_loss: 0.1296 - val_accuracy: 0.9449
Epoch 15/30
2851/2851 [==============================] - 6s 2ms/step - loss: 0.1041 - accuracy: 0.9727 - val_loss: 0.1663 - val_accuracy: 0.9240
Epoch 16/30
2851/2851 [==============================] - 6s 2ms/step - loss: 0.1000 - accuracy: 0.9732 - val_loss: 0.1270 - val_accuracy: 0.9451
Epoch 17/30
2851/2851 [==============================] - 6s 2ms/step - loss: 0.0965 - accuracy: 0.9743 - val_loss: 0.1336 - val_accuracy: 0.9428
Epoch 18/30
2851/2851 [==============================] - 6s 2ms/step - loss: 0.0934 - accuracy: 0.9753 - val_loss: 0.1126 - val_accuracy: 0.9531
Epoch 19/30
2851/2851 [==============================] - 6s 2ms/step - loss: 0.0906 - accuracy: 0.9761 - val_loss: 0.0977 - val_accuracy: 0.9614
Epoch 20/30
2851/2851 [==============================] - 6s 2ms/step - loss: 0.0881 - accuracy: 0.9761 - val_loss: 0.1198 - val_accuracy: 0.9496
Epoch 21/30
2851/2851 [==============================] - 6s 2ms/step - loss: 0.0868 - accuracy: 0.9769 - val_loss: 0.1151 - val_accuracy: 0.9516
Epoch 22/30
2851/2851 [==============================] - 6s 2ms/step - loss: 0.0838 - accuracy: 0.9778 - val_loss: 0.1020 - val_accuracy: 0.9589
Epoch 23/30
2851/2851 [==============================] - 6s 2ms/step - loss: 0.0821 - accuracy: 0.9788 - val_loss: 0.1158 - val_accuracy: 0.9520
Epoch 24/30
2851/2851 [==============================] - 6s 2ms/step - loss: 0.0808 - accuracy: 0.9787 - val_loss: 0.0817 - val_accuracy: 0.9680
Epoch 25/30
2851/2851 [==============================] - 6s 2ms/step - loss: 0.0793 - accuracy: 0.9793 - val_loss: 0.1027 - val_accuracy: 0.9583
Epoch 26/30
2851/2851 [==============================] - 6s 2ms/step - loss: 0.0779 - accuracy: 0.9794 - val_loss: 0.1023 - val_accuracy: 0.9581
Epoch 27/30
2851/2851 [==============================] - 6s 2ms/step - loss: 0.0771 - accuracy: 0.9803 - val_loss: 0.0926 - val_accuracy: 0.9620
Epoch 28/30
2851/2851 [==============================] - 6s 2ms/step - loss: 0.0762 - accuracy: 0.9804 - val_loss: 0.0982 - val_accuracy: 0.9609
Epoch 29/30
2851/2851 [==============================] - 6s 2ms/step - loss: 0.0748 - accuracy: 0.9804 - val_loss: 0.1009 - val_accuracy: 0.9597
Epoch 30/30
2851/2851 [==============================] - 6s 2ms/step - loss: 0.0741 - accuracy: 0.9812 - val_loss: 0.0991 - val_accuracy: 0.9606
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 512)               6144      
                                                                 
 dropout (Dropout)           (None, 512)               0         
                                                                 
 dense_1 (Dense)             (None, 1)                 513       
                                                                 
=================================================================
Total params: 6,657
Trainable params: 6,657
Non-trainable params: 0
_________________________________________________________________
</pre>
</details>

And by running it on the test data,

<details>
<summary>View Code</summary>
  
```python
y_pred = NN.predict(x_test) >0.5
y_pred_proba = NN.predict_proba(x_test)[::,1]
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
242/242 [==============================] - 0s 1ms/step
Accuracy:  0.9903
Precision:  0.9970
Recall:  0.9335
F1-Score:  0.9642
array([[6631,    3],
       [  72, 1010]])
</pre>

# Model Evaluation - ROC

<details>
<summary>View Code</summary>
  
```python
plt.plot(LR_fp,LR_tp, color='blue', label='Logistic Regression')
plt.plot(NB_fp,NB_tp, color='red', label='Naive Bayes')
plt.plot(DT_fp,DT_tp, color='cyan', label='Decision Tree')
plt.plot(KN_fp,KN_tp, color='green', label='K Neighbors')
plt.plot(MLP_fp,MLP_tp, color='grey', label='MLP')
plt.plot(NN_fp,NN_tp, color='magenta', label='Sequential NN')
plt.plot([0,1],[0,1],'--', color='black')
plt.title('ROC Curve Comparison')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.ylim(0,1)
plt.xlim(0,1)
plt.legend(loc=4)
plt.show()
```
### Output
</details>

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/loan-default-classification/images/7.png?raw=true)

Surprisingly, the Logistic Regressor performs on par with the Neural Network algorithms, there is no doubt that we would choose the Logistic Regressor in this case.

The Decision Tree classfier also performed impressively, but still lacks behind slightly in terms of accuracy and recall (far from the top border).

Therefore, the **enhanced Logistic Regressor** model comes out as the winner in terms of both performance and interpretability in this case.

# Analysis on current loans

With the chosen model, we are now able to predict the final outcomes of the current ongoing loans with accuracy.

But first, they will go through the same preprocessing steps earlier.

<details>
<summary>View Code</summary>

```python
df_test=df_test[['earliest_cr_line','last_credit_pull_d','issue_d','total_rec_prncp','total_rec_int','revol_util','int_rate','funded_amnt','grade','sub_grade','total_acc','open_acc','dti','term']]
df_test['term']=df_test['term'].replace({' 36 months':36, ' 60 months':60}).astype(int)
df_test['grade']=df_test['grade'].replace({'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6})
df_test['sub_grade']=df_test['sub_grade'].str[1].astype(int)
df_test['int_rate']=df_test['int_rate'].str.replace('%','').astype(float)
df_test['revol_util']=df_test['revol_util'].str.replace('%','').astype(float)
df_test['earliest_cr_line'] = pd.to_datetime(df_test['earliest_cr_line'], format='%b-%y')
df_test['last_credit_pull_d'] = pd.to_datetime(df_test['last_credit_pull_d'], format='%b-%y')
df_test['issue_d'] = pd.to_datetime(df_test['issue_d'], format='%b-%y')
df_test['last_credit_pull_d']=df_test.apply(lambda row: row['issue_d'] + pd.DateOffset(months=36) if pd.isna(row['last_credit_pull_d']) else row['last_credit_pull_d'], axis=1)
df_test['loan_prncp_perc']=df_test['total_rec_prncp']/(df_test['total_rec_prncp']+df_test['total_rec_int'])
df_test['loan_prncp_perc']=df_test.apply(lambda row: (df_test[df_test['loan_status']==1]['loan_prncp_perc']).mean() if pd.isna(row['loan_prncp_perc']) else row['loan_prncp_perc'], axis=1)
df_test['issue_to_credit_pull']=round((df_test['last_credit_pull_d']-df_test['issue_d'])/np.timedelta64(1,'M'))
df_test['revol_util']=df_test['revol_util'].fillna(0)
df_test=df_test[['revol_util', 'int_rate', 'funded_amnt', 'grade', 'sub_grade','issue_to_credit_pull', 'total_acc', 'open_acc', 'dti','loan_prncp_perc', 'term']]
df_test = pd.DataFrame(scaler_copy.transform(df_test), columns=df_test.columns)
```
</details>

And by using our model to predict the outcomes and default probability,

```python
current_pred_proba=list(LR.predict_proba(df_test)[::,1])
current_pred=list(LR.predict(df_test))
df_test['default_probability']=current_pred_proba
df_test['prediction']=current_pred
df_test.head()
```

### Output

<pre>
<table border="0" class="dataframe">
  <tbody>
    <tr style="text-align: right;">
      <th></th>
      <th>revol_util</th>
      <th>int_rate</th>
      <th>funded_amnt</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>issue_to_credit_pull</th>
      <th>total_acc</th>
      <th>open_acc</th>
      <th>dti</th>
      <th>loan_prncp_perc</th>
      <th>term</th>
      <th>default_probability</th>
      <th>prediction</th>
    </tr>
    <tr>
      <th>0</th>
      <td>0.539540</td>
      <td>0.383035</td>
      <td>0.072464</td>
      <td>0.166667</td>
      <td>1.00</td>
      <td>0.607407</td>
      <td>0.409091</td>
      <td>0.309524</td>
      <td>0.598199</td>
      <td>0.707950</td>
      <td>1.0</td>
      <td>0.632781</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.636637</td>
      <td>0.555321</td>
      <td>0.275362</td>
      <td>0.333333</td>
      <td>1.00</td>
      <td>0.600000</td>
      <td>0.329545</td>
      <td>0.214286</td>
      <td>0.761254</td>
      <td>0.650152</td>
      <td>1.0</td>
      <td>0.808488</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.500501</td>
      <td>0.383035</td>
      <td>0.347826</td>
      <td>0.166667</td>
      <td>1.00</td>
      <td>0.607407</td>
      <td>0.250000</td>
      <td>0.095238</td>
      <td>0.534845</td>
      <td>0.708223</td>
      <td>1.0</td>
      <td>0.585261</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.794795</td>
      <td>0.624341</td>
      <td>0.391304</td>
      <td>0.500000</td>
      <td>0.50</td>
      <td>0.607407</td>
      <td>0.090909</td>
      <td>0.071429</td>
      <td>0.211404</td>
      <td>0.627973</td>
      <td>1.0</td>
      <td>0.808174</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.897898</td>
      <td>0.876712</td>
      <td>0.428986</td>
      <td>0.833333</td>
      <td>0.75</td>
      <td>0.607407</td>
      <td>0.284091</td>
      <td>0.285714</td>
      <td>0.686229</td>
      <td>0.556202</td>
      <td>1.0</td>
      <td>0.870890</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</pre>

## Distribution of loan cases

<details>
<summary>View Code</summary>

```python
plt.pie(df_test['prediction'].value_counts(), labels=['Fully Paid','Charged-Off'], autopct=lambda p: '{:.0f} ({:.1f}%)'.format(p * sum(df_test['prediction'].value_counts()) / 100, p), startangle=90)
plt.title("Distribution of Default and Non-Default Cases")
plt.show()
```
### Outcome
</details>

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/loan-default-classification/images/9.png?raw=true)
