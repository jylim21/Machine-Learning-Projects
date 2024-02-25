<h1 align="center">Amazon Clothing Store Review Analysis</h1>

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/clothing-store-review-analysis/images/clothing-store.jpg?raw=true)

Customer reviews are invaluable in gauging product sentiment and improving offerings for retailers. However, manually parsing through unstructured free-text feedback to identify key themes is infeasible given large review volumes. By utilizing N-grams to break reviews into common phrases to evaluate sentiments, and TF-IDF to transform text into weighted vectors highlighting informative terms, we can perform topic modeling to identify key review themes and aggregate ratings for products. This allows identifying frequently cited complaints on sizing, recommendations on new apparel lines, aggregating ratings for items, and more. 

In this project, we are leveraging the said NLP techniques to programmatically analyze women’s boutique customer reviews. This provides the boutique data-backed input on customer pain points and preferences to guide inventory, marketing and design decisions. 

## THE PROJECT

The dataset used here is the [Consumer Review of Clothing Product](https://data.mendeley.com/datasets/pg3s4hw68k/2) which can be downloaded from *Mendeley Data*, an open source repository containing a rich diversity of research datasets. Our client is a clothing boutique well known to the locals of *Selangor* and has garnered up to a total of 50k reviews for their products. They were actually quite worried when they first showed me the reviews, "are these way too many for you?", no young girl, you're doing me a pretty good favor!

Some of the notable libraries used specifically here are:
- Textblob: It has a built in sentiment analysis function which predicts the polarity and subjectivity of a corpus.
- SpaCy: To handle general pre-processing tasks such as stop words removal, stemming, and POS-tagging.
- SKlearn: We will use its TFIDF-Vectorizer to generate the TFIDF frequency and N-grams for each set of words.

# First Glance

<details>
<summary>View Code</summary>

```python
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re
import json
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
from nltk.chunk import RegexpParser
from pandas import json_normalize
from nltk.stem import WordNetLemmatizer
import spacy
import string
import warnings
pd.set_option('display.max_colwidth', None)
warnings.simplefilter('ignore', pd.errors.SettingWithCopyWarning)

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

Let's load the data into Python:

```python
df = pd.read_csv("/kaggle/input/consumer-review-of-clothing-product/Consumer Review of Clothing Product/data_amazon.xlsx - Sheet1.csv")
df.head()
```

### Output

<pre>
<table border="0" class="dataframe">
  <tbody>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Review</th>
      <th>Cons_rating</th>
      <th>Cloth_class</th>
      <th>Materials</th>
      <th>Construction</th>
      <th>Color</th>
      <th>Finishing</th>
      <th>Durability</th>
    </tr>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>Absolutely wonderful - silky and sexy and comfortable</td>
      <td>4.0</td>
      <td>Intimates</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>Love this dress!  it's sooo pretty.  i happened to find it in a store, and i'm glad i did bc i never would have ordered it online bc it's petite.  i bought a petite and am 5'8".  i love the length on me- hits just a little below the knee.  would definitely be a true midi on someone who is truly petite.</td>
      <td>5.0</td>
      <td>Dresses</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Some major design flaws</td>
      <td>I had such high hopes for this dress and really wanted it to work for me. i initially ordered the petite small (my usual size) but i found this to be outrageously small. so small in fact that i could not zip it up! i reordered it in petite medium, which was just ok. overall, the top half was comfortable and fit nicely, but the bottom half had a very tight under layer and several somewhat cheap (net) over layers. imo, a major design flaw was the net over layer sewn directly into the zipper - it c</td>
      <td>3.0</td>
      <td>Dresses</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>My favorite buy!</td>
      <td>I love, love, love this jumpsuit. it's fun, flirty, and fabulous! every time i wear it, i get nothing but great compliments!</td>
      <td>5.0</td>
      <td>Pants</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Flattering shirt</td>
      <td>This shirt is very flattering to all due to the adjustable front tie. it is the perfect length to wear with leggings and it is sleeveless so it pairs well with any cardigan. love this shirt!!!</td>
      <td>5.0</td>
      <td>Blouses</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</pre>

Notice that there are some NaN entries in the 'Title' column, this stimulated our curiosity on the completeness of the data. By running the 'summary' function I defined earlier, we can observe the quality of the dataset:

```python
summary(df)
```

### Output
<pre>
                 Type          NaN      Zeros
Title          object    3968 (8%)     0 (0%)
Review         object     831 (1%)     0 (0%)
Cons_rating   float64     214 (0%)     0 (0%)
Cloth_class    object      16 (0%)     0 (0%)
Materials     float64  43597 (88%)  3989 (8%)
Construction  float64  43595 (88%)  2849 (5%)
Color         float64  43596 (88%)  4258 (8%)
Finishing     float64  43601 (88%)  4212 (8%)
Durability    float64  43604 (88%)  4514 (9%)
</pre>

From above, we can notice the individual ratings on Materials, Contruction, Color, Finishing, and Durability are mostly empty (88%), so it would be desirable to drop them off.

Although the 'Title' column gives a summary on the 'Review' following it, it is also deemed redundant as the 'Review' column will not only give us the same info, but with equal or much more granular details.

```python
mean_data=df.groupby('Cloth_class')['Cons_rating'].mean().reset_index().sort_values(by='Cons_rating', ascending=False)
sns.barplot(y='Cloth_class', x='Cons_rating', data=mean_data, orient='h')
```

### Output

</details>

![alt text](https://github.com/jylim21/bear-with-data.github.io/blob/main/clothing-store-review-analysis/images/1.jpg?raw=true)
