from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud


filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df=pd.read_csv("wiki_data.csv",sep=",")
df.head()
#Part 1 Preprocessing
#Step 1
def clean_text(dataframe,column='text'):
    dataframe[column]=dataframe[column].str.replace('[^\w\s]','')
    dataframe[column]=dataframe[column].str.lower()
    dataframe[column]=dataframe[column].str.replace('\d','')
#Step 2
clean_text(df)
df.head()
#Step 3 Stopwords
import nltk
#nltk.download("stopwords")
sw=stopwords.words("english")
def remove_stopwords(dataframe,column='text'):
    dataframe[column]=dataframe[column].apply(lambda x:" ".join(x for x in str(x).split() if x not in sw))

#Step 4
remove_stopwords(df)
df.head()

#Step 5 Rare Words
temp_df = pd.Series(' '.join(df["text"]).split()).value_counts()
drops=temp_df[temp_df<=2000]
df['text']=df['text'].apply(lambda x:' '.join(x for x in x.split() if x not in drops))

#Step 6 Tokenization
#nltk.download('punkt')
df['text'].apply(lambda x:TextBlob(x).words).head()

#Step 7 Lemmatization 
#nltk.download("wordnet")
df['text']=df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df.head()
#Part 2 Data Visualization
#Step 1 word count
tf = df['text'].apply(lambda x : pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns=["words","tf"]
tf.sort_values("tf",ascending=False)
#Step 2 BarPlot
tf[tf["tf"]>7000].plot.bar(x="words",y="tf")
plt.show()
#Step 3 WordCloud
text = " ".join(x for x in df.text)
wordcloud = WordCloud(max_font_size=50,
                     max_words=100,
                     background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis("off")
plt.show()
wordcloud.to_file("wordcloud.png")

#Part 3
#Step 1,2,3 all together in a function
def nlp_one_function(dataframe,column="text",plot=False):
    """
    Metin değerleri için ön işleme görselleştirme sağlamaktadır.
    
    Parameters
    ----------
    dataframe: Metin ön işlemesi için kullanılacak dataframe.
    column: Dataframe sütunu.
    plot:Görselleştirme gerçekleştirme.
    
    Return
    ------
    None
    
    """
    #punctuation
    dataframe[column]=dataframe[column].str.replace('[^\w\s]','')
    #lower
    dataframe[column]=dataframe[column].str.lower()
    #numbers
    dataframe[column]=dataframe[column].str.replace('\d','')
    #stop words
    sw=stopwords.words("english")
    dataframe[column]=dataframe[column].apply(lambda x:" ".join(x for x in str(x).split() if x not in sw))
    #rare words
    temp_df = pd.Series(' '.join(dataframe[column]).split()).value_counts()
    drops=temp_df[temp_df<=2000]
    dataframe[column]=dataframe[column].apply(lambda x:' '.join(x for x in x.split() if x not in drops))
    #tokenization
    dataframe[column].apply(lambda x:TextBlob(x).words).head()
    #lemmatization
    dataframe[column]=dataframe[column].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    
    if plot:
        text = " ".join(x for x in dataframe.text)
        wordcloud = WordCloud(max_font_size=50,
                     max_words=100,
                     background_color="white").generate(text)
        plt.figure()
        plt.imshow(wordcloud,interpolation="bilinear")
        plt.axis("off")
        plt.show()
        wordcloud.to_file("wordcloud.png")
