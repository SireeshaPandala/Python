import pandas as pd
import nltk

from nltk import sent_tokenize
from nltk import word_tokenize

df = pd.read_csv('./imdb_master.csv',encoding='latin-1' )
df['label'] = df['label'].map({'neg': 0, 'pos':1})

review = df['review'].head(10)


for row in review:
    wtokens = word_tokenize(row)

wtokens = [w.lower() for w in wtokens]
wtokens

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# singles1 = [lemmatizer.lemmatize(plural) for plural in plurals]
singles1 = [lemmatizer.lemmatize(plural) for plural in wtokens]

print('\nThe single Lemmars are :', singles1)

# Stop Word
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
words = [w for w in wtokens if not w in stop_words]
print(words[:100])