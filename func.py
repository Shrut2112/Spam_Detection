import re
import pandas as pd
import numpy as np
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#spam keywords list
spam_keywords = ['Urgent','Winner','Claim','Free','Risk-free','Prize','Congratulations','Credit','Debt','Exclusive','Guaranteed','Instant','Click here','Limited-time offer','Earn money',
'Loan','Winner','Deposit','Investment','Unsecured','Apply now','Win big','Offers','Unclaimed','Cash','Verify','Account','Promotion','Win','won','Reward','Cheap','Make money','Best offer','Help you','Opportunity','Bargain','Call now','Confirm','Unsubscribe','Risk-free trial']

#function to count spam keywords
def count_keywords(text,keywords):
  return sum([1 for word in text.split() if word in keywords])

#function for preprocess
def preprocess(text):
  
  text = [text]
  df = pd.DataFrame(text,columns=['Message'])
  df['Message_length'] = df['Message'].apply(len)

  df['count'] = df['Message'].apply(lambda x: count_keywords(x,spam_keywords))
  df['Message']= df['Message'].apply(clean_text)
  
  return df

#function for text cleanning 
def clean_text(text):
  text = re.sub(r'\S+@\S+','emailaddress',text)

  #substituting urls
  text = re.sub(r'https?://\S+|www\.\S+','url',text)

  #substituting numbers
  text = re.sub(r'\d+','number',text)

  #removing all punctuation marks
  text = text.translate(str.maketrans('','',string.punctuation))

  #text to lower case
  text = text.lower()

  return text

#tokenizing and stop word removal
def tokenize_stopWord(text):

  #setting stopword to
  stop_words = set(stopwords.words('english'))
  words = word_tokenize(text)
  filtered_words = [word for word in words if word not in stop_words]

  lemmatizer = WordNetLemmatizer()
  lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_words]


  return ' '.join(lemmatized_tokens)
