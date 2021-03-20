import numpy as np 
import pandas as pd
from nltk.stem import WordNetLemmatizer
import nltk
import re
import emoji
import string

nltk.download('words')
nltk.download('wordnet')
np.random.seed(500) 

lower = False
lemma = False
em_rep = False
stem = False
num = False
stopw = False
misspell = False

crypto_path = "./data/crypto_db"
stock_path = "./data/stock_db"


def preprocess(data_df, punc=False, lower=False, lemma=False, em_rep=False, num=False, stopw=False, misspell=False):
    
    #remove \n
    data_df['message'] = data_df['message'].str.replace('\n',' ') 
    

    if lower:     #lower data
        data_df['message'] = data_df['message'].str.lower()
        
    if punc:      #remove punctuation
        data_df['message'] = data_df['message'].str.replace('[^\w\s]','')
        
    if num:       #remove numerical values
        data_df['message'] = data_df['message'].str.replace(r'[0-9]+','') 
    
    if stopw:     #remove stop words
        data_df['message'] = data_df['message'].apply(lambda x: " ".join([item for item in x.split() if item not in stop_words]))
    
    if lemma: #Use lemmatization
        lemmatizer = WordNetLemmatizer()
        data_df['message'] = data_df['message'].apply(lambda x: " ".join([lemmatizer.lemmatize(item) for item in x.split()]))
    
    if misspell:  #remove not english words (other language + misspelled words)
        words = set(nltk.corpus.wordnet.words())
        data_df['message'] = data_df['message'].apply(lambda x: " ".join(w for w in nltk.wordpunct_tokenize(x) if w.lower() in words or not w.isalpha()))
        
    if em_rep:    #replace emoticons
        data_df['message'] = data_df['message'].apply(convert_emoticons)
        
    return data_df

    
def convert_emoticons(text): 
    text = emoji.demojize(text, delimiters=(" :",": "), use_aliases=True)
    text = re.sub(' +', ' ', text)
    return text
        
    
stop_words = ['you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
