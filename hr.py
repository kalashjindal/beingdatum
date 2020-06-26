# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 22:43:49 2020

@author: kalas
"""
from stop_words import get_stop_words


stop_words = get_stop_words('english')

import pandas as pd
import re
from ftfy import fix_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors




df =pd.read_csv('./jd_final.csv') 
df['test']=df['Job_Description'].apply(lambda x: ' '.join([word for word in str(x).split() if len(word)>2 and word not in (stop_words)]))

def data_table(skills):
    org_name_clean = skills   
    def ngrams(string, n=3):
        string = fix_text(string) # fix text
        string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars
        string = string.lower()
        chars_to_remove = [")","(",".","|","[","]","{","}","'"]
        rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
        string = re.sub(rx, '', string)
        string = string.replace('&', 'and')
        string = string.replace(',', ' ')
        string = string.replace('-', ' ')
        string = string.title() # normalise case - capital at start of each word
        string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single
        string = ' '+ string +' ' # pad names for ngrams...
        string = re.sub(r'[,-./]|\sBD',r'', string)
        ngrams = zip(*[string[i:] for i in range(n)])
        return [''.join(ngram) for ngram in ngrams]
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
    tfidf = vectorizer.fit_transform(org_name_clean)    
    
    def getNearestN(query):
      queryTFIDF_ = vectorizer.transform(query)
      distances, indices = nbrs.kneighbors(queryTFIDF_)
      return distances, indices
    nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
    unique_org = (df['test'].values)
    distances, indices = getNearestN(unique_org)
    unique_org = list(unique_org)
    matches = []
    for i,j in enumerate(indices):
        dist=round(distances[i][0],2)
  
        temp = [dist]
        matches.append(temp)
    matches = pd.DataFrame(matches, columns=['Match confidence'])
    df['match']=matches['Match confidence']
    df1=df.sort_values('match')
    df2=df1[['Position', 'Company','Location','url']].head(10).reset_index()
    
    
    
    
    
    return (df2)

    
    
        

    
