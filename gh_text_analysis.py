#!/usr/bin/env python
# coding: utf-8

import time
import os
import pandas as pd
import re
import numpy as np
import nltk
import string
from nltk import word_tokenize
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
import csv
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns',50)


os.getcwd()


# words list
words = pd.read_csv('gender_label_words.csv')
male_words = list(words[words['label'] == 'male']['word'])
female_words = list(words[words['label'] == 'female']['word'])


# read data
df = pd.read_csv("Train_rev1.csv\\Train_rev1.csv")
df.info()


# text pre processing
df["FullDescription"] = df["FullDescription"].apply(lambda s: ' '.join(re.sub("(w+://S+)", " ", s).split()))
df["FullDescription"] = df["FullDescription"].apply(lambda s: ' '.join(re.sub("[.,!?:;-='...@#_/*]", " ", s).split()))
df["FullDescription"].replace('[0-9]+', '', regex=True, inplace=True)

# stopwords
nltk.download('stopwords')
stop = set(stopwords.words('english'))
# stop words removal function
def rem_en(input_txt):
    words = input_txt.lower().split()
    noise_free_words = [word for word in words if word not in stop] 
    noise_free_text = " ".join(noise_free_words) 
    return noise_free_text
df["FullDescription"] = df["FullDescription"].apply(lambda s: rem_en(s))

# remove punctuations
def rem_pu(input_txt):
    words = input_txt.lower().split()
    noise_free_words = [word for word in words if word not in punctuation] 
    noise_free_text = " ".join(noise_free_words) 
    return noise_free_text
df["FullDescription"] = df["FullDescription"].apply(lambda s: rem_pu(s))

# tokenize the job descriptions
from nltk.tokenize import RegexpTokenizer
tokeniser = RegexpTokenizer(r'\w+')
df["FullDescription_token"] = df["FullDescription"].apply(lambda x: tokeniser.tokenize(x))


# function to get word count from job descriptions
def word_count_token(male_words, female_words, text_token):
    
    # initialize dictionaries to store word count
    male_words_all_count = {}
    female_words_all_count = {}
    
    # male words count
    for i in range(len(male_words)):
        male_word = male_words[i]
        male_all_count = 0
        if (re.match(male_word, text_token)):
            male_count_match=1
        else:
            male_count_match=0
        male_all_count=male_count_match+len(re.findall(' '+male_word, text_token))
        male_words_all_count[male_word] = male_all_count
    
    # female words count
    for j in range(len(female_words)):
        female_word = female_words[j]
        female_all_count = 0
        if (re.match(female_word, text_token)):
            female_count_match=1
        else:
            female_count_match=0
        female_all_count=female_count_match+len(re.findall(' '+female_word, text_token))
        female_words_all_count[female_word] = female_all_count
        
    return male_words_all_count, female_words_all_count


# update main data frame
df['all_word_counts'] = df["FullDescription"].apply(lambda x: word_count_token(male_words, female_words, x))

# subset dimensions
df1 = df[['LocationNormalized','Category','SalaryNormalized', 'all_word_counts']]

# get all separate counts
df1['male_words_all_count'] = df1["all_word_counts"].apply(lambda x: x[0])
df1['female_words_all_count'] = df1["all_word_counts"].apply(lambda x: x[1])

# dimensions
df_dim = df1[['LocationNormalized','Category', 'SalaryNormalized']]


# make all data frames
# dataframe 1
male_words_all_count_list = list(df1['male_words_all_count'])
df_male_words_all_count = pd.DataFrame(male_words_all_count_list)
dfs = [df_dim, df_male_words_all_count]
df_male_words_all_count = pd.concat(dfs, axis = 1)
df_mwac = df_male_words_all_count.melt(id_vars=['LocationNormalized','Category','SalaryNormalized'], var_name = 'words', value_name = 'count').reset_index(drop=True)
df_mwac['gender'] = 'male'
df_mwac['count_type'] = 'total'
# data frame 2
female_words_all_count_list = list(df1['female_words_all_count'])
df_female_words_all_count = pd.DataFrame(female_words_all_count_list)
dfs = [df_dim, df_female_words_all_count]
df_female_words_all_count = pd.concat(dfs, axis = 1)
df_fwac = df_female_words_all_count.melt(id_vars=['LocationNormalized','Category','SalaryNormalized'], var_name = 'words', value_name = 'count').reset_index(drop=True)
df_fwac['gender'] = 'female'
df_fwac['count_type'] = 'total'

# get total counts
df_female_words_all_count['total_female_words'] = df_female_words_all_count.loc[:, female_words].sum(1)
df_male_words_all_count['total_male_words'] = df_male_words_all_count.loc[:, male_words].sum(1)

# make data frames for unique counts
# data frame 3
df_mwc = df_mwac.copy()
df_mwc.loc[df_mwc['count'] > 1, 'count'] = 1
df_mwc.loc[df_mwc['count_type']=='total','count_type'] = 'unique'
# dataframe 4
df_fwc = df_fwac.copy()
df_fwc.loc[df_fwc['count'] > 1, 'count'] = 1
df_fwc.loc[df_fwc['count_type']=='total','count_type'] = 'unique'

# combine all data
df_all = [df_mwc, df_mwac, df_fwc, df_fwac]
df_final = pd.concat(df_all, axis = 0)
# remove rows where count is 0
df_final = df_final[df_final['count'] > 0]

# group by dimensions and aggreagte counts
df_final_group = df_final.groupby(['LocationNormalized', 'Category',
                                   'words','gender','count_type'])['count'].sum().reset_index()


# save df_final_group
compression_opts = dict(method='zip',
                         archive_name='df_final_group.csv')
df_final_group.to_csv('out.zip', index=False,
           compression=compression_opts)


# make data frame for comparison
df_compare = df_male_words_all_count[['LocationNormalized', 'Category', 'SalaryNormalized', 'total_male_words']]
df_compare['total_female_words'] = df_female_words_all_count['total_female_words']

# ratio calculation
def compare(x):
    if (x['total_male_words']==0 and x['total_female_words']==0):
        ratio = -1
    elif (x['total_male_words']>0 and x['total_female_words']==0):
        ratio = 10
    elif (x['total_male_words']==0 and x['total_female_words']>0):
        ratio = 0
    else:
        ratio = x['total_male_words'] / x['total_female_words']
    return ratio

df_compare['ratio'] = df_compare.apply(compare, axis = 1)

# label maker
def label(x):
    if abs(x) == 1:
        label = 'neutral'
    elif x > 1:
        label = 'male'
    else:
        label = 'female'
    return label

df_compare['label'] = df_compare['ratio'].apply(lambda x: label(x))


# save df compare
compression_opts1 = dict(method='zip',
                         archive_name='df_compare.csv')
df_compare.to_csv('out1.zip', index=True,
           compression=compression_opts1)

