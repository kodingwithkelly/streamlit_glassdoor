# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 19:08:07 2021

@author: kellylam
"""

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import nltk
# from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re


st.set_page_config(layout='wide')
st.title('Glassdoor Data Analyst Guide')
st.write('Welcome to an all extensive guide for Data Analysts which include top 10 covetted job skills, 146 popular interview questions from Glassdoor, and a salary estimator. We first begin with an EDA of data scraped from Glassdoor.')
df = pd.read_csv('Glassdoor_w_Seniority.csv')
st.write('Here is my data for reference. This was scraped in March 2021')
st.dataframe(df)

from matplotlib.backends.backend_agg import RendererAgg
import matplotlib
_lock = RendererAgg.lock
matplotlib.use("Agg")

row1_1, row1_2, row1_3 = st.beta_columns(
    (1,1,1))

# SENIORITY PLOT
with row1_1, _lock:
    sns.set_theme()
    fig = plt.figure()
    ax = sns.barplot(x='Seniority', y='Max Salary (Thousands)', data = df, order = ['intern', 'junior', 'mid-level', 'senior'], ci = None, color = 'steelblue') 
    ax = sns.barplot(x='Seniority', y='Min Salary (Thousands)', data = df, order = ['intern', 'junior', 'mid-level', 'senior'], ci = None, color = 'rosybrown') 
    plt.title('Salary by Seniority', fontsize = 20)
    plt.xlabel('Seniority', fontsize = 15)
    plt.ylabel('Salary', fontsize = 15)
    st.pyplot(fig)
    st.write('We see that junior data analysts get paid more than mid-level data analysts which is surprising without knowing the data. There are only 6 junior level listings whose mean will be more than mid-level positions who have 821 listings due to the overwhelming difference in listing volume.')
    
# DISTRIBUTON OF SAL
with row1_2, _lock:
    fig = plt.figure()
    sns.histplot(data = df, x = 'Min Salary (Thousands)', color = 'steelblue')
    sns.histplot(data = df, x = 'Max Salary (Thousands)', color = 'rosybrown')
    plt.title("Distribution of Min and Max Salary", fontsize = 17)
    plt.xlabel('Salary', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    st.pyplot(fig)   
    st.write("Mean value for Min Salary:", df['Min Salary (Thousands)'].mean())
    st.write("Mean value for Max Salary:", df['Max Salary (Thousands)'].mean())

# SALARY BY CITY
with row1_3, _lock:
    cities = st.multiselect('Pick cities to analyze:', options= df.City.tolist(), default = df.groupby(['City'])['Max Salary (Thousands)'].mean().sort_values(ascending = False).index.values[:10])
    
    df_1=df.groupby("City")['Max Salary (Thousands)'].mean().sort_values(ascending=False)
    order_1 = df_1[df_1.index.isin(cities)].index
    
    fig = plt.figure()
    ax = sns.barplot(x='City', y='Max Salary (Thousands)', data = df, order = order_1, ci = None, color = 'steelblue') 
    ax = sns.barplot(x='City', y='Min Salary (Thousands)', data = df, order = order_1, ci = None, color = 'rosybrown') 
    plt.xticks(rotation=45)
    plt.title('Salary by City', fontsize = 20)
    plt.xlabel('City', fontsize = 15)
    plt.ylabel('Salary', fontsize = 15)    
    st.pyplot(fig)
    

row2_1, row2_2, row2_3 = st.beta_columns(
    (1,1,1))
# TOP PAYING INDUSTRY
with row2_1, _lock:
    fig = plt.figure()
    plot_order = df.groupby(['Industry'])['Average Salary (Thousands)'].mean().sort_values(ascending = False).index.values
    ax = sns.barplot(x = 'Industry', y = 'Average Salary (Thousands)', data = df, order = plot_order, ci = None)
    plt.xticks(rotation=90)
    plt.title("Top Paying Industry", fontsize = 17)
    plt.xlabel('Industry', fontsize = 14)
    plt.ylabel('Salary (Thousands)', fontsize = 14)
    st.pyplot(fig)
    st.write('Though media is at the forefront as a top paying industry, there are only 8 companies which are in that industry and same with travel & tourism which skews the data quite a bit.')
    
# TOP PAYING COMPANIES
with row2_2, _lock:
    fig = plt.figure()
    plot_order = df.groupby(['Company'])['Average Salary (Thousands)'].mean().sort_values(ascending = False).index.values[:20]
    ax = sns.barplot(x = 'Company', y = 'Average Salary (Thousands)', data = df, order = plot_order, ci = None)
    plt.xticks(rotation=90)
    plt.title("20 Top Paying Companies", fontsize = 17)
    plt.xlabel('Company', fontsize = 14)
    plt.ylabel('Salary (Thousands)', fontsize = 14)
    st.pyplot(fig)

# SALARY BY STATE
with row2_3, _lock:
    states = st.multiselect('Pick states to analyze:', options= df.State.tolist(), default = df.groupby(['State'])['Average Salary (Thousands)'].mean().sort_values(ascending = False).index.values[:10])
    df_2=df.groupby("State")['Average Salary (Thousands)'].mean().sort_values(ascending=False)
    order_2 = df_2[df_2.index.isin(states)].index

    fig = plt.figure()
    ax = sns.barplot(x='State', y='Average Salary (Thousands)', data = df, order = order_2, ci = None, palette = 'husl') 
    plt.xticks(rotation=55)
    plt.title('Top 20 Salary by State', fontsize = 20)
    plt.xlabel('State', fontsize = 15)
    plt.ylabel('Average Salary (Thousands)', fontsize = 15)
    st.pyplot(fig)
    
    
row3_1, row3_2 = st.beta_columns(
    (1,1.7))

# STATE WITH THE MOST AMOUNT OF LISTINGS
with row3_1, _lock:
    states_1 = st.multiselect('Pick states to analyze:', options= df.State.tolist(), default = df.State.value_counts().iloc[:10].index.tolist())

    df_3=df['State'].value_counts().sort_values(ascending=False)
    order_3 = df_3[df_3.index.isin(states_1)].index
    
    fig = plt.figure()
    ax = sns.countplot(y = 'State', data = df, order = order_3, palette = 'husl')
    plt.title('States By Listing', fontsize = 20)
    plt.xlabel('State', fontsize = 15)
    plt.ylabel('Count', fontsize = 15)
    st.pyplot(fig)
    
# COUNT INDUSTRY
with row3_2, _lock:
    fig = plt.figure()
    ax = sns.countplot(y = 'Industry', data = df, order = df.Industry.value_counts().index, palette = 'husl')
    plt.title('Industry By Listing', fontsize = 20)
    plt.xlabel('Count', fontsize = 15)
    plt.ylabel('Industry', fontsize = 15)
    st.pyplot(fig)
        
# MAP
st.subheader('Map of Job Listing Dispersion')
map_coor = pd.read_csv('glassdoor_map.csv')
map_coor['lat'] = map_coor['Latitude']
map_coor['lon'] = map_coor['Longitude']
st.map(map_coor, 7)
st.write('Note: The map shows points outside the U.S.. I will have to look into my data further to see why there is that error.')
st.info('Streamlit is loading the next section. Thank you for your patience.')

# WORD2VEC
all_sentences = []
for i in range(len(df['Job Description'])):
    text = re.sub(r'[^\w\s]', ' ', df['Job Description'][i])
    text = re.sub(r'\s+',' ',text) 
    text = text.lower() 
    sentences = nltk.sent_tokenize(text) 
    all_sentences.extend(sentences)

all_sentences = [nltk.word_tokenize(all_sentence) for all_sentence in all_sentences]
for i in range(len(all_sentences)):
    all_sentences[i] = [i for i in all_sentences[i] if i not in stopwords.words('english')]
    
model_1 = Word2Vec(all_sentences, min_count=1)
words = model_1.wv.vocab

st.subheader('Word2Vec')
word = st.text_input("Enter in a word/skill to see related words to it: ", 'sql')
topn = st.number_input('Enter the amount of related topics/words you would like to see with your keyword:', min_value=3, step=1)
if word != '' and topn is not None:
    similar = model_1.wv.similar_by_word(word, topn)
    similar = pd.DataFrame(similar)
    st.write(similar)
else:
    similar = model_1.wv.similar_by_word('sql', 3)
    similar = pd.DataFrame(similar)
    st.write(similar)


# RULE BASED MATCHING 
skills = pd.read_csv('skill_count_output.csv')
skills = skills.sort_values('Count', ascending = False)

row4_1, row4_2, row4_3 = st.beta_columns(
    (1,1.5,1))
with row4_2, _lock:
    fig = plt.figure(figsize=(5,5))
    ax = sns.barplot(x = skills.Skill.values[:20], y = skills.Count.values[:20])
    plt.xticks(rotation=90)
    plt.title('Top 20 Skills', fontsize = 20)
    plt.xlabel('Skills', fontsize = 15)
    plt.ylabel('Count in 1,000 Job Listings', fontsize = 15)
    st.pyplot(fig)

# INTERVIEW QUESTIONS
interview_q = pd.read_csv('Study_Guide.csv')
interview_q['Company Name'] = interview_q['name']
cate = st.selectbox('Choose which interview category you would like to see: ', ['Experience', 'Probability', 'Behavioral', 'Technical', 'Brain Teaser', 'Case'])
if cate == 'Experience':
    st.table(interview_q.loc[interview_q['category'] == 'experience'][['Company Name', 'interview questions']])
elif cate == 'Probability':
    st.table(interview_q.loc[interview_q['category'] == 'probability'][['Company Name', 'interview questions']])
elif cate == 'Behavorial':
    st.table(interview_q.loc[interview_q['category'] == 'behavioral'][['Company Name', 'interview questions']])
elif cate == 'Technical':
    st.table(interview_q.loc[interview_q['category'] == 'technical'][['Company Name', 'interview questions']])
elif cate =='Brain Teaser':
    st.table(interview_q.loc[interview_q['category'] == 'brain teaser'][['Company Name', 'interview questions']])
else:
    st.table(interview_q.loc[interview_q['category'] == 'case'][['Company Name', 'interview questions']])

#  estimator deployment
st.subheader('Salary Estimator')
st.write("Choose the inputs to estimate a data analyst's salary.")
import pickle
load_estimator = pickle.load(open('salary_estimator_svr.pkl', 'rb'))
df_model = df[['Rating', 'State', 'Size', 'Industry', 'Seniority']]
rating_sorted = df.Rating.sort_values()

def user_input_features():
    rating  = st.select_slider('Rating', options= rating_sorted[rating_sorted.notnull()].unique().tolist(), value=3)
    state = st.selectbox('State', options=df.State.sort_values().drop_duplicates().tolist(), index=0)
    size = st.selectbox('Size of Company', options=['1 to 50 Employees', '51 to 200 Employees', '201 to 500 Employees', '501 to 1000 Employees', '1001 to 5000 Employees', '5001 to 10000 Employees', '10000+ Employees', 'Unknown'], index=0)
    industry = st.selectbox('Industry of Company', options=df.Industry.unique().tolist(), index=0)
    seniority= st.selectbox('Level of position (Seniority)', options=['intern', 'junior', 'mid-level', 'senior'], index=0)
    data = {'Rating': rating,
                    'State': state,
                    'Size': size,
                    'Industry': industry,
                    'Seniority': seniority}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()
df_pred = pd.concat([input_df, df_model], axis=0)
dummy = pd.get_dummies(df_pred)
dummy = dummy[:1]

prediction = load_estimator.predict(dummy)
st.header('Predicted Salary in Thousands: ')
st.write(prediction)

