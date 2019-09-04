from flask import Flask
app = Flask(__name__)

# Load required libraries
from flask import render_template
#from flaskexample import app
from flask import request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from gensim import models
from gensim.models import Word2Vec
from statsmodels import genmod
from statsmodels.genmod import generalized_linear_model
from statsmodels.genmod.generalized_linear_model import GLM
import WordModel1
from WordModel1 import *
# import os
# os.getcwd()

df = pd.read_csv('word_acq_pct1.csv')

@app.route('/')
@app.route('/index')
@app.route('/index.html')

def index():

    return render_template('index.html')

@app.route('/output')
@app.route('/output.html')
def output():
    age=request.args.get('age')
    print(type(age))
    gender=request.args.get('gender')
    birth_order=request.args.get('birth_order')
    age=request.args.get('age')
    word=request.args.get('word')
    print(age)

    # Customize the word, percent, and trajectory given the input
    blurb=word
    df = pd.read_csv('dataset/word_acq_pct1.csv')
    print(len(df))
    print(df.columns)
    print(word)
    row = df[df.definition2 == str(word)]
    print(row)
    if row.empty == True:
        return render_template('error.html')
    else:
        pct = row.iloc[0][str(age)]
        pct = pct*100
        the_result = int(pct)
        print(pct)

        word_cat =''
        feedback =''
        if pct < 25:
            word_cat = 'A hard word'
            feedback = 'Your kid is ahead of others!'
        elif pct >= 25 and pct < 50:
            word_cat = 'An advanced word'
            feedback = 'Your kid is good at learning words!'
        elif pct >= 50 and pct < 75:
            word_cat = 'A regular word'
            feedback = 'Your kid is right on track!'
        elif pct >= 75:
            word_cat = 'An easy word'
            feedback = 'Keep working, your kid can learn more!'

        plt.clf()
        get_name=WordModel1.word_trajectory(df,word,age)

        # Creat Amz url
        amz1 = 'https://www.amazon.com/s?k=children+book+about+'
        amz2 = '&ref=nb_sb_noss_2'

        # Customize words emerging similarly this month and next month
        current_words=same_mo(age,word)
        print(current_words)
        similar_word1=current_words[0]
        sim_url1 = amz1 + similar_word1 + amz2
        similar_word2=current_words[1]
        sim_url2 = amz1 + similar_word2 + amz2
        similar_word3=current_words[2]
        sim_url3 = amz1 + similar_word3 + amz2
        similar_word4=current_words[3]
        sim_url4 = amz1 + similar_word4 + amz2
        similar_word5=current_words[4]
        sim_url5 = amz1 + similar_word5 + amz2

        if int(age) < 29:
            age1 = int(age) + 1
            age1 = str(age1)
            future_words1=WordModel1.same_mo(age1,word)
            print(future_words1)
            age2 = int(age) + 2
            age2 = str(age2)
            future_words2=WordModel1.same_mo(age2,word)
            print(future_words2)
            df_ls = [x for x in future_words1 if x not in future_words2]
            future_words = future_words2 + tuple(df_ls)
            return_words = [x for x in future_words if x not in current_words]
            print(return_words)
        elif int(age) == 29:             # return last 5 and 30 mo
            late_sim = current_words[5:]
            age1 = int(age) + 1
            age1 = str(age1)
            future_words1=WordModel1.same_mo(age1,word)
            df_ls = [x for x in late_sim if x not in future_words1]
            future_words = future_words1 + tuple(df_ls)
            return_words = [x for x in future_words if x not in current_words]
        elif int(age) == 30:             # return last 5 of 30 mo
            return_words = current_words[5:]

        future_similar_word1=return_words[0]
        fut_url1 = amz1 + future_similar_word1 + amz2
        future_similar_word2=return_words[1]
        fut_url2 = amz1 + future_similar_word2 + amz2
        future_similar_word3=return_words[2]
        fut_url3 = amz1 + future_similar_word3 + amz2
        future_similar_word4=return_words[3]
        fut_url4 = amz1 + future_similar_word4 + amz2
        future_similar_word5=return_words[4]
        fut_url5 = amz1 + future_similar_word5 + amz2


        return render_template('output.html', blurb=blurb, age = age,word_cat = word_cat,feedback=feedback,get_name=get_name, \
            similar_word1=similar_word1,similar_word2=similar_word2,similar_word3=similar_word3, \
            similar_word4=similar_word4,similar_word5=similar_word5,future_similar_word1=future_similar_word1, \
            future_similar_word2=future_similar_word2,future_similar_word3=future_similar_word3, \
            future_similar_word4=future_similar_word4,future_similar_word5=future_similar_word5, \
            sim_url1=sim_url1,sim_url2=sim_url2,sim_url3=sim_url3,sim_url4=sim_url4,sim_url5=sim_url5, \
            fut_url1=fut_url1,fut_url2=fut_url2,fut_url3=fut_url3,fut_url4=fut_url4,fut_url5=fut_url5)

@app.route('/about')
@app.route('/slides')
@app.route('/about.html')
def about():
    return render_template("about.html")
