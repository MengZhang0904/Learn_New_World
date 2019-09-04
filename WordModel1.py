import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from gensim.models import Word2Vec

def find_pct(word,age):
    df = pd.read_csv('word_acq_pct1.csv')
    row = df[df.definition2 == word]
    pct = row.iloc[0][int(age)]
    pct = pct*100
    pct = int(pct)
    return pct

def word_trajectory(df,word):
    x = np.array(df.columns[1:])
    y = np.array(df[df.definition2 == word])
    from matplotlib import style
    style.use('ggplot')
    plt.bar(x,y[0,1:])
    plt.ylim([0,1])
    plt.xlabel('Age of Month')
    plt.ylabel('Proportion of acquision')
    save_name = 'static/plots/' + word + '.png'
    print(save_name)
    get_name = '../static/plots/' + word + '.png'
    print(get_name)
    plt.savefig(save_name)
    return get_name

def same_mo(age,word):
    file = 'dataset/w2v_model_' + age + 'mo_s1000.bin'
    model = Word2Vec.load(file)
    similar_w = model.most_similar(word)
    words, score = zip(*similar_w)
    return words
