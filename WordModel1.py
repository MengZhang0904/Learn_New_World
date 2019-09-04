import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import seaborn as sns
from gensim.models import Word2Vec

def find_pct(word,age):
    df = pd.read_csv('word_acq_pct1.csv')
    row = df[df.definition2 == word]
    pct = row.iloc[0][int(age)]
    pct = pct*100
    pct = int(pct)
    return pct

def word_trajectory(df,word,age):
    x = list(range(16,31))
    print(x)
    len(x)
    y = np.array(df[df.definition2 == word])
    y = y[0,1:]
    type(y)
    y =y.astype(float)
    print(y)
    len(y)
    from matplotlib import style
    style.use('ggplot')
    plt.rcParams.update({'font.size': 13})
    # set customized color for dots
    color1 = ['dodgerblue' for x in range(16,31)]
    idx = int(age) - 16
    color1[idx] = 'crimson'
    color2 = color1

    ax = sns.regplot(x, y,logistic=True, marker = 'o',scatter_kws={'facecolors':color2,'s':65,'edgecolor':'none'},line_kws={'color':'y','linewidth':4.0})
    plt.rcParams.update({'font.size': 13})
    ax.set(xlabel='Age(Month)', ylabel='Proportion of Kids Knowing This Word')

    save_name = 'static/plots/' + word + age + '.png'
    print(save_name)
    get_name = '../static/plots/' + word + age + '.png'
    print(get_name)
    plt.savefig(save_name)
    return get_name

def same_mo(age,word):
    file = 'dataset/w2v_' + age + '_mo_s1500.bin'
    model = Word2Vec.load(file)
    similar_w = model.most_similar(word)
    words, score = zip(*similar_w)
    return words
