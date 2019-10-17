import pandas as pd
import csv
import random
from random import shuffle
import copy
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import math

# Random sampling training and testing dataset for each month
def split_sets(age,f):
    file = 'ValProcess/DataPerAge/data_' + str(age) + 'mo.csv'
    df = pd.read_csv(file)
    df = df.drop(['comprehension','birth_order','ethnicity','sex','mom_ed'],axis=1)
    pre_train_df = df.sample(frac = f, replace = False)
    train_df = pre_train_df.drop(['data_id','age','production'],axis=1)
    test_df = df.drop(train_df.index)
    name1 = 'ValProcess/train/binary/train_' + str(age) + '.csv'
    name2 = 'ValProcess/test/test_' + str(age) + '.csv'
    train_df.to_csv(name1,index=False)
    test_df.to_csv(name2,index=False)
    return train_df, test_df

# For training dataframe, convert binary values into strings
def bin2str(df):
    num_word = df.shape[1]
    header = list(df.columns.values)
    word_list = []
    for index, rows in df.iterrows():
        word = []
        for j in range(num_word):
            if rows[j] == 1:
                word.append(header[j])
        word_list.append(word)
    return word_list

# Save list of strings as csv
def spec_save1(ls,age):
    file = 'ValProcess/train/str/train_' + str(age) + 'mo_str.csv'
    with open(file,'w') as output:
        writer = csv.writer(output,lineterminator='\n')
        writer.writerows(ls)

# Save shuffled list into csv
def spec_save2(ls,age,sft):
    file = 'ValProcess/train/shuffled/' + str(age) + 'mo_' + 'train_shuffle_' + str(sft) + '.csv'
    with open(file,'w') as output:
        writer = csv.writer(output,lineterminator='\n')
        writer.writerows(ls)

# For training data, shuffle words for each individual
def myshuffle(ls,sft):
    shuffled = []
    for i in range(sft):
        # print('Shuffle No.', i)
        for item in ls:
            shuffle(item)
            item_copy = copy.deepcopy(item)
            shuffled.append(item_copy)
    print('Shuffle is Done. ' + 'Total length: ' + str(len(shuffled)))
    return shuffled

# Read saved training data that were strings and shuffled
def spec_read1(age,sft):
    file = 'ValProcess/train/shuffled/' + str(age) + 'mo_' + 'train_shuffle_' + str(sft) + '.csv'
    with open(file, 'r') as f:
      reader = csv.reader(f)
      ls = list(reader)
    return ls

# Train word embedding using training data for each age group
def word_embedding(ls,age,n_com=2):
    model = Word2Vec(ls,size=100, window=5, min_count=1, workers=4)
    X = model[model.wv.vocab]
    pca = PCA(n_com)
    result = pca.fit_transform(X)
    plt.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    plt.show()
    fig_name = 'w2v_plots/' + str(age) + 'mo.png'
    plt.savefig(fig_name)
    return model

# Save trained word embedding in .bin file
def spec_save3(model,m,sft):
    file = 'ValProcess/train/w2v/w2v_train_' + str(m) + 'mo_s' + str(sft) + '.bin'
    model.save(file)

# Read saved Word2Vec .bin file and return similar words given age
def same_mo(age,word,sft):
    file = 'ValProcess/train/w2v/w2v_train_' + str(age) + 'mo_s' + str(sft) + '.bin'
    model = Word2Vec.load(file)
    similar_w = model.most_similar(word)
    return similar_w

# Plot bar chart to better visualize the how similar the nearby words are
def plot_similar(age,word):
    sim_ls = same_mo(age,word)
    words,scores = zip(*sim_ls)
    plt.bar(words,score)
    plt.ylim([0,1])
    plt.rcParams['xtick.labelsize']=18
    plt.rcParams['ytick.labelsize']=18
    plt.show()

# Create a dynamic n for finding n nearby in test
# def return_n(age,word):
#     word_sum = pd.read_csv('General_csv/By-Word Summary Data_WS.csv')
#     pct_age = word_sum[['definition2',str(age)]]
#     pct_age['rank'] = pct_age[str(age)].rank(ascending = False)
#     pct_age.sort_values('rank',inplace = True)
#     rank_word = pct_age.loc[pct_age['definition2'] == word,'rank']
#     n = math.ceil(10 + rank_word * 0.3)
#     return n

# Find the rank of a given word in whole dataset of an age group
def get_word_rank(age,word):
    file = 'ValProcess/train/binary/train_' + str(age) + '.csv'
    df = pd.read_csv(file)
    see_mean = df.describe()
    pct = see_mean.iloc[1,:]
    pct_rank = pct.rank(ascending = False)
    pct_rank.sort_values(inplace = True)
    rank = pct_rank.index.get_loc(word)
    # word_sum = pd.read_csv('General_csv/By-Word Summary Data_WS.csv')
    # pct = word_sum[['definition2',str(age)]]
    # pct['rank'] = pct[str(age)].rank(ascending = False)
    # pct.sort_values('rank',inplace = True)
    # rank = pct.loc[pct['definition2'] == word]['rank'].values
    # print(rank)
    return rank

# Give a word, find the rows in the test dataset where book == 1
def n_nearby_test(age,word):
    file = 'ValProcess/test/test_' + str(age) + '.csv'
    df_read = pd.read_csv(file)
    df_use = df_read.loc[df_read[word] == 1]
    see_mean = df_use.describe()
    pct = see_mean.iloc[1,3:]
    pct_rank = pct.rank(ascending = False)
    pct_rank.sort_values(inplace = True)
    # print(pct_rank.head(20))
    # n = return_n(age, word)
    rank = get_word_rank(age,word)
    l_bound = math.floor(rank-(rank*0.4+10))
    if l_bound > 0:
        l_bound = l_bound
    else:
        l_bound = 0
    #print(l_bound)
    h_bound = math.ceil(rank+(rank*0.4+10))
    # word_rank = pct_rank.nsmallest(n,keep='first')
    nearby = pct_rank.index[l_bound:h_bound]
    return nearby

# Calculate chance level following given this validation metric
def chance(age): # this works, put it in for loop save results in csv
    word_ls = age_word_ls(age)
    chance_prob = []
    for word in word_ls:
        rank = get_word_rank(age,word)
        # print(rank)
        l_bound = math.floor(rank-(rank*0.4+10))
        if rank < 16.66:
            w = math.floor(rank+(rank*0.4+10)) # change to floor
        else:
            w = math.floor(20 + rank * 0.8)
        #print(w)
        prob1 = (w/680)*((w-1)/680)*((w-2)/680)*((w-3)/680)*((w-4)/680)*((w-5)/680)*((w-6)/680)
        prob2 = prob1*120
        if prob2 > 1:
            prob2 = 1
        else:
            prob2 = prob2
        #print(prob2)
        chance_prob.append(prob2)
    mean_chance = np.mean(chance_prob)
    return mean_chance, chance_prob

# Compare similar words from model and nearby words from test
def train_test_comp(age, word, sft):
    word_sim = same_mo(age,word,sft)
    train_words,scores = zip(*word_sim)
    # print(train_words)
    nearby = n_nearby_test(age,word)
    nearby = nearby.tolist()
    # print(nearby)
    diff_words = [x for x in train_words if x not in nearby]
    denom = len(word_sim) + np.finfo(float).eps
    miss_rate = len(diff_words)/denom
    return word, train_words, nearby, miss_rate

# Generate the word list for each month
def age_word_ls(age):
    word_sum = pd.read_csv('General_csv/By-Word Summary Data_WS.csv')
    pct_age = word_sum[['definition2',str(age)]]
    word_ls = pct_age.loc[pct_age[str(age)] > 0.2,'definition2']
    word_ls = word_ls.tolist()
    return word_ls

# Iterate and calculate the miss_rate for each word
def total_miss(age,sft):
    word_ls = age_word_ls(age)
    total_miss_info = []
    total_miss_rate = []
    for word in word_ls:
        print(word)
        word, train_words, nearby, miss_rate = train_test_comp(age,word,sft)
        print(miss_rate)
        total_miss_info.append([word, train_words, nearby, miss_rate])
    miss_rates = [el[3] for el in total_miss_info]
    return total_miss_info,miss_rates

# Save miss info as csv file
def spec_save4(ls,age,sft):
    file = 'ValProcess/miss_rate/miss_info_' + str(age) + 'mo_s' + str(sft) + '.csv'
    with open(file,'w') as output:
        writer = csv.writer(output,lineterminator='\n')
        writer.writerows(ls)

# For setting up figure parameters
def my_fig_set(w,h,font):
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = w
    fig_size[1] = h
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams.update({'font.size': font})

# For plotting precision and chance
def plot_precision(filename):
    new_sum = pd.read_csv(filename)
    Age = np.array(new_sum.iloc[:,0])
    Precision = np.array(new_sum.iloc[:,2])
    Chance = np.array(new_sum.iloc[:,3])
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(Age,Precision,'r--',linewidth=4,label='Current Model')
    ax.plot(Age,Chance,'b--',linewidth=4,label='Base Model')
    ax.legend()
    plt.ylim(0,1)
    plt.xlim(16,30)
    plt.xlabel('Age (Month)')
    plt.ylabel('Precision')
    plt.show()

# Minor changes for refining web display
# Save list of strings as csv
def spec_save4(ls,age):
    file = 'WebRefine/str/' + str(age) + '_mo_str.csv'
    with open(file,'w') as output:
        writer = csv.writer(output,lineterminator='\n')
        writer.writerows(ls)

# Save shuffled list into csv
def spec_save5(ls,age,sft):
    file = 'WebRefine/shuffled/' + str(age) + '_mo_' + 'shuffle_' + str(sft) + '.csv'
    with open(file,'w') as output:
        writer = csv.writer(output,lineterminator='\n')
        writer.writerows(ls)

# Save trained word embedding in .bin file
def spec_save6(model,m,sft):
    file = 'WebRefine/w2v/w2v_' + str(m) + '_mo_s' + str(sft) + '.bin'
    model.save(file)
