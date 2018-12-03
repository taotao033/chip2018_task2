import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import logging
from wordcloud import WordCloud
from collections import Counter, defaultdict
from sklearn.metrics import roc_auc_score
import timeit
import _pickle as cPickle
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from tqdm import tqdm_notebook
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize
import nltk
nltk.download('punkt')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)


def build_word_embedding_dict(path="./dataset/word_embedding.txt"):

    f = open(path, "r", encoding="utf-8")
    word_embedding_dict = {}
    for line in f.readlines():
        values = line.split()
        wid_key = values[0]
        wid_values = np.asarray(values[1:], dtype='float32')
        word_embedding_dict[wid_key] = wid_values

    return word_embedding_dict


def build_char_embedding_dict(path="./dataset/char_embedding.txt"):

    f = open(path, "r", encoding="utf-8")
    char_embedding_dict = {}
    for line in f.readlines():
        values = line.split()
        cid_key = values[0]
        cid_values = np.asarray(values[1:], dtype='float32')
        char_embedding_dict[cid_key] = cid_values

    return char_embedding_dict


def get_train_data_base_word(path="./dataset/train.csv"):

    train_reform_save_path = "./dataset/dataset_reform/base_word/train_reform_base_word.csv"
    if os.path.exists(train_reform_save_path):
        df_train = pd.read_csv(train_reform_save_path, encoding="utf-8")
    else:
        #os.makedirs("./dataset/dataset_reform/")
        df_train = pd.read_csv(path, encoding="utf-8")
        df_question_id = pd.read_csv("./dataset/question_id.csv", encoding="utf-8")
        #start process qid1
        train_qid1_list = df_train["qid1"]
        temp_concat = []
        for li in train_qid1_list:
            word = df_question_id[df_question_id["qid"] == li]
            temp = word["wid"].values[0]
            temp_concat.append(temp)

        df_train["qid1"] = temp_concat
        #complete process qid1

        #start process qid2
        train_qid2_list = df_train["qid2"]
        temp_concat = []
        for li in train_qid2_list:
            word = df_question_id[df_question_id["qid"] == li]
            temp = word["wid"].values[0]
            temp_concat.append(temp)

        df_train["qid2"] = temp_concat
        #complete process qid2
        df_train.to_csv(train_reform_save_path, index=False, sep=",", encoding="utf-8")
    return df_train


def get_test_data_base_word(path="./dataset/test.csv"):

    test_reform_save_path = "./dataset/dataset_reform/baes_word/test_reform_base_word.csv"
    if os.path.exists(test_reform_save_path):
        df_test = pd.read_csv(test_reform_save_path, encoding="utf-8")
    else:
        #os.makedirs("./dataset/dataset_reform/")
        df_test = pd.read_csv(path, encoding="utf-8")
        df_question_id = pd.read_csv("./dataset/question_id.csv", encoding="utf-8")
        #start process qid1
        train_qid1_list = df_test["qid1"]
        temp_concat = []
        for li in train_qid1_list:
            word = df_question_id[df_question_id["qid"] == li]
            temp = word["wid"].values[0]
            temp_concat.append(temp)

        df_test["qid1"] = temp_concat
        #complete process qid1

        #start process qid2
        train_qid2_list = df_test["qid2"]
        temp_concat = []
        for li in train_qid2_list:
            word = df_question_id[df_question_id["qid"] == li]
            temp = word["wid"].values[0]
            temp_concat.append(temp)

        df_test["qid2"] = temp_concat
        #complete process qid2
        df_test.to_csv(test_reform_save_path, index=False, sep=",", encoding="utf-8")

    return df_test


def get_train_data_base_char(path="./dataset/train.csv"):

    train_reform_save_path = "./dataset/dataset_reform/base_word/train_reform_base_char.csv"
    if os.path.exists(train_reform_save_path):
        df_train = pd.read_csv(train_reform_save_path, encoding="utf-8")
    else:
        #os.makedirs("./dataset/dataset_reform/")
        df_train = pd.read_csv(path, encoding="utf-8")
        df_question_id = pd.read_csv("./dataset/question_id.csv", encoding="utf-8")
        #start process qid1
        train_qid1_list = df_train["qid1"]
        temp_concat = []
        for li in train_qid1_list:
            word = df_question_id[df_question_id["qid"] == li]
            temp = word["cid"].values[0]
            temp_concat.append(temp)

        df_train["qid1"] = temp_concat
        #complete process qid1

        #start process qid2
        train_qid2_list = df_train["qid2"]
        temp_concat = []
        for li in train_qid2_list:
            word = df_question_id[df_question_id["qid"] == li]
            temp = word["cid"].values[0]
            temp_concat.append(temp)

        df_train["qid2"] = temp_concat
        #complete process qid2
        df_train.to_csv(train_reform_save_path, index=False, sep=",", encoding="utf-8")
    return df_train


def get_test_data_base_char(path="./dataset/test.csv"):

    test_reform_save_path = "./dataset/dataset_reform/base_char/test_reform_base_char.csv"
    if os.path.exists(test_reform_save_path):
        df_test = pd.read_csv(test_reform_save_path, encoding="utf-8")
    else:
        #os.makedirs("./dataset/dataset_reform/")
        df_test = pd.read_csv(path, encoding="utf-8")
        df_question_id = pd.read_csv("./dataset/question_id.csv", encoding="utf-8")
        #start process qid1
        test_qid1_list = df_test["qid1"]
        temp_concat = []
        for li in test_qid1_list:
            word = df_question_id[df_question_id["qid"] == li]
            temp = word["cid"].values[0]
            temp_concat.append(temp)

        df_test["qid1"] = temp_concat
        #complete process qid1

        #start process qid2
        test_qid2_list = df_test["qid2"]
        temp_concat = []
        for li in test_qid2_list:
            word = df_question_id[df_question_id["qid"] == li]
            temp = word["cid"].values[0]
            temp_concat.append(temp)

        df_test["qid2"] = temp_concat
        #complete process qid2
        df_test.to_csv(test_reform_save_path, index=False, sep=",", encoding="utf-8")

    return df_test


def sorting_trainset_testset(train_path, test_path):

    train_sort_df = pd.read_csv(train_path, encoding='utf-8')
    test_sort_df = pd.read_csv(test_path, encoding='utf-8')

    train_q1_q2_len = train_sort_df["question1"].apply(lambda x: get_word_count(x)) + \
                      train_sort_df["question2"].apply(lambda x: get_word_count(x))
    train_sort_df["q1_q2_len"] = train_q1_q2_len
    train_sort_df.sort_values("q1_q2_len", axis=0, ascending=True, inplace=True)
    train_sort_df.to_csv('./dataset/dataset_reform/base_word/add_8features/train_sorted_accordding_q1_q2_len.csv', index=False)

    test_q1_q2_len = test_sort_df["question1"].apply(lambda x: get_word_count(x)) + \
                      test_sort_df["question2"].apply(lambda x: get_word_count(x))
    test_sort_df["q1_q2_len"] = test_q1_q2_len
    test_sort_df.sort_values("q1_q2_len", axis=0, ascending=True, inplace=True)
    test_sort_df.to_csv('./dataset/dataset_reform/base_word/add_8features/test_sorted_accordding_q1_q2_len.csv', index=False)



#_WORD_SPLIT = re.compile(",")
UNI_BLEU_WEIGHTS = (1, 0, 0, 0)
BI_BLEU_WEIGHTS = (0, 1, 0, 0)
BLEU2_WEIGHTS = (0.5, 0.5, 0, 0)


def save_results(predictions, IDs, filename):
    with open(filename, 'w') as f:
        f.write("test_id,is_duplicate\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (IDs[i], pred))


def tokenizer(sentence):
    """Very basic tokenizer: split the sentence by space into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
      words.append(space_separated_fragment)
    return words


def char_ngram_tokenizer(sentence, n):
    """Character ngram tokenizer: split the sentence into a list of char ngram tokens."""
    return [sentence[i:i+n] for i in range(len(sentence)-n+1)]


def get_word_count(x):
    return len(tokenizer(str(x)))

def get_char_count(x):
    return len(tokenizer(str(x)))

def word_overlap(x):
    return len(set(str(x['question1']).lower().split()).intersection(
             set(str(x['question2']).lower().split())))


def char_unigram_overlap(x):
    return len(set(str(x['question1']).lower().split()).intersection(
             set(str(x['question2']).lower().split())))


def get_uni_BLEU(x):
    s_function = SmoothingFunction()
    # method 2 is add 1 smoothing
    return sentence_bleu([tokenizer(str(x['question2']))],
                         tokenizer(str(x['question1'])),
                         weights=UNI_BLEU_WEIGHTS,
                         smoothing_function=s_function.method2)


def get_bi_BLEU(x):
    s_function = SmoothingFunction()
    # method 2 is add 1 smoothing
    return sentence_bleu([tokenizer(str(x['question2']))],
                         tokenizer(str(x['question1'])),
                         weights=BI_BLEU_WEIGHTS,
                         smoothing_function=s_function.method2)


def get_BLEU2(x):
    s_function = SmoothingFunction()
    # method 2 is add 1 smoothing
    return sentence_bleu([tokenizer(str(x['question2']))],
                         tokenizer(str(x['question1'])),
                         weights=BLEU2_WEIGHTS,
                         smoothing_function=s_function.method2)


def feature_eng(df, df2):

    # word count of question 1
    df['q1_word_count'] = df['question1'].apply(get_word_count)

    # word count of question 2
    df['q2_word_count'] = df['question2'].apply(get_word_count)

    # word count difference
    df['word_count_diff'] = abs(df['q1_word_count'] - df['q2_word_count'])

    # number of word overlap between q1 and q2
    df['word_overlap'] = df.apply(word_overlap, axis=1)

    # unigram BLEU score
    df['uni_BLEU'] = df.apply(get_uni_BLEU, axis=1)

    # bigram BLEU score
    df['bi_BLEU'] = df.apply(get_bi_BLEU, axis=1)

    # BLEU2 score
    df['BLEU2'] = df.apply(get_BLEU2, axis=1)

    # character unigram overlap
    df['char_unigram_overlap'] = df2.apply(char_unigram_overlap, axis=1)


    return df


def feature_eng_base_char(df):

    # word count of question 1
    df['q1_char_count'] = df['question1'].apply(get_char_count)

    # word count of question 2
    df['q2_char_count'] = df['question2'].apply(get_char_count)

    # word count difference
    df['char_count_diff'] = abs(df['q1_char_count'] - df['q2_char_count'])

    # number of char overlap between q1 and q2
    df['char_overlap'] = df.apply(char_unigram_overlap, axis=1)

    # unigram BLEU score
    df['uni_BLEU'] = df.apply(get_uni_BLEU, axis=1)

    # bigram BLEU score
    df['bi_BLEU'] = df.apply(get_bi_BLEU, axis=1)

    # BLEU2 score
    df['BLEU2'] = df.apply(get_BLEU2, axis=1)



    return df


def generate_8features_base_word():
    train_data_path_base_word = "./dataset/dataset_reform/base_word/train_reform_base_word.csv"
    train_data_path_base_char = "./dataset/dataset_reform/base_char/train_reform_base_char.csv"
    test_data_path_base_word = "./dataset/dataset_reform/base_word/test_reform_base_word.csv"
    test_data_path_base_char = "./dataset/dataset_reform/base_chartest_reform_base_char.csv"
    df_train_base_word = pd.read_csv(train_data_path_base_word, encoding='utf-8')
    df_train_base_char = pd.read_csv(train_data_path_base_char, encoding='utf-8')
    # add features to training data
    print('Start engineering 8 Hand-Craft-Features for training data...')
    print('This might take a while...')
    df_train = feature_eng(df_train_base_word, df_train_base_char)

    print(df_train.head())

    # update train.csv with new features as columns
    train_8features_path = './dataset/dataset_reform/base_word/add_8features/train_8features_base_word.csv'
    df_train.to_csv(train_8features_path, index=False)
    print('Finish engineering 8 Hand-Craft-Features for training data and save in ' + \
          train_8features_path + '\n')

    # add features to testing data
    print('Loading test data....')
    df_test_base_word = pd.read_csv(test_data_path_base_word, encoding='utf-8')
    df_test_base_char = pd.read_csv(test_data_path_base_char, encoding='utf-8')
    print('Start engineering 8 Hand-Craft-Features for testing data...')
    print('This might take a while...')
    df_test = feature_eng(df_test_base_word, df_test_base_char)

    # save new features to testing data
    test_8features_path = './dataset/dataset_reform/base_word/add_8features/test_8features_base_word.csv'
    df_test.to_csv(test_8features_path, index=False)
    print('Finish engineering 8 Hand-Craft-Features for chip2018 testing data and save in ' + \
          test_8features_path + '\n')


def generate_7features_base_char():
    train_data_path_base_char = "./dataset/dataset_reform/base_char/train_reform_base_char.csv"
    test_data_path_base_char = "./dataset/dataset_reform/base_char/test_reform_base_char.csv"
    df_train_base_char = pd.read_csv(train_data_path_base_char, encoding='utf-8')
    # add features to training data
    print('Start engineering 7 Hand-Craft-Features for training data...')
    print('This might take a while...')
    df_train = feature_eng_base_char(df_train_base_char)

    # print(df_train.head())

    # update train.csv with new features as columns
    train_7features_path = './dataset/dataset_reform/base_char/add_7features/train_7features_base_char.csv'
    df_train.to_csv(train_7features_path, index=False)
    print('Finish engineering 7 Hand-Craft-Features for training data and save in ' + \
          train_7features_path + '\n')

    # add features to testing data
    print('Loading test data....')
    df_test_base_char = pd.read_csv(test_data_path_base_char, encoding='utf-8')
    print('Start engineering 8 Hand-Craft-Features for testing data...')
    print('This might take a while...')
    df_test = feature_eng_base_char(df_test_base_char)

    # save new features to testing data
    test_7features_path = './dataset/dataset_reform/base_char/add_7features/test_7features_base_char.csv'
    df_test.to_csv(test_7features_path, index=False)
    print('Finish engineering 7 Hand-Craft-Features for chip2018 testing data and save in ' + \
          test_7features_path + '\n')


def generate_magic_features_base_word():
    TRAIN_CSV = "./dataset/dataset_reform/base_word/train_reform_base_word.csv"
    TEST_CSV = "./dataset/dataset_reform/base_word/test_reform_base_word.csv"
    PROCESSED_DATA_PATH = "./dataset/dataset_reform/base_word/add_magic_features"
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)

    # initializing the dataframes
    train_df = pd.read_csv(TRAIN_CSV).fillna('none')
    test_df = pd.read_csv(TEST_CSV).fillna('none')

    # adding additional columns to the dataframe ,and add the values later
    # min_freq
    # common_neighbours
    # qid1, qid2
    def add_cols_dataframe(df):
        df['min_freq'] = 0
        df['common_neighbours'] = 0

    # add q_ids for only test data set
    def add_qids_dataframe(df):
        len = df.shape[0]
        evens = [x for x in range(len * 2 + 2) if x % 2 == 0 and x != 0]
        odds = [x for x in range(len * 2) if x % 2 != 0]
        df['qid1'] = odds
        df['qid2'] = evens

    # add cols for both test and train data frames
    add_cols_dataframe(train_df)
    add_cols_dataframe(test_df)

    # add for only test data frame
    add_qids_dataframe(test_df)

    # save dataframes as csv
    # train_df.to_csv(PROCESSED_DATA_PATH + '/' + "p_train.csv", sep=',', index=False)
    # test_df.to_csv(PROCESSED_DATA_PATH + '/' + "p_test.csv", sep=',', index=False)

    nodes = {}
    nodeCount = 0
    freq = {}
    edges = {}
    question_cols = ['question1', 'question2']
    for dataTuple in [train_df, test_df]:
        for index, row in dataTuple.iterrows():
            nodeIds = []
            for question in question_cols:
                if (row[question] not in nodes):
                    nodes[row[question]] = nodeCount
                    freq[nodeCount] = 0
                    edges[nodeCount] = set()
                    nodeCount = nodeCount + 1
                tmpNodeId = nodes[row[question]]
                freq[tmpNodeId] = freq[tmpNodeId] + 1
                nodeIds.append(tmpNodeId)
            edges[nodeIds[0]].add(nodeIds[1])
            edges[nodeIds[1]].add(nodeIds[0])
            if (index % 10000 == 0):
                print(index)

    for dataTuple in [train_df, test_df]:
        dataTuple['q_len1'] = 0
        dataTuple['q_len2'] = 0
        dataTuple['diff_len'] = 0
        dataTuple['word_len1'] = 0
        dataTuple['word_len2'] = 0
        dataTuple['common_words'] = 0
        dataTuple['fuzzy_qratio'] = 0
        dataTuple['fuzzy_wratio'] = 0
        dataTuple['fuzzy_partial_ratio'] = 0
        dataTuple['fuzzy_partial_token_set_ratio'] = 0
        dataTuple['fuzzy_partial_token_sort_ratio'] = 0
        dataTuple['fuzzy_token_set_ratio'] = 0
        dataTuple['fuzzy_token_sort_ratio'] = 0

        for index, row in dataTuple.iterrows():
            a = nodes[row['question1']]
            b = nodes[row['question2']]
            dataTuple.set_value(index, 'qid1', a)
            dataTuple.set_value(index, 'qid2', b)

            dataTuple.set_value(index, 'min_freq', min(freq[a], freq[b]))
            dataTuple.set_value(index, 'common_neighbours', len(edges[a].intersection(edges[b])))

            q1, q2 = str(row['question1']), str(row['question2'])
            q_len1, q_len2 = len(q1), len(q2)

            # question length metrics
            dataTuple.set_value(index, 'q_len1', q_len1)
            dataTuple.set_value(index, 'q_len2', q_len2)
            dataTuple.set_value(index, 'diff_len', q_len1 - q_len2)

            words_q1, words_q2 = q1.split(), q2.split()
            word_len1, word_len2 = len(words_q1), len(words_q2)

            # number of words metric
            dataTuple.set_value(index, 'word_len1', word_len1)
            dataTuple.set_value(index, 'word_len2', word_len2)
            tmp = len(set(words_q1).intersection(words_q2))
            dataTuple.set_value(index, 'common_words', (tmp))

            # fuzzy metrics
            dataTuple.set_value(index, 'fuzzy_qratio', fuzz.QRatio(q1, q2))
            dataTuple.set_value(index, 'fuzzy_wratio', fuzz.WRatio(q1, q2))
            dataTuple.set_value(index, 'fuzzy_partial_ratio', fuzz.partial_ratio(q1, q2))
            dataTuple.set_value(index, 'fuzzy_partial_token_set_ratio', fuzz.partial_token_set_ratio(q1, q2))
            dataTuple.set_value(index, 'fuzzy_partial_token_sort_ratio', fuzz.partial_token_sort_ratio(q1, q2))
            dataTuple.set_value(index, 'fuzzy_token_set_ratio', fuzz.token_set_ratio(q1, q2))
            dataTuple.set_value(index, 'fuzzy_token_sort_ratio', fuzz.token_sort_ratio(q1, q2))

            if (index % 10000 == 0):
                print(index)

    train_df.to_csv(PROCESSED_DATA_PATH + '/' + 'train.csv', sep=',', index=False)
    test_df.to_csv(PROCESSED_DATA_PATH + '/' + 'test.csv', sep=',', index=False)


def generate_magic_features_base_char():
    TRAIN_CSV = "./dataset/dataset_reform/base_char/train_reform_base_char.csv"
    TEST_CSV = "./dataset/dataset_reform/base_char/test_reform_base_char.csv"
    PROCESSED_DATA_PATH = "./dataset/dataset_reform/base_char/add_magic_features"
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)

    # initializing the dataframes
    train_df = pd.read_csv(TRAIN_CSV).fillna('none')
    test_df = pd.read_csv(TEST_CSV).fillna('none')

    # adding additional columns to the dataframe ,and add the values later
    # min_freq
    # common_neighbours
    # qid1, qid2
    def add_cols_dataframe(df):
        df['min_freq'] = 0
        df['common_neighbours'] = 0

    # add q_ids for only test data set
    def add_qids_dataframe(df):
        len = df.shape[0]
        evens = [x for x in range(len * 2 + 2) if x % 2 == 0 and x != 0]
        odds = [x for x in range(len * 2) if x % 2 != 0]
        df['qid1'] = odds
        df['qid2'] = evens

    # add cols for both test and train data frames
    add_cols_dataframe(train_df)
    add_cols_dataframe(test_df)

    # add for only test data frame
    add_qids_dataframe(test_df)

    # save dataframes as csv
    # train_df.to_csv(PROCESSED_DATA_PATH + '/' + "p_train.csv", sep=',', index=False)
    # test_df.to_csv(PROCESSED_DATA_PATH + '/' + "p_test.csv", sep=',', index=False)

    nodes = {}
    nodeCount = 0
    freq = {}
    edges = {}
    question_cols = ['question1', 'question2']
    for dataTuple in [train_df, test_df]:
        for index, row in dataTuple.iterrows():
            nodeIds = []
            for question in question_cols:
                if (row[question] not in nodes):
                    nodes[row[question]] = nodeCount
                    freq[nodeCount] = 0
                    edges[nodeCount] = set()
                    nodeCount = nodeCount + 1
                tmpNodeId = nodes[row[question]]
                freq[tmpNodeId] = freq[tmpNodeId] + 1
                nodeIds.append(tmpNodeId)
            edges[nodeIds[0]].add(nodeIds[1])
            edges[nodeIds[1]].add(nodeIds[0])
            if (index % 10000 == 0):
                print(index)

    for dataTuple in [train_df, test_df]:
        dataTuple['q_len1'] = 0
        dataTuple['q_len2'] = 0
        dataTuple['diff_len'] = 0
        dataTuple['char_len1'] = 0
        dataTuple['char_len2'] = 0
        dataTuple['common_chars'] = 0
        dataTuple['fuzzy_qratio'] = 0
        dataTuple['fuzzy_wratio'] = 0
        dataTuple['fuzzy_partial_ratio'] = 0
        dataTuple['fuzzy_partial_token_set_ratio'] = 0
        dataTuple['fuzzy_partial_token_sort_ratio'] = 0
        dataTuple['fuzzy_token_set_ratio'] = 0
        dataTuple['fuzzy_token_sort_ratio'] = 0

        for index, row in dataTuple.iterrows():
            a = nodes[row['question1']]
            b = nodes[row['question2']]
            dataTuple.set_value(index, 'qid1', a)
            dataTuple.set_value(index, 'qid2', b)

            dataTuple.set_value(index, 'min_freq', min(freq[a], freq[b]))
            dataTuple.set_value(index, 'common_neighbours', len(edges[a].intersection(edges[b])))

            q1, q2 = str(row['question1']), str(row['question2'])
            q_len1, q_len2 = len(q1), len(q2)

            # question length metrics
            dataTuple.set_value(index, 'q_len1', q_len1)
            dataTuple.set_value(index, 'q_len2', q_len2)
            dataTuple.set_value(index, 'diff_len', q_len1 - q_len2)

            words_q1, words_q2 = q1.split(), q2.split()
            word_len1, word_len2 = len(words_q1), len(words_q2)

            # number of words metric
            dataTuple.set_value(index, 'char_len1', word_len1)
            dataTuple.set_value(index, 'char_len2', word_len2)
            tmp = len(set(words_q1).intersection(words_q2))
            dataTuple.set_value(index, 'common_chars', (tmp))

            # fuzzy metrics
            dataTuple.set_value(index, 'fuzzy_qratio', fuzz.QRatio(q1, q2))
            dataTuple.set_value(index, 'fuzzy_wratio', fuzz.WRatio(q1, q2))
            dataTuple.set_value(index, 'fuzzy_partial_ratio', fuzz.partial_ratio(q1, q2))
            dataTuple.set_value(index, 'fuzzy_partial_token_set_ratio', fuzz.partial_token_set_ratio(q1, q2))
            dataTuple.set_value(index, 'fuzzy_partial_token_sort_ratio', fuzz.partial_token_sort_ratio(q1, q2))
            dataTuple.set_value(index, 'fuzzy_token_set_ratio', fuzz.token_set_ratio(q1, q2))
            dataTuple.set_value(index, 'fuzzy_token_sort_ratio', fuzz.token_sort_ratio(q1, q2))

            if (index % 10000 == 0):
                print(index)

    train_df.to_csv(PROCESSED_DATA_PATH + '/' + 'train.csv', sep=',', index=False)
    test_df.to_csv(PROCESSED_DATA_PATH + '/' + 'test.csv', sep=',', index=False)
    logger.info("Complete generate magic features,saved path: ./dataset/dataset_reform/base_char/add_magic_features")


def generate_distance_features_base_word():

    def sent2vec(s):
        words = str(s)
        words = word_tokenize(words)
        M = []
        for w in words:
            try:
                M.append(embeddings_index[w])
            except:
                continue
        M = np.array(M)
        v = M.sum(axis=0)
        return v / np.sqrt((v ** 2).sum())
    logger.info("Loading data...")
    train_df = pd.read_csv("./dataset/dataset_reform/base_word/train_reform_base_word.csv", encoding='utf-8')
    test_df = pd.read_csv("./dataset/dataset_reform/base_word/test_reform_base_word.csv", encoding='utf-8')
    print('Indexing word vectors')
    embeddings_index = build_word_embedding_dict(path='./dataset/word_embedding.txt')
    print('Found %d word vectors of word_embedding.txt.' % len(embeddings_index))

    logger.info("Processing trainseting...")

    question1_vectors = np.zeros((train_df.shape[0], 300))

    for i, q in enumerate(tqdm_notebook(train_df.question1.values)):
        question1_vectors[i, :] = sent2vec(q)

    question2_vectors = np.zeros((train_df.shape[0], 300))
    for i, q in enumerate(tqdm_notebook(train_df.question2.values)):
        question2_vectors[i, :] = sent2vec(q)

    train_df['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    train_df['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    train_df['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    train_df['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    train_df['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    train_df['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    train_df['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    train_df['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
    train_df['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
    train_df['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
    train_questions_vector_save_path='./dataset/dataset_reform/base_word/questions_vectors/'
    if not os.path.exists(train_questions_vector_save_path):
        os.makedirs(train_questions_vector_save_path)
    cPickle.dump(question1_vectors, open(train_questions_vector_save_path + 'train_q1_w2v.pkl', 'wb'), -1)
    cPickle.dump(question2_vectors, open(train_questions_vector_save_path + 'train_q2_w2v.pkl', 'wb'), -1)

    logger.info("(base on word trainseting)question1_vectors and question2_vectors are saved at: "
                + train_questions_vector_save_path)
    train_df_save_path = './dataset/dataset_reform/base_word/add_questions_distance_features/'
    if not os.path.exists(train_df_save_path):
        os.makedirs(train_df_save_path)
    train_df.to_csv(train_df_save_path + "train.csv", index=False)

    logger.info("Processing testset...")

    test_question1_vectors = np.zeros((test_df.shape[0], 300))

    for i, q in enumerate(tqdm_notebook(test_df.question1.values)):
        test_question1_vectors[i, :] = sent2vec(q)

    test_question2_vectors = np.zeros((test_df.shape[0], 300))
    for i, q in enumerate(tqdm_notebook(test_df.question2.values)):
        test_question2_vectors[i, :] = sent2vec(q)

    test_df['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(test_question1_vectors),
                                                                  np.nan_to_num(test_question2_vectors))]

    test_df['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(test_question1_vectors),
                                                                        np.nan_to_num(test_question2_vectors))]

    test_df['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(test_question1_vectors),
                                                                    np.nan_to_num(test_question2_vectors))]

    test_df['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(test_question1_vectors),
                                                                      np.nan_to_num(test_question2_vectors))]

    test_df['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(test_question1_vectors),
                                                                        np.nan_to_num(test_question2_vectors))]

    test_df['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(test_question1_vectors),
                                                                           np.nan_to_num(test_question2_vectors))]

    test_df['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(test_question1_vectors),
                                                                          np.nan_to_num(test_question2_vectors))]

    test_df['skew_q1vec'] = [skew(x) for x in np.nan_to_num(test_question1_vectors)]
    test_df['skew_q2vec'] = [skew(x) for x in np.nan_to_num(test_question2_vectors)]
    test_df['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(test_question1_vectors)]
    test_questions_vector_save_path = './dataset/dataset_reform/base_word/questions_vectors/'
    if not os.path.exists(test_questions_vector_save_path):
        os.makedirs(train_questions_vector_save_path)
    cPickle.dump(question1_vectors, open(test_questions_vector_save_path + 'test_q1_w2v.pkl', 'wb'), -1)
    cPickle.dump(question2_vectors, open(test_questions_vector_save_path + 'test_q2_w2v.pkl', 'wb'), -1)

    logger.info("(base on word testset)question1_vectors and question2_vectors are saved at: "
                + test_questions_vector_save_path)
    test_df_save_path = './dataset/dataset_reform/base_word/add_questions_distance_features/'
    if not os.path.exists(train_df_save_path):
        os.makedirs(test_df_save_path)
    test_df.to_csv(test_df_save_path + "test.csv", index=False)


def generate_distance_features_base_char():

    def sent2vec(s):
        words = str(s)
        words = word_tokenize(words)
        M = []
        for w in words:
            try:
                M.append(embeddings_index[w])
            except:
                continue
        M = np.array(M)
        v = M.sum(axis=0)
        return v / np.sqrt((v ** 2).sum())
    logger.info("Loading data...")
    train_df = pd.read_csv("./dataset/dataset_reform/base_char/train_reform_base_char.csv", encoding='utf-8')
    test_df = pd.read_csv("./dataset/dataset_reform/base_char/test_reform_base_char.csv", encoding='utf-8')
    print('Indexing word vectors')
    embeddings_index = build_word_embedding_dict(path='./dataset/char_embedding.txt')
    print('Found %d word vectors of char_embedding.txt.' % len(embeddings_index))

    logger.info("Processing trainseting...")

    question1_vectors = np.zeros((train_df.shape[0], 300))

    for i, q in enumerate(tqdm_notebook(train_df.question1.values)):
        question1_vectors[i, :] = sent2vec(q)

    question2_vectors = np.zeros((train_df.shape[0], 300))
    for i, q in enumerate(tqdm_notebook(train_df.question2.values)):
        question2_vectors[i, :] = sent2vec(q)

    train_df['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    train_df['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    train_df['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    train_df['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    train_df['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    train_df['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    train_df['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    train_df['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
    train_df['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
    train_df['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
    train_questions_vector_save_path='./dataset/dataset_reform/base_char/questions_vectors/'
    if not os.path.exists(train_questions_vector_save_path):
        os.makedirs(train_questions_vector_save_path)
    cPickle.dump(question1_vectors, open(train_questions_vector_save_path + 'train_q1_w2v.pkl', 'wb'), -1)
    cPickle.dump(question2_vectors, open(train_questions_vector_save_path + 'train_q2_w2v.pkl', 'wb'), -1)

    logger.info("(base on char trainseting)question1_vectors and question2_vectors are saved at: "
                + train_questions_vector_save_path)
    train_df_save_path = './dataset/dataset_reform/base_char/add_questions_distance_features/'
    if not os.path.exists(train_df_save_path):
        os.makedirs(train_df_save_path)
    train_df.to_csv(train_df_save_path + "train.csv", index=False)

    logger.info("Processing testset...")

    test_question1_vectors = np.zeros((test_df.shape[0], 300))

    for i, q in enumerate(tqdm_notebook(test_df.question1.values)):
        test_question1_vectors[i, :] = sent2vec(q)

    test_question2_vectors = np.zeros((test_df.shape[0], 300))
    for i, q in enumerate(tqdm_notebook(test_df.question2.values)):
        test_question2_vectors[i, :] = sent2vec(q)

    test_df['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(test_question1_vectors),
                                                                  np.nan_to_num(test_question2_vectors))]

    test_df['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(test_question1_vectors),
                                                                        np.nan_to_num(test_question2_vectors))]

    test_df['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(test_question1_vectors),
                                                                    np.nan_to_num(test_question2_vectors))]

    test_df['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(test_question1_vectors),
                                                                      np.nan_to_num(test_question2_vectors))]

    test_df['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(test_question1_vectors),
                                                                        np.nan_to_num(test_question2_vectors))]

    test_df['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(test_question1_vectors),
                                                                           np.nan_to_num(test_question2_vectors))]

    test_df['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(test_question1_vectors),
                                                                          np.nan_to_num(test_question2_vectors))]

    test_df['skew_q1vec'] = [skew(x) for x in np.nan_to_num(test_question1_vectors)]
    test_df['skew_q2vec'] = [skew(x) for x in np.nan_to_num(test_question2_vectors)]
    test_df['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(test_question1_vectors)]
    test_questions_vector_save_path = './dataset/dataset_reform/base_char/questions_vectors/'
    if not os.path.exists(test_questions_vector_save_path):
        os.makedirs(train_questions_vector_save_path)
    cPickle.dump(question1_vectors, open(test_questions_vector_save_path + 'test_q1_w2v.pkl', 'wb'), -1)
    cPickle.dump(question2_vectors, open(test_questions_vector_save_path + 'test_q2_w2v.pkl', 'wb'), -1)

    logger.info("(base on char testset)question1_vectors and question2_vectors are saved at: "
                + test_questions_vector_save_path)
    test_df_save_path = './dataset/dataset_reform/base_char/add_questions_distance_features/'
    if not os.path.exists(train_df_save_path):
        os.makedirs(test_df_save_path)
    test_df.to_csv(test_df_save_path + "test.csv", index=False)


def features_merge_base_word():
    #train1: q1_word_count,q2_word_count,word_count_diff,word_overlap(common_words),uni_BLEU,bi_BLEU,BLEU2,char_unigram_overlap
    #train2: word_match,tfidf_word_match
    #train3: q1_hash,q2_hash,q1_freq,q2_freq
    #train4: word_len1,word_len2,common_words(word_overlap),fuzzy_qratio,fuzzy_wratio,
    # fuzzy_partial_ratio,fuzzy_partial_token_set_ratio,fuzzy_partial_token_sort_ratio,
    # fuzzy_token_set_ratio,fuzzy_token_sort_ratio
    #train5:'cosine_distance', 'cityblock_distance', 'jaccard_distance', 'canberra_distance', 'euclidean_distance',
    # 'minkowski_distance', 'braycurtis_distance', 'skew_q1vec', 'skew_q2vec', 'kur_q1vec'

    logger.info("Load data ...")
    train1 = pd.read_csv("./dataset/dataset_reform/base_word/add_8features/train_8features_base_word.csv",
                         encoding='utf-8')
    train2 = pd.read_csv("./dataset/dataset_reform/base_word/add_feature1/train_add_feature1_base_word.csv",
                         encoding='utf-8')
    train3 = pd.read_csv("./dataset/dataset_reform/base_word/add_feature2/train_add_feature2_base_word.csv",
                         encoding='utf-8')
    train4 = pd.read_csv("./dataset/dataset_reform/base_word/add_magic_features/train.csv",
                         encoding='utf-8')
    train5 = pd.read_csv("./dataset/dataset_reform/base_word/add_questions_distance_features/train.csv",
                         encoding='utf-8')
    train6 = pd.read_csv("./dataset/dataset_reform/base_word/add_8features/train_add_single_feature_q1_q2_intersect.csv",
                         encoding='utf-8')

    logger.info("Processing trainseting data ...")

    columns = ['question1', 'question2', 'label', 'q1_word_count', 'q2_word_count', 'word_count_diff', 'word_overlap',
               'char_unigram_overlap', 'q1_hash', 'q2_hash', 'q1_freq', 'q2_freq', "q1_q2_intersect",
               'fuzzy_qratio', 'fuzzy_wratio',
               'fuzzy_partial_ratio', 'fuzzy_partial_token_set_ratio', 'fuzzy_partial_token_sort_ratio',
               'fuzzy_token_set_ratio', 'fuzzy_token_sort_ratio', 'word_match', 'tfidf_word_match',
               'uni_BLEU', 'bi_BLEU', 'BLEU2', 'cosine_distance', 'cityblock_distance', 'jaccard_distance',
               'canberra_distance', 'euclidean_distance', 'minkowski_distance', 'braycurtis_distance', 'skew_q1vec',
                'skew_q2vec', 'kur_q1vec']

    data_frame_train = pd.DataFrame({
        "question1": train1["question1"],
        "question2": train1["question2"],
        "label": train1["label"],
        "q1_word_count": train1["q1_word_count"],
        "q2_word_count": train1["q2_word_count"],
        "word_count_diff": train1["word_count_diff"],
        "word_overlap": train1["word_overlap"],
        "char_unigram_overlap": train1["char_unigram_overlap"],
        "q1_hash": train3["q1_hash"],
        "q2_hash": train3["q2_hash"],
        "q1_freq": train3["q1_freq"],
        "q2_freq": train3["q2_freq"],
        "q1_q2_intersect": train6["q1_q2_intersect"],
        "fuzzy_qratio": train4["fuzzy_qratio"],
        "fuzzy_wratio": train4["fuzzy_wratio"],
        "fuzzy_partial_ratio": train4["fuzzy_partial_ratio"],
        "fuzzy_partial_token_set_ratio": train4["fuzzy_partial_token_set_ratio"],
        "fuzzy_partial_token_sort_ratio": train4["fuzzy_partial_token_sort_ratio"],
        "fuzzy_token_set_ratio": train4["fuzzy_token_set_ratio"],
        "fuzzy_token_sort_ratio": train4["fuzzy_token_sort_ratio"],
        "word_match": train2["word_match"],
        "tfidf_word_match": train2["tfidf_word_match"],
        "uni_BLEU": train1["uni_BLEU"],
        "bi_BLEU": train1["bi_BLEU"],
        "BLEU2": train1["BLEU2"],
        "cosine_distance": train5["cosine_distance"],
        'cityblock_distance': train5['cityblock_distance'],
        'jaccard_distance': train5['jaccard_distance'],
        'canberra_distance': train5['canberra_distance'],
        'euclidean_distance': train5['euclidean_distance'],
        'minkowski_distance': train5['minkowski_distance'],
        'braycurtis_distance': train5['braycurtis_distance'],
        'skew_q1vec': train5['skew_q1vec'],
        'skew_q2vec': train5['skew_q2vec'],
        'kur_q1vec': train5['kur_q1vec']

    })



    test1 = pd.read_csv("./dataset/dataset_reform/base_word/add_8features/test_8features_base_word.csv",
                         encoding='utf-8')
    test2 = pd.read_csv("./dataset/dataset_reform/base_word/add_feature1/test_add_feature1_base_word.csv",
                         encoding='utf-8')
    test3 = pd.read_csv("./dataset/dataset_reform/base_word/add_feature2/test_add_feature2_base_word.csv",
                         encoding='utf-8')
    test4 = pd.read_csv("./dataset/dataset_reform/base_word/add_magic_features/test.csv",
                         encoding='utf-8')
    test5 = pd.read_csv("./dataset/dataset_reform/base_word/add_questions_distance_features/test.csv",
                         encoding='utf-8')
    test6 = pd.read_csv("./dataset/dataset_reform/base_word/add_8features/test_add_single_feature_q1_q2_intersect.csv",
                         encoding='utf-8')

    logger.info("Processing testset data ...")
    data_frame_test = pd.DataFrame({
        "question1": test1["question1"],
        "question2": test1["question2"],
        "label": test1["label"],
        "q1_word_count": test1["q1_word_count"],
        "q2_word_count": test1["q2_word_count"],
        "word_count_diff": test1["word_count_diff"],
        "word_overlap": test1["word_overlap"],
        "char_unigram_overlap": test1["char_unigram_overlap"],
        "q1_hash": test3["q1_hash"],
        "q2_hash": test3["q2_hash"],
        "q1_freq": test3["q1_freq"],
        "q2_freq": test3["q2_freq"],
        "q1_q2_intersect": test6["q1_q2_intersect"],
        "fuzzy_qratio": test4["fuzzy_qratio"],
        "fuzzy_wratio": test4["fuzzy_wratio"],
        "fuzzy_partial_ratio": test4["fuzzy_partial_ratio"],
        "fuzzy_partial_token_set_ratio": test4["fuzzy_partial_token_set_ratio"],
        "fuzzy_partial_token_sort_ratio": test4["fuzzy_partial_token_sort_ratio"],
        "fuzzy_token_set_ratio": test4["fuzzy_token_set_ratio"],
        "fuzzy_token_sort_ratio": test4["fuzzy_token_sort_ratio"],
        "word_match": test2["word_match"],
        "tfidf_word_match": test2["tfidf_word_match"],
        "uni_BLEU": test1["uni_BLEU"],
        "bi_BLEU": test1["bi_BLEU"],
        "BLEU2": test1["BLEU2"],
        "cosine_distance": test5["cosine_distance"],
        'cityblock_distance': test5['cityblock_distance'],
        'jaccard_distance': test5['jaccard_distance'],
        'canberra_distance': test5['canberra_distance'],
        'euclidean_distance': test5['euclidean_distance'],
        'minkowski_distance': test5['minkowski_distance'],
        'braycurtis_distance': test5['braycurtis_distance'],
        'skew_q1vec': test5['skew_q1vec'],
        'skew_q2vec': test5['skew_q2vec'],
        'kur_q1vec': test5['kur_q1vec']

    })

    save_path = "./dataset/dataset_reform/base_word/features_merged/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data_frame_train.to_csv("./dataset/dataset_reform/base_word/features_merged/train.csv",
                            index=False, columns=columns, encoding='utf-8')
    data_frame_test.to_csv("./dataset/dataset_reform/base_word/features_merged/test.csv",
                            index=False, columns=columns, encoding='utf-8')
    logger.info("Complete merge features,and saved path: ./dataset/dataset_reform/base_word/features_merged")


def features_merge_base_char():
    # train1: q1_char_count,q2_char_count,char_count_diff,char_overlap(common_words),uni_BLEU,bi_BLEU,BLEU2
    # train2: char_match,tfidf_char_match
    # train3: q1_hash,q2_hash,q1_freq,q2_freq
    # train4: char_len1,word_len2,common_words(word_overlap),fuzzy_qratio,fuzzy_wratio,
    # fuzzy_partial_ratio,fuzzy_partial_token_set_ratio,fuzzy_partial_token_sort_ratio,
    # fuzzy_token_set_ratio,fuzzy_token_sort_ratio

    logger.info("Load data ...")
    train1 = pd.read_csv("./dataset/dataset_reform/base_char/add_7features/train_7features_base_char.csv",
                         encoding='utf-8')
    train2 = pd.read_csv("./dataset/dataset_reform/base_char/add_feature1/train_add_feature1_base_char.csv",
                         encoding='utf-8')
    train3 = pd.read_csv("./dataset/dataset_reform/base_char/add_feature2/train_add_feature2_base_char.csv",
                         encoding='utf-8')
    train4 = pd.read_csv("./dataset/dataset_reform/base_char/add_magic_features/train.csv",
                         encoding='utf-8')
    train5 = pd.read_csv("./dataset/dataset_reform/base_char/add_questions_distance_features/train.csv",
                         encoding='utf-8')
    train6 = pd.read_csv("./dataset/dataset_reform/base_char/add_7features/train_add_single_feature_q1_q2_intersect.csv",
                         encoding='utf-8')

    logger.info("Processing trainseting data ...")

    columns = ['question1', 'question2', 'label', 'q1_char_count', 'q2_char_count', 'char_count_diff',
               'char_overlap', 'q1_hash', 'q2_hash', 'q1_freq', 'q2_freq', 'q1_q2_intersect',
               'fuzzy_qratio', 'fuzzy_wratio', 'fuzzy_partial_ratio', 'fuzzy_partial_token_set_ratio',
               'fuzzy_partial_token_sort_ratio',
               'fuzzy_token_set_ratio', 'fuzzy_token_sort_ratio', 'char_match', 'tfidf_char_match',
               'uni_BLEU', 'bi_BLEU', 'BLEU2','cosine_distance', 'cityblock_distance', 'jaccard_distance',
               'canberra_distance', 'euclidean_distance', 'minkowski_distance', 'braycurtis_distance', 'skew_q1vec',
                'skew_q2vec', 'kur_q1vec']

    data_frame_train = pd.DataFrame({
        "question1": train1["question1"],
        "question2": train1["question2"],
        "label": train1["label"],
        "q1_char_count": train1["q1_char_count"],
        "q2_char_count": train1["q2_char_count"],
        "char_count_diff": train1["char_count_diff"],
        "char_overlap": train1["char_overlap"],
        "q1_hash": train3["q1_hash"],
        "q2_hash": train3["q2_hash"],
        "q1_freq": train3["q1_freq"],
        "q2_freq": train3["q2_freq"],
        "q1_q2_intersect": train6["q1_q2_intersect"],
        "fuzzy_qratio": train4["fuzzy_qratio"],
        "fuzzy_wratio": train4["fuzzy_wratio"],
        "fuzzy_partial_ratio": train4["fuzzy_partial_ratio"],
        "fuzzy_partial_token_set_ratio": train4["fuzzy_partial_token_set_ratio"],
        "fuzzy_partial_token_sort_ratio": train4["fuzzy_partial_token_sort_ratio"],
        "fuzzy_token_set_ratio": train4["fuzzy_token_set_ratio"],
        "fuzzy_token_sort_ratio": train4["fuzzy_token_sort_ratio"],
        "char_match": train2["char_match"],
        "tfidf_char_match": train2["tfidf_char_match"],
        "uni_BLEU": train1["uni_BLEU"],
        "bi_BLEU": train1["bi_BLEU"],
        "BLEU2": train1["BLEU2"],
        "cosine_distance": train5["cosine_distance"],
        'cityblock_distance': train5['cityblock_distance'],
        'jaccard_distance': train5['jaccard_distance'],
        'canberra_distance': train5['canberra_distance'],
        'euclidean_distance': train5['euclidean_distance'],
        'minkowski_distance': train5['minkowski_distance'],
        'braycurtis_distance': train5['braycurtis_distance'],
        'skew_q1vec': train5['skew_q1vec'],
        'skew_q2vec': train5['skew_q2vec'],
        'kur_q1vec': train5['kur_q1vec']

    })

    test1 = pd.read_csv("./dataset/dataset_reform/base_char/add_7features/test_7features_base_char.csv",
                         encoding='utf-8')
    test2 = pd.read_csv("./dataset/dataset_reform/base_char/add_feature1/test_add_feature1_base_char.csv",
                         encoding='utf-8')
    test3 = pd.read_csv("./dataset/dataset_reform/base_char/add_feature2/test_add_feature2_base_char.csv",
                         encoding='utf-8')
    test4 = pd.read_csv("./dataset/dataset_reform/base_char/add_magic_features/test.csv",
                         encoding='utf-8')
    test5 = pd.read_csv("./dataset/dataset_reform/base_char/add_questions_distance_features/test.csv",
                        encoding='utf-8')
    test6 = pd.read_csv("./dataset/dataset_reform/base_char/add_7features/test_add_single_feature_q1_q2_intersect.csv",
                        encoding='utf-8')
    logger.info("Processing testset data ...")

    data_frame_test = pd.DataFrame({
        "question1": test1["question1"],
        "question2": test1["question2"],
        "label": test1["label"],
        "q1_char_count": test1["q1_char_count"],
        "q2_char_count": test1["q2_char_count"],
        "char_count_diff": test1["char_count_diff"],
        "char_overlap": test1["char_overlap"],
        "q1_hash": test3["q1_hash"],
        "q2_hash": test3["q2_hash"],
        "q1_freq": test3["q1_freq"],
        "q2_freq": test3["q2_freq"],
        "q1_q2_intersect": test6["q1_q2_intersect"],
        "fuzzy_qratio": test4["fuzzy_qratio"],
        "fuzzy_wratio": test4["fuzzy_wratio"],
        "fuzzy_partial_ratio": test4["fuzzy_partial_ratio"],
        "fuzzy_partial_token_set_ratio": test4["fuzzy_partial_token_set_ratio"],
        "fuzzy_partial_token_sort_ratio": test4["fuzzy_partial_token_sort_ratio"],
        "fuzzy_token_set_ratio": test4["fuzzy_token_set_ratio"],
        "fuzzy_token_sort_ratio": test4["fuzzy_token_sort_ratio"],
        "char_match": test2["char_match"],
        "tfidf_char_match": test2["tfidf_char_match"],
        "uni_BLEU": test1["uni_BLEU"],
        "bi_BLEU": test1["bi_BLEU"],
        "BLEU2": test1["BLEU2"],
        "cosine_distance": test5["cosine_distance"],
        'cityblock_distance': test5['cityblock_distance'],
        'jaccard_distance': test5['jaccard_distance'],
        'canberra_distance': test5['canberra_distance'],
        'euclidean_distance': test5['euclidean_distance'],
        'minkowski_distance': test5['minkowski_distance'],
        'braycurtis_distance': test5['braycurtis_distance'],
        'skew_q1vec': test5['skew_q1vec'],
        'skew_q2vec': test5['skew_q2vec'],
        'kur_q1vec': test5['kur_q1vec']

    })

    save_path = "./dataset/dataset_reform/base_char/features_merged/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data_frame_train.to_csv("./dataset/dataset_reform/base_char/features_merged/train.csv",
                            index=False, columns=columns, encoding='utf-8')
    data_frame_test.to_csv("./dataset/dataset_reform/base_char/features_merged/test.csv",
                           index=False, columns=columns, encoding='utf-8')
    logger.info("Complete merge features,and saved path: ./dataset/dataset_reform/base_char/features_merged")


class data_analysis_base_word():

    #def __init__(self, train_df, test_df):
        ##The below analyse base on word
        #Analysis stage start:
        # self.train_df = pd.read_csv("./dataset/dataset_reform/train_reform_base_word.csv", encoding="utf-8")
        # self.test_df = pd.read_csv("./dataset/dataset_reform/test_reform_base_word.csv", encoding="utf-8")
        # self.train_qs = pd.Series(self.train_df['question1'].tolist() + self.train_df['question2'].tolist()).astype(str)
        # self.test_qs = pd.Series(self.test_df['question1'].tolist() + self.test_df['question2'].tolist()).astype(str)
    def simple_analyse(self, train_df, test_df):
        print('Total number of question pairs for training: {}'.format(len(train_df)))
        print('Duplicate pairs: {}%'.format(round(train_df['label'].mean() * 100, 2)))
        qids = pd.Series(train_df['question1'].tolist() + train_df['question2'].tolist())
        print('Total number of questions in the training data: {}'.format(len(np.unique(qids))))
        print('Number of questions that appear multiple times: {}'.format(np.sum(qids.value_counts() > 1)))

        plt.figure(figsize=(12, 5))
        plt.hist(qids.value_counts(), bins=30, align='mid')
        plt.yscale('log', nonposy='clip')
        plt.title('Log-Histogram of question appearance counts')
        plt.xlabel('Number of occurences of question')
        plt.ylabel('Number of questions')
        #plt.show()
        train_qs = pd.Series(train_df['question1'].tolist() + train_df['question2'].tolist()).astype(str)
        test_qs = pd.Series(test_df['question1'].tolist() + test_df['question2'].tolist()).astype(str)
        dist_train = train_qs.apply(lambda x: len(x.split(' ')))
        dist_test = test_qs.apply(lambda x: len(x.split(' ')))

        plt.figure(figsize=(15, 10))
        plt.hist(dist_train, bins=50, range=[0, 50], color=color[2], density=True, label='train')
        plt.hist(dist_test, bins=50, range=[0, 50], color=color[1], density=True, alpha=0.5, label='test')
        plt.title('Normalised histogram of word count in questions (base on word)', fontsize=15)
        plt.legend()
        plt.xlabel('Number of words', fontsize=15)
        plt.ylabel('Probability', fontsize=15)
        #plt.show()
        print('mean-train {:.2f} std-train {:.2f} min-train {:.2f} max-train {:.2f} '
              'mean-test {:.2f} std-test {:.2f} min-test {:.2f} max-test {:.2f}'.format(
              dist_train.mean(), dist_train.std(), dist_train.min(), dist_train.max(),
              dist_test.mean(), dist_test.std(), dist_test.min(), dist_test.max()))
        """ We see a similar distribution for word count, with most questions being about 10 words long.
        It looks to me like the distribution of the training set seems more "pointy", while on the 
        test set it is wider.Nevertheless, they are quite similar.
        So what are the most common words? Let's take a look at a word cloud"""

        cloud = WordCloud(width=1440, height=1080).generate(" ".join(train_qs.astype(str)))
        plt.figure(figsize=(20, 15))
        plt.imshow(cloud)
        plt.axis('off')
        #plt.show()

    def get_word_count(self, train_path="./dataset/dataset_reform/base_word/add_8features/train_8features_base_word.csv",
                       test_path="./dataset/dataset_reform/base_word/add_8features/test_8features_base_word.csv"):
        train_df = pd.read_csv(train_path, encoding='utf-8')
        test_df = pd.read_csv(test_path, encoding='utf-8')
        word_count_train = pd.Series(train_df["q1_word_count"].tolist() + train_df["q2_word_count"].tolist())
        word_count_test = pd.Series(test_df["q1_word_count"].tolist() + test_df["q2_word_count"].tolist())
        print("word_count_train: " + ' ' + "min:" + str(word_count_train.min()) + ' '
              + "max:" + str(word_count_train.max()) + ' '
              + "mean:" + str(word_count_train.mean()) + ' '
              + "std:" + str(word_count_train.std()))
        print("word_count_train: " + ' ' + "min:" + str(word_count_test.min()) + ' '
              + "max:" + str(word_count_test.max()) + ' '
              + "mean:" + str(word_count_test.mean()) + ' '
              + "std:" + str(word_count_test.std()))

    def word_match_share(self, row):
        stops = []
        q1words = {}
        q2words = {}
        for word in str(row['question1']).split():
            if word not in stops:  # 非停用词，保存到字典
                q1words[word] = 1
        for word in str(row['question2']).split():
            if word not in stops:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # The computer-generated chaff includes a few questions that are nothing but stopwords
            return 0
        shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
        shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
        R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))  # 计算q1和q2的非停用词相似度
        return R


    # TF-IDF 特征的提取
    def get_weight(self, count, eps=3000, min_count=2):
        if count < min_count:
            return 0
        else:
            return 1 / (count + eps)

    def tfidf_word_match_share(self, row):
        stops = []
        q1words = {}
        q2words = {}
        for word in str(row['question1']).split():
            if word not in stops:
                q1words[word] = 1
        for word in str(row['question2']).split():
            if word not in stops:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # The computer-generated chaff includes a few questions that are nothing but stopwords
            return 0
        # 重新计算两个句子的相似度，加上权tfidf权值...
        shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                        q2words.keys() if w in q1words]
        total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

        R = np.sum(shared_weights) / np.sum(total_weights)
        return R

    def try_apply_dict(x, dict_to_apply):
        try:
            return dict_to_apply[x]
        except KeyError:
            return 0

    def q1_q2_intersect(self, row):
        return (len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))


class data_analysis_base_char():

    def simple_analyse(self, train_df, test_df):
        print('(Base on char) Total number of question pairs for training: {}'.format(len(train_df)))
        print('(Base on char) Duplicate pairs: {}%'.format(round(train_df['label'].mean() * 100, 2)))
        qids = pd.Series(train_df['question1'].tolist() + train_df['question2'].tolist())
        print('(Base on char) Total number of questions in the training data: {}'.format(len(np.unique(qids))))
        print('(Base on char) Number of questions that appear multiple times: {}'.format(np.sum(qids.value_counts() > 1)))

        plt.figure(figsize=(12, 5))
        plt.hist(qids.value_counts(), bins=30, align='mid')
        plt.yscale('log', nonposy='clip')
        plt.title('Log-Histogram of question appearance counts (base on char)')
        plt.xlabel('Number of occurences of question (base on char)')
        plt.ylabel('Number of questions (base on char)')
        plt.show()
        train_qs = pd.Series(train_df['question1'].tolist() + train_df['question2'].tolist()).astype(str)
        test_qs = pd.Series(test_df['question1'].tolist() + test_df['question2'].tolist()).astype(str)
        dist_train = train_qs.apply(lambda x: len(x.split(' ')))
        dist_test = test_qs.apply(lambda x: len(x.split(' ')))

        plt.figure(figsize=(15, 10))
        plt.hist(dist_train, bins=50, range=[0, 50], color=color[2], density=True, label='train')
        plt.hist(dist_test, bins=50, range=[0, 50], color=color[1], density=True, alpha=0.5, label='test')
        plt.title('Normalised histogram of char count in questions (base on char)', fontsize=15)
        plt.legend()
        plt.xlabel('Number of chars', fontsize=15)
        plt.ylabel('Probability', fontsize=15)
        plt.show()
        print('mean-train {:.2f} std-train {:.2f} min-train {:.2f} max-train {:.2f} '
              'mean-test {:.2f} std-test {:.2f} min-test {:.2f} max-test {:.2f}'.format(
            dist_train.mean(), dist_train.std(), dist_train.min(), dist_train.max(),
            dist_test.mean(), dist_test.std(), dist_test.min(), dist_test.max()))
        """ We see a similar distribution for word count, with most questions being about 10 words long.
        It looks to me like the distribution of the training set seems more "pointy", while on the 
        test set it is wider.Nevertheless, they are quite similar.
        So what are the most common words? Let's take a look at a word cloud"""

        cloud = WordCloud(width=1440, height=1080).generate(" ".join(train_qs.astype(str)))
        plt.figure(figsize=(20, 15))
        plt.imshow(cloud)
        plt.axis('off')
        plt.show()

    def get_word_count(self,
                       train_path="./dataset/dataset_reform/base_char/add_8features/train_8features_base_char.csv",
                       test_path="./dataset/dataset_reform/base_word/add_8features/test_8features_base_char.csv"):
        train_df = pd.read_csv(train_path, encoding='utf-8')
        test_df = pd.read_csv(test_path, encoding='utf-8')
        word_count_train = pd.Series(train_df["q1_word_count"].tolist() + train_df["q2_word_count"].tolist())
        word_count_test = pd.Series(test_df["q1_word_count"].tolist() + test_df["q2_word_count"].tolist())
        print("word_count_train: " + ' ' + "min:" + str(word_count_train.min()) + ' '
              + "max:" + str(word_count_train.max()) + ' '
              + "mean:" + str(word_count_train.mean()) + ' '
              + "std:" + str(word_count_train.std()))
        print("word_count_train: " + ' ' + "min:" + str(word_count_test.min()) + ' '
              + "max:" + str(word_count_test.max()) + ' '
              + "mean:" + str(word_count_test.mean()) + ' '
              + "std:" + str(word_count_test.std()))

    def char_match_share(self, row):
        stops = []
        q1words = {}
        q2words = {}
        for word in str(row['question1']).split():
            if word not in stops:  # 非停用词，保存到字典
                q1words[word] = 1
        for word in str(row['question2']).split():
            if word not in stops:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # The computer-generated chaff includes a few questions that are nothing but stopwords
            return 0
        shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
        shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
        R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))  # 计算q1和q2的非停用词相似度
        return R

        # TF-IDF 特征的提取

    def get_weight(self, count, eps=3000, min_count=2):
        if count < min_count:
            return 0
        else:
            return 1 / (count + eps)

    def tfidf_char_match_share(self, row):
        stops = []
        q1words = {}
        q2words = {}
        for word in str(row['question1']).split():
            if word not in stops:
                q1words[word] = 1
        for word in str(row['question2']).split():
            if word not in stops:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # The computer-generated chaff includes a few questions that are nothing but stopwords
            return 0
        # 重新计算两个句子的相似度，加上权tfidf权值...
        shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                        q2words.keys() if w in q1words]
        total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

        R = np.sum(shared_weights) / np.sum(total_weights)
        return R

    def try_apply_dict(x, dict_to_apply):
        try:
            return dict_to_apply[x]
        except KeyError:
            return 0

    def q1_q2_intersect(self, row):
        return (len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))

    def plot_number_of_chars_in_question(self, train_df_path):
        train_df = pd.read_csv(train_df_path, encoding='utf-8')
        all_ques_df = pd.DataFrame(pd.concat([train_df['question1'], train_df['question2']]))
        all_ques_df.columns = ["questions"]
        all_ques_df["num_of_chars"] = all_ques_df["questions"].apply(lambda x: len(str(x).split()))
        cnt_srs = all_ques_df['num_of_chars'].value_counts()

        plt.figure(figsize=(50, 8))
        sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])
        plt.ylabel('Number of Occurrences', fontsize=12)
        plt.xlabel('Number of chars in the question', fontsize=12)
        plt.title("Number of chars in the question in trainseting (base on char)")
        plt.xticks(rotation='vertical')
        plt.show()



if __name__=='__main__':

    ##The below analyse base on word.
    df_train = pd.read_csv("./dataset/dataset_reform/base_word/train_reform_base_word.csv", encoding="utf-8")
    df_test = pd.read_csv("./dataset/dataset_reform/base_word/test_reform_base_word.csv", encoding="utf-8")
    train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
    test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)

    data_analysis_base_word = data_analysis_base_word()
    data_analysis_base_word.simple_analyse(df_train, df_test)

    ## TF-IDF 特征的提取
    eps = 1000
    words = (" ".join(train_qs)).split()  # 所有出现的词语
    counts = Counter(words)  # 单词计数
    print(len(counts))
    # output:7611
    weights = {word: data_analysis_base_word.get_weight(count, eps=eps) for word, count in counts.items()}  # 计算权值，出现次数越小，权值越大。只出现一次的单词舍去
    print('Most common words and weights: \n')
    print(sorted(weights.items(), key=lambda x: x[1] if x[1] > 0 else 9999)[:10])
    print('\nLeast common words and weights: ')
    print((sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10]))

    """
    Most common words and weights:

    [('W104416', 6.97350069735007e-05), ('W107170', 0.00010315659170621003), ('W105754', 0.0001124985937675779), ('W106503', 0.00011595547309833024), ('W100806', 0.00011966016513102789), ('W100323', 0.00012425447316103378), ('W100914', 0.00012933264355923435), ('W108280', 0.00014664906877841325), ('W101396', 0.00014940983116689077), ('W104806', 0.00015586034912718204)]

    Least common words and weights:
    [('W107941', 0.000998003992015968), ('W108389', 0.000998003992015968), ('W103927', 0.000998003992015968), ('W100516', 0.000998003992015968), ('W101355', 0.000998003992015968), ('W101777', 0.000998003992015968), ('W108879', 0.000998003992015968), ('W105551', 0.000998003992015968), ('W108615', 0.000998003992015968), ('W106301', 0.000998003992015968)]
    """
    ##Label distribution over word_match_share
    plt.figure(figsize=(15, 5))
    train_word_match = df_train.apply(data_analysis_base_word.word_match_share, axis=1, raw=True)
    plt.hist(train_word_match[df_train['label'] == 0], bins=20, density=True, label='Not Duplicate')
    plt.hist(train_word_match[df_train['label'] == 1], bins=20, density=True, alpha=0.7, label='Duplicate')
    plt.legend()
    plt.title('Label distribution over word_match_share', fontsize=15)
    plt.xlabel('word_match_share', fontsize=15)
    #plt.show()

    plt.figure(figsize=(15, 5))
    tfidf_train_word_match = df_train.apply(data_analysis_base_word.tfidf_word_match_share, axis=1, raw=True)
    plt.hist(tfidf_train_word_match[df_train['label'] == 0].fillna(0), bins=20, density=True,
             label='Not Duplicate')
    plt.hist(tfidf_train_word_match[df_train['label'] == 1].fillna(0), bins=20, density=True, alpha=0.7,
             label='Duplicate')
    plt.legend()
    plt.title('Label distribution over tfidf_word_match_share', fontsize=15)
    plt.xlabel('word_match_share', fontsize=15)
    #plt.show()

    print('Original AUC:', roc_auc_score(df_train['label'], train_word_match))
    print('   TFIDF AUC:', roc_auc_score(df_train['label'], tfidf_train_word_match.fillna(0)))

    """Original AUC: 0.55063027
          TFIDF AUC: 0.641369045
    """

    ## First we create our training and testing data
    x_train = df_train[:]
    x_test = df_test[:]
    x_train['word_match'] = train_word_match
    x_train['tfidf_word_match'] = tfidf_train_word_match
    x_test['word_match'] = df_test.apply(data_analysis_base_word.word_match_share, axis=1, raw=True)
    x_test['tfidf_word_match'] = df_test.apply(data_analysis_base_word.tfidf_word_match_share, axis=1, raw=True)

    y_train = df_train['label'].values

    print("x_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("x_test:", x_test.shape)
    x_train['label'] = y_train
    x_train.to_csv('./dataset/dataset_reform/base_word/add_feature1/train_add_feature1_base_word.csv',
                   index=False, encoding="utf-8")
    x_test.to_csv('./dataset/dataset_reform/base_word/add_feature1/test_add_feature1_base_word.csv',
                  index=False, encoding="utf-8")

    #extract 4features-'q1_hash','q2_hash','q1_freq','q2_freq'
    train_orig = pd.read_csv("./dataset/dataset_reform/base_word/train_reform_base_word.csv", encoding="utf-8", header=0)
    test_orig = pd.read_csv("./dataset/dataset_reform/base_word/test_reform_base_word.csv", encoding="utf-8", header=0)

    tic0 = timeit.default_timer()
    df1 = train_orig[['question1']].copy()
    df2 = train_orig[['question2']].copy()
    df1_test = test_orig[['question1']].copy()
    df2_test = test_orig[['question2']].copy()
    df2.rename(columns={'question2': 'question1'}, inplace=True)
    df2_test.rename(columns={'question2': 'question1'}, inplace=True)

    train_questions = df1.append(df2)
    print(train_questions.shape) #output:(40000, 1)
    train_questions = train_questions.append(df1_test)
    print(train_questions.shape) #output:(50000, 1)
    train_questions = train_questions.append(df2_test)
    print(train_questions.shape) #output:(60000, 1)

    train_questions.drop_duplicates(subset=['question1'], inplace=True)
    print(train_questions.shape) #output:(35268, 1)

    train_questions.reset_index(inplace=True, drop=True)
    questions_dict = pd.Series(train_questions.index.values, index=train_questions.question1.values).to_dict()

    train_cp = train_orig.copy()
    test_cp = test_orig.copy()

    test_cp['label'] = -1
    comb = pd.concat([train_cp, test_cp])

    comb['q1_hash'] = comb['question1'].map(questions_dict)
    comb['q2_hash'] = comb['question2'].map(questions_dict)

    q1_vc = comb.q1_hash.value_counts().to_dict()  # 计算一个Series中各词出现的频率
    q2_vc = comb.q2_hash.value_counts().to_dict()

    comb['q1_freq'] = comb['q1_hash'].map(lambda x: data_analysis_base_word.try_apply_dict(x, q1_vc) +
                                                    data_analysis_base_word.try_apply_dict(x, q2_vc))
    comb['q2_freq'] = comb['q2_hash'].map(lambda x: data_analysis_base_word.try_apply_dict(x, q1_vc) +
                                                    data_analysis_base_word.try_apply_dict(x, q2_vc))
    train_comb = comb[comb['label'] >= 0][['question1', 'question2', 'label', 'q1_hash', 'q2_hash', 'q1_freq', 'q2_freq']]
    test_comb = comb[comb['label'] < 0][['question1', 'question2', 'label', 'q1_hash', 'q2_hash', 'q1_freq', 'q2_freq']]
    print(train_comb.shape) #output:(20000, 7)
    print(test_comb.shape)  #output:(10000, 7)
    train_comb.to_csv('./dataset/dataset_reform/base_word/add_feature2/train_add_feature2_base_word.csv',
                      index=False, encoding="utf-8")
    test_comb.to_csv('./dataset/dataset_reform/base_word/add_feature2/test_add_feature2_base_word.csv',
                     index=False, encoding="utf-8")
    test_comb_empty_label = pd.read_csv('./dataset/dataset_reform/base_word/add_feature2/test_add_feature2_base_word.csv',
                                        encoding='utf-8')
    temp = []
    test_comb_empty_label["label"] = pd.Series(temp)
    test_comb_empty_label.to_csv('./dataset/dataset_reform/base_word/add_feature2/test_add_feature2_base_word.csv',
                                 index=False, encoding="utf-8")
    feature2_save_path = './dataset/dataset_reform/base_word/add_feature2/'

    # train_corr_mat = train_comb.corr()  # 关于相关系数coefficient的计算可以调用pearson方法或kendell方法或spearman方法，默认使用pearson方法。
    # test_corr_mat = test_comb.corr()
    # train_corr_mat.to_csv('./dataset/dataset_reform/base_word/add_feature2/train_add_feature2_coefficient_base_word.csv',
    #                        index=False, encoding="utf-8")
    # test_corr_mat.to_csv('./dataset/dataset_reform/base_word/add_feature2/test_add_feature2_coefficient_base_word.csv',
    #                      index=False, encoding="utf-8")
    # print(corr_mat.head(10))
    logger.info("Feature2 is done ! saved path: %s" % feature2_save_path)

    #sorting trainseting and testset
    sorting_trainset_testset(train_path='./dataset/dataset_reform/base_word/add_8features/train_8features_base_word.csv',
                             test_path='./dataset/dataset_reform/base_word/add_8features/test_8features_base_word.csv')
    logger.info("Complete sort trainseting and testset,and saved path: ./dataset/dataset_reform/base_word/add_8features")

    #plot q1_q2_intersect
    train_origin = pd.read_csv("./dataset/dataset_reform/base_word/train_reform_base_word.csv", encoding="utf-8", header=0)
    test_origin = pd.read_csv("./dataset/dataset_reform/base_word/test_reform_base_word.csv", encoding="utf-8", header=0)
    questions = pd.concat([train_origin[['question1', 'question2']],
                           test_origin[['question1', 'question2']]], axis=0).reset_index(drop='index')
    print(questions.shape)

    q_dict = defaultdict(set)
    for i in range(questions.shape[0]):
        q_dict[questions.question1[i]].add(questions.question2[i])
        q_dict[questions.question2[i]].add(questions.question1[i])

    train_origin['q1_q2_intersect'] = train_origin.apply(data_analysis_base_word.q1_q2_intersect, axis=1, raw=True)
    test_origin['q1_q2_intersect'] = test_origin.apply(data_analysis_base_word.q1_q2_intersect, axis=1, raw=True)
    temp = train_origin.q1_q2_intersect.value_counts()
    sns.barplot(temp.index[:20], temp.values[:20])
    plt.show(sns)
    train_origin['q1_q2_intersect'] = train_origin['q1_q2_intersect']
    test_origin['q1_q2_intersect'] = test_origin['q1_q2_intersect']
    train_origin.to_csv("./dataset/dataset_reform/base_word/add_8features/"
                        "train_add_single_feature_q1_q2_intersect.csv", index=False)
    test_origin.to_csv("./dataset/dataset_reform/base_word/add_8features/"
                       "test_add_single_feature_q1_q2_intersect.csv", index=False)
    logger.info("Complete plot q1_q2_intersect,and saved path: ./dataset/data_analysis/base_word/")
    logger.info("(train_add_single_feature_q1_q2_intersect.csv)"
                " and (test_add_single_feature_q1_q2_intersect.csv) saved path: "
                "./dataset/dataset_reform/base_word/add_8features/")


    ###The below analyse base on char
    logger.info("Start data analysis base on char...")
    data_analysis_base_char = data_analysis_base_char()
    ##plot unmber of chars in question
    data_analysis_base_char.plot_number_of_chars_in_question(train_df_path=
                                                            "dataset/dataset_reform/base_char/train_reform_base_char.csv")

    logger.info("Generating 7 features base on char ...")
    generate_7features_base_char(train_data_path_base_char="./dataset/dataset_reform/base_char/train_reform_base_char.csv",
                                 test_data_path_base_char="./dataset/dataset_reform/base_char/test_reform_base_char.csv")
    logger.info("Complete generate 7 features base on char, saved path: ./dataset/dataset_reform/base_char/add_7features")


    df_train_base_char = pd.read_csv("./dataset/dataset_reform/base_char/train_reform_base_char.csv", encoding="utf-8")
    df_test_base_char = pd.read_csv("./dataset/dataset_reform/base_char/test_reform_base_char.csv", encoding="utf-8")
    train_qs_base_char = pd.Series(df_train_base_char['question1'].tolist() + df_train_base_char['question2'].tolist()).astype(str)
    test_qs_base_char = pd.Series(df_test_base_char['question1'].tolist() + df_test_base_char['question2'].tolist()).astype(str)

    data_analysis_base_char.simple_analyse(df_train_base_char, df_test_base_char)

    ## TF-IDF 特征的提取
    eps = 1000
    words = (" ".join(train_qs_base_char)).split()  # 所有出现的词语
    counts = Counter(words)  # 单词计数
    print(len(counts))
    # output:7611
    weights = {word: data_analysis_base_char.get_weight(count, eps=eps) for word, count in counts.items()}  # 计算权值，出现次数越小，权值越大。只出现一次的单词舍去
    print('Most common chars and weights: \n')
    print(sorted(weights.items(), key=lambda x: x[1] if x[1] > 0 else 9999)[:10])
    print('\nLeast common chars and weights: ')
    print((sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10]))
    #output:
    """
    Most common chars and weights:
    [('C100114', 5.0584248065152515e-05), ('C100291', 6.006006006006006e-05), ('C101318', 6.205013651030033e-05),
     ('C101643', 6.563402467839328e-05), ('C101061', 6.832001093120175e-05), ('C101350', 6.932889628397117e-05),
     ('C102221', 9.292816652727442e-05), ('C100800', 9.926543577526305e-05), ('C100376', 0.00010097950116126426),
     ('C100882', 0.00010940919037199125)]

    Least common chars and weights:
    [('C100529', 0.000998003992015968), ('C100925', 0.000998003992015968), ('C100462', 0.000998003992015968),
     ('C101233', 0.000998003992015968), ('C100786', 0.000998003992015968), ('C101150', 0.000998003992015968),
     ('C101240', 0.000998003992015968), ('C100453', 0.000998003992015968), ('C100366', 0.000998003992015968),
     ('C100946', 0.000998003992015968)]
    """

    ##Label distribution over char_match_share
    plt.figure(figsize=(15, 5))
    train_char_match = df_train_base_char.apply(data_analysis_base_char.char_match_share, axis=1, raw=True)
    plt.hist(train_char_match[df_train_base_char['label'] == 0], bins=20, density=True, label='Not Duplicate')
    plt.hist(train_char_match[df_train_base_char['label'] == 1], bins=20, density=True, alpha=0.7, label='Duplicate')
    plt.legend()
    plt.title('Label distribution over char_match_share', fontsize=15)
    plt.xlabel('char_match_share', fontsize=15)
    plt.show()

    plt.figure(figsize=(15, 5))
    tfidf_train_char_match = df_train_base_char.apply(data_analysis_base_char.tfidf_char_match_share, axis=1, raw=True)
    plt.hist(tfidf_train_char_match[df_train_base_char['label'] == 0].fillna(0), bins=20, density=True,
             label='Not Duplicate')
    plt.hist(tfidf_train_char_match[df_train_base_char['label'] == 1].fillna(0), bins=20, density=True, alpha=0.7,
             label='Duplicate')
    plt.legend()
    plt.title('Label distribution over tfidf_char_match_share', fontsize=15)
    plt.xlabel('char_match_share', fontsize=15)
    plt.show()

    print('Original AUC:', roc_auc_score(df_train_base_char['label'], train_char_match))
    print('   TFIDF AUC:', roc_auc_score(df_train_base_char['label'], tfidf_train_char_match.fillna(0)))

    """Original AUC: 0.58355663
          TFIDF AUC: 0.687069685
    """

    ## Now we create our training and testing data
    x_train_base_char = df_train_base_char[:]
    x_test_base_char = df_test_base_char[:]
    x_train_base_char['char_match'] = train_char_match
    x_train_base_char['tfidf_char_match'] = tfidf_train_char_match
    x_test_base_char['char_match'] = df_test_base_char.apply(data_analysis_base_char.char_match_share, axis=1, raw=True)
    x_test_base_char['tfidf_char_match'] = df_test_base_char.apply(data_analysis_base_char.tfidf_char_match_share, axis=1, raw=True)

    y_train_base_char = df_train_base_char['label'].values

    print("x_train:", x_train_base_char.shape) #x_train: (20000, 5)
    print("y_train:", y_train_base_char.shape) #y_train: (20000,)
    print("x_test:", x_test_base_char.shape)   #x_test: (10000, 5)
    x_train_base_char['label'] = y_train_base_char
    x_train_base_char.to_csv('./dataset/dataset_reform/base_char/add_feature1/train_add_feature1_base_char.csv', index=False,
                             encoding="utf-8")
    x_test_base_char.to_csv('./dataset/dataset_reform/base_char/add_feature1/test_add_feature1_base_char.csv', index=False,
                             encoding="utf-8")

    # extract 4features-'q1_hash','q2_hash','q1_freq','q2_freq'
    train_orig_base_char = pd.read_csv("./dataset/dataset_reform/base_char/train_reform_base_char.csv", encoding="utf-8",
                                       header=0)
    test_orig_base_char = pd.read_csv("./dataset/dataset_reform/base_char/test_reform_base_char.csv", encoding="utf-8",
                                      header=0)

    tic0 = timeit.default_timer()
    df1 = train_orig_base_char[['question1']].copy()
    df2 = train_orig_base_char[['question2']].copy()
    df1_test = test_orig_base_char[['question1']].copy()
    df2_test = test_orig_base_char[['question2']].copy()
    df2.rename(columns={'question2': 'question1'}, inplace=True)
    df2_test.rename(columns={'question2': 'question1'}, inplace=True)

    train_questions_base_char = df1.append(df2)
    print(train_questions_base_char.shape)  # output:(40000, 1)
    train_questions_base_char = train_questions_base_char.append(df1_test)
    print(train_questions_base_char.shape)  # output:(50000, 1)
    train_questions_base_char = train_questions_base_char.append(df2_test)
    print(train_questions_base_char.shape)  # output:(60000, 1)

    train_questions_base_char.drop_duplicates(subset=['question1'], inplace=True)
    print(train_questions_base_char.shape)  # output:(35266, 1)

    train_questions_base_char.reset_index(inplace=True, drop=True)
    questions_dict = pd.Series(train_questions_base_char.index.values, index=train_questions_base_char.question1.values).to_dict()

    train_cp = train_orig_base_char.copy()
    test_cp = test_orig_base_char.copy()

    test_cp['label'] = -1
    comb = pd.concat([train_cp, test_cp])

    comb['q1_hash'] = comb['question1'].map(questions_dict)
    comb['q2_hash'] = comb['question2'].map(questions_dict)

    q1_vc = comb.q1_hash.value_counts().to_dict()  # 计算一个Series中各词出现的频率
    q2_vc = comb.q2_hash.value_counts().to_dict()

    comb['q1_freq'] = comb['q1_hash'].map(lambda x: data_analysis_base_word.try_apply_dict(x, q1_vc) +
                                                    data_analysis_base_word.try_apply_dict(x, q2_vc))
    comb['q2_freq'] = comb['q2_hash'].map(lambda x: data_analysis_base_word.try_apply_dict(x, q1_vc) +
                                                    data_analysis_base_word.try_apply_dict(x, q2_vc))
    train_comb = comb[comb['label'] >= 0][
        ['question1', 'question2', 'label', 'q1_hash', 'q2_hash', 'q1_freq', 'q2_freq']]
    test_comb = comb[comb['label'] < 0][['question1', 'question2', 'label', 'q1_hash', 'q2_hash', 'q1_freq', 'q2_freq']]
    print(train_comb.shape)  # output:(20000, 7)
    print(test_comb.shape)  # output:(10000, 7)
    train_comb.to_csv('./dataset/dataset_reform/base_char/add_feature2/train_add_feature2_base_char.csv',
                      index=False, encoding="utf-8")
    test_comb.to_csv('./dataset/dataset_reform/base_char/add_feature2/test_add_feature2_base_char.csv',
                     index=False, encoding="utf-8")
    test_comb_empty_label = pd.read_csv('./dataset/dataset_reform/base_char/add_feature2/test_add_feature2_base_char.csv',
                                         encoding='utf-8')
    temp = []
    test_comb_empty_label["label"] = pd.Series(temp)
    test_comb_empty_label.to_csv('./dataset/dataset_reform/base_char/add_feature2/test_add_feature2_base_char.csv',
                                 index=False, encoding="utf-8")
    feature2_save_path = './dataset/dataset_reform/base_char/add_feature2/'

    # train_corr_mat = train_comb.corr()  # 关于相关系数coefficient的计算可以调用pearson方法或kendell方法或spearman方法，默认使用pearson方法。
    # test_corr_mat = test_comb.corr()
    # train_corr_mat.to_csv('./dataset/dataset_reform/base_word/add_feature2/train_add_feature2_coefficient_base_word.csv',
    #                        index=False, encoding="utf-8")
    # test_corr_mat.to_csv('./dataset/dataset_reform/base_word/add_feature2/test_add_feature2_coefficient_base_word.csv',
    #                      index=False, encoding="utf-8")
    # print(corr_mat.head(10))
    logger.info("(Base on char) Feature2 is done ! saved path: %s" % feature2_save_path)

    # # sorting trainseting and testset
    # sorting_trainset_testset(
    #     train_path='./dataset/dataset_reform/base_word/add_8features/train_8features_base_word.csv',
    #     test_path='./dataset/dataset_reform/base_word/add_8features/test_8features_base_word.csv')
    # logger.info(
    #     "Complete sort trainseting and testset,and saved path: ./dataset/dataset_reform/base_word/add_8features")

    # plot q1_q2_intersect
    train_origin_base_char = pd.read_csv("./dataset/dataset_reform/base_char/train_reform_base_char.csv", encoding="utf-8",
                               header=0)
    test_origin_base_char = pd.read_csv("./dataset/dataset_reform/base_char/test_reform_base_char.csv", encoding="utf-8",
                              header=0)
    questions = pd.concat([train_origin_base_char[['question1', 'question2']],
                           test_origin_base_char[['question1', 'question2']]], axis=0).reset_index(drop='index')
    print(questions.shape) #(30000,2)

    q_dict = defaultdict(set)
    for i in range(questions.shape[0]):
        q_dict[questions.question1[i]].add(questions.question2[i])
        q_dict[questions.question2[i]].add(questions.question1[i])

    train_origin_base_char['q1_q2_intersect'] = train_origin_base_char.apply(data_analysis_base_char.q1_q2_intersect,
                                                                             axis=1, raw=True)
    test_origin_base_char['q1_q2_intersect'] = test_origin_base_char.apply(data_analysis_base_char.q1_q2_intersect,
                                                                           axis=1, raw=True)
    temp = train_origin_base_char.q1_q2_intersect.value_counts()
    sns.barplot(temp.index[:20], temp.values[:20])
    plt.show(sns)
    train_origin_base_char['q1_q2_intersect'] = train_origin_base_char['q1_q2_intersect']
    test_origin_base_char['q1_q2_intersect'] = test_origin_base_char['q1_q2_intersect']
    train_origin_base_char.to_csv("./dataset/dataset_reform/base_char/add_7features/"
                        "train_add_single_feature_q1_q2_intersect.csv", index=False)
    test_origin_base_char.to_csv("./dataset/dataset_reform/base_char/add_7features/"
                       "test_add_single_feature_q1_q2_intersect.csv", index=False)
    logger.info("(Base on char) Complete plot q1_q2_intersect,and saved path: ./dataset/data_analysis/base_char/")
    logger.info("(train_add_single_feature_q1_q2_intersect.csv)"
                " and (test_add_single_feature_q1_q2_intersect.csv) saved path: "
                "./dataset/dataset_reform/base_char/add_7features/")

    generate_8features_base_word()
    generate_7features_base_char()
    generate_magic_features_base_word()
    generate_magic_features_base_char()
    generate_distance_features_base_word()
    generate_distance_features_base_char()

    features_merge_base_word()
    features_merge_base_char()
    logger.info('Complete all features generate and merged,saved path: ./dataset/dataset_reform/base_word/'
                'features_merged and ./dataset/dataset_reform/base_char/features_merged')