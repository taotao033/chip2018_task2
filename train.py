'''
Tested under Keras 2.2.4 with Tensorflow-gpu 1.9 backend
'''

########################################
## import packages
########################################
import os
import time
import csv
import codecs
import numpy as np
import pandas as pd
import json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from data_preprocess import build_word_embedding_dict, build_char_embedding_dict
from attention_model import decomposable_attention
from sklearn.preprocessing import StandardScaler

########################################
## set directories and parameters
########################################
date = '20181126'
number_of_experiment = 5
EMBEDDING_FILE = './dataset/word_embedding.txt'
EMBEDDING_FILE_base_char = './dataset/char_embedding.txt'
TRAIN_DATA_FILE = "./dataset/dataset_reform/base_word/features_merged/train.csv"
TEST_DATA_FILE = "./dataset/dataset_reform/base_word/features_merged/test.csv"
TRAIN_DATA_FILE_base_char = "./dataset/dataset_reform/base_char/features_merged/train.csv"
TEST_DATA_FILE_base_char = "./dataset/dataset_reform/base_char/features_merged/test.csv"
EMBEDDING_DIM = 300
MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 35 #{min:, max:}
Magic_feat_length = 18
Dis_feat_length = 10
projection_dim = 300
projection_hidden = 0
projection_dropout = 0.2
compare_dim = 500
compare_dropout = 0.2
dense_dim = 300
dense_dropout = 0.2
lr = 1e-3
activation = 'elu'
batch_size = 32
VALIDATION_SPLIT = 0.1
weight_0 = 1
weight_1 = 1
weight_ratio = weight_0 / weight_1

re_weight = True # whether to re-weight classes to fit the % share in test set

STAMP = 'train_decomposable_soft_attention_batch_size_%d_embedding_dim_%d_projection_dim_%d_' \
        'projection_hidden_%d_projection_dropout_%.2f_compare_dim_%d_compare_dropout_%.2f_dense_dim_%d_' \
        'dense_dropout_%.2f_magic_feat_len_%d_dis_feat_len_%d_maxlen_%d_weight_ratio_%.2f' \
        % (batch_size, EMBEDDING_DIM, projection_dim, projection_hidden, projection_dropout, compare_dim,
           compare_dropout, dense_dim, dense_dropout, Magic_feat_length, Dis_feat_length,
           MAX_SEQUENCE_LENGTH, weight_ratio)

########################################
## index word vectors
########################################
print('Indexing word vectors')

embeddings_index = build_word_embedding_dict(path=EMBEDDING_FILE)
embeddings_index_base_char = build_char_embedding_dict(path=EMBEDDING_FILE_base_char)

print('Found %d word vectors of word_embedding.txt.' % len(embeddings_index))
print('Found %d char vectors of char_embedding.txt.' % len(embeddings_index_base_char))

########################################
## process texts in datasets
########################################
print('Processing text dataset')

##process corpous base word

texts_1 = []
texts_2 = []
labels = []

with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        texts_1.append(values[0])
        texts_2.append(values[1])
        labels.append(int(values[2]))
print('Base on word,Found %s texts in train.csv' % len(texts_1))

test_texts_1 = []
test_texts_2 = []
with codecs.open(TEST_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        test_texts_1.append(values[0])
        test_texts_2.append(values[1])
print('Base on word,Found %s texts in test.csv' % len(test_texts_1))


##process corpous base char

texts_1_base_char = []
texts_2_base_char = []
labels_base_char = []

with codecs.open(TRAIN_DATA_FILE_base_char, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        texts_1_base_char.append(values[0])
        texts_2_base_char.append(values[1])
        labels_base_char.append(int(values[2]))
print('Base on char,Found %s texts in train.csv' % len(texts_1_base_char))

test_texts_1_base_char = []
test_texts_2_base_char = []
with codecs.open(TEST_DATA_FILE_base_char, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        test_texts_1_base_char.append(values[0])
        test_texts_2_base_char.append(values[1])
print('Base on char,Found %s texts in test.csv' % len(test_texts_1_base_char))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=False, split=" ", oov_token="UNK")
if os.path.exists("word_char_vocab.json"):
    with open("word_char_vocab.json", encoding="utf-8") as f:

        vocab = json.load(f)
        tokenizer.word_index = vocab
else:
    tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2 + texts_1_base_char +
                           texts_2_base_char + test_texts_1_base_char + test_texts_2_base_char)
    vocab = tokenizer.word_index
    with open("word_char_vocab.json", encoding="utf-8", mode="w") as f:

        json.dump(vocab, f)

sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)
test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

sequences_1_base_char = tokenizer.texts_to_sequences(texts_1_base_char)
sequences_2_base_char = tokenizer.texts_to_sequences(texts_2_base_char)
test_sequences_1_base_char = tokenizer.texts_to_sequences(test_texts_1_base_char)
test_sequences_2_base_char = tokenizer.texts_to_sequences(test_texts_2_base_char)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
data_1_base_char = pad_sequences(sequences_1_base_char, maxlen=MAX_SEQUENCE_LENGTH)
data_2_base_char = pad_sequences(sequences_2_base_char, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(labels)
print('Base on word,shape of data tensor: ', data_1.shape)
print('Base on char,shape of data tensor: ', data_1_base_char.shape)
print('Shape of label tensor:', labels.shape)

test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
test_data_1_base_char = pad_sequences(test_sequences_1_base_char, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2_base_char = pad_sequences(test_sequences_2_base_char, maxlen=MAX_SEQUENCE_LENGTH)

#merge trainseting and testset

data_1_merged = np.hstack((data_1, data_1_base_char)) #length * 2
data_2_merged = np.hstack((data_2, data_2_base_char)) #length * 2
print('After merge word and char, shape of data tensor: ', data_1_merged.shape)
test_data_1_merged = np.hstack((test_data_1, test_data_1_base_char))  #length * 2
test_data_2_merged = np.hstack((test_data_2, test_data_2_base_char))  #length * 2



########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index))+1

print('nbwords:', nb_words)

if not os.path.exists("./dataset/dataset_reform/merged_word_char_embedding_matrix.npy"):
    embed_save_path = "./dataset/dataset_reform/"
    if not os.path.exists(embed_save_path):
        os.makedirs(embed_save_path)
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        embedding_vector_base_char = embeddings_index_base_char.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        if embedding_vector_base_char is not None:
            embedding_matrix[i] = embedding_vector_base_char
    np.save(embed_save_path + "merged_word_char_embedding_matrix.npy", embedding_matrix)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
else:
    embedding_matrix = np.load("./dataset/dataset_reform/merged_word_char_embedding_matrix.npy")
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

train_df = pd.read_csv(TRAIN_DATA_FILE, encoding='utf-8')
test_df = pd.read_csv(TEST_DATA_FILE, encoding='utf-8')
train_df_char = pd.read_csv(TRAIN_DATA_FILE_base_char, encoding='utf-8')
test_df_char = pd.read_csv(TEST_DATA_FILE_base_char, encoding='utf-8')

#All features:
#q1_word_count, q2_word_count,word_count_diff,word_overlap,char_unigram_overlap,q1_hash,q2_hash,q1_freq,q2_freq,
# q1_q2_intersect,fuzzy_qratio,fuzzy_wratio,fuzzy_partial_ratio,fuzzy_partial_token_set_ratio,
# fuzzy_partial_token_sort_ratio,fuzzy_token_set_ratio,fuzzy_token_sort_ratio,word_match,tfidf_word_match,
# uni_BLEU,bi_BLEU,BLEU2,cosine_distance,cityblock_distance,jaccard_distance,canberra_distance,euclidean_distance,
# minkowski_distance,braycurtis_distance,skew_q1vec,skew_q2vec,kur_q1vec

magic_feat = train_df[['q1_word_count', 'q2_word_count', 'word_count_diff', 'word_overlap',
                       'q1_hash', 'q2_hash', 'q1_freq', 'q2_freq', 'q1_q2_intersect', 'word_match', 'tfidf_word_match',
                       'fuzzy_qratio', 'fuzzy_wratio', 'fuzzy_partial_ratio', 'fuzzy_partial_token_set_ratio',
                       'fuzzy_partial_token_sort_ratio', 'fuzzy_token_set_ratio', 'fuzzy_token_sort_ratio']]


test_magic_feat = test_df[['q1_word_count', 'q2_word_count', 'word_count_diff', 'word_overlap', 'q1_hash', 'q2_hash',
                           'q1_freq', 'q2_freq', 'q1_q2_intersect', 'word_match', 'tfidf_word_match', 'fuzzy_qratio',
                           'fuzzy_wratio', 'fuzzy_partial_ratio', 'fuzzy_partial_token_set_ratio',
                           'fuzzy_partial_token_sort_ratio', 'fuzzy_token_set_ratio', 'fuzzy_token_sort_ratio']]

magic_feat_char = train_df_char[['q1_char_count', 'q2_char_count', 'char_count_diff', 'char_overlap',
                                 'q1_hash', 'q2_hash', 'q1_freq', 'q2_freq', 'q1_q2_intersect', 'char_match',
                                 'tfidf_char_match', 'fuzzy_qratio', 'fuzzy_wratio', 'fuzzy_partial_ratio',
                                 'fuzzy_partial_token_set_ratio', 'fuzzy_partial_token_sort_ratio',
                                 'fuzzy_token_set_ratio', 'fuzzy_token_sort_ratio']]


test_magic_feat_char = test_df_char[['q1_char_count', 'q2_char_count', 'char_count_diff', 'char_overlap',
                                     'q1_hash', 'q2_hash', 'q1_freq', 'q2_freq', 'q1_q2_intersect', 'char_match',
                                     'tfidf_char_match', 'fuzzy_qratio', 'fuzzy_wratio', 'fuzzy_partial_ratio',
                                     'fuzzy_partial_token_set_ratio', 'fuzzy_partial_token_sort_ratio',
                                     'fuzzy_token_set_ratio', 'fuzzy_token_sort_ratio']]


dis_feat = train_df[['cosine_distance', 'cityblock_distance', 'jaccard_distance', 'canberra_distance',
                     'euclidean_distance', 'minkowski_distance', 'braycurtis_distance', 'skew_q1vec',
                     'skew_q2vec', 'kur_q1vec']]
test_dis_feat = test_df[['cosine_distance', 'cityblock_distance', 'jaccard_distance', 'canberra_distance',
                     'euclidean_distance', 'minkowski_distance', 'braycurtis_distance', 'skew_q1vec',
                     'skew_q2vec', 'kur_q1vec']]

dis_feat_char = train_df_char[['cosine_distance', 'cityblock_distance', 'jaccard_distance', 'canberra_distance',
                     'euclidean_distance', 'minkowski_distance', 'braycurtis_distance', 'skew_q1vec',
                     'skew_q2vec', 'kur_q1vec']]
test_dis_feat_char = test_df_char[['cosine_distance', 'cityblock_distance', 'jaccard_distance', 'canberra_distance',
                     'euclidean_distance', 'minkowski_distance', 'braycurtis_distance', 'skew_q1vec',
                     'skew_q2vec', 'kur_q1vec']]


ss1 = StandardScaler()
ss1.fit(np.vstack((magic_feat, test_magic_feat, magic_feat_char, test_magic_feat_char)))
magic_feat = ss1.transform(magic_feat)
test_magic_feat = ss1.transform(test_magic_feat)
magic_feat_char = ss1.transform(magic_feat_char)
test_magic_feat_char = ss1.transform(test_magic_feat_char)

ss2 = StandardScaler()
ss2.fit(np.vstack((dis_feat, test_dis_feat, dis_feat_char, test_dis_feat_char)))
dis_feat = ss2.transform(dis_feat)
test_dis_feat = ss2.transform(test_dis_feat)
dis_feat_char = ss2.transform(dis_feat_char)
test_dis_feat_char = ss2.transform(test_dis_feat_char)

magic_feat_merged = np.hstack((magic_feat, magic_feat_char))
test_magic_feat_merged = np.hstack((test_magic_feat, test_magic_feat_char))
dis_feat_merged = np.hstack((dis_feat, dis_feat_char))
test_dis_feat_merged = np.hstack((test_dis_feat, test_dis_feat_char))

#save training data, prepare for predict stage.
training_data_save_path = './model_files/' + date + "/decomposable_soft_attention_ensembles_embedding" \
                          + str(number_of_experiment) + "/training_data_cache/"
if not os.path.exists(training_data_save_path):
    os.makedirs(training_data_save_path)
np.save(training_data_save_path + 'data1_merged.npy', data_1_merged)
np.save(training_data_save_path + 'data2_merged.npy', data_2_merged)
np.save(training_data_save_path + 'test_data1_merged.npy', test_data_1_merged)
np.save(training_data_save_path + 'test_data2_merged.npy', test_data_2_merged)
np.save(training_data_save_path + 'magic_feat_merged.npy', magic_feat_merged)
np.save(training_data_save_path + 'test_magic_feat_merged.npy', test_magic_feat_merged)
np.save(training_data_save_path + 'dis_feat_merged.npy', dis_feat_merged)
np.save(training_data_save_path + 'test_dis_feat_merged.npy', test_dis_feat_merged)


#save parameters

features ="magic_feat: " + str(['q1_word_count', 'q2_word_count', 'word_count_diff', 'word_overlap', 'char_unigram_overlap',
           'q1_hash', 'q2_hash', 'q1_freq', 'q2_freq', 'q1_q2_intersect', 'word_match', 'tfidf_word_match',
           'fuzzy_qratio', 'fuzzy_wratio', 'fuzzy_partial_ratio', 'fuzzy_partial_token_set_ratio',
           'fuzzy_partial_token_sort_ratio', 'fuzzy_token_set_ratio', 'fuzzy_token_sort_ratio']) \
            + '\n' + "magic_feat_char: " + str(['q1_char_count', 'q2_char_count', 'char_count_diff', 'char_overlap',
           'q1_hash', 'q2_hash', 'q1_freq', 'q2_freq', 'q1_q2_intersect', 'char_match', 'tfidf_char_match',
           'fuzzy_qratio', 'fuzzy_wratio', 'fuzzy_partial_ratio', 'fuzzy_partial_token_set_ratio',
           'fuzzy_partial_token_sort_ratio', 'fuzzy_token_set_ratio', 'fuzzy_token_sort_ratio']) + '\n' + \
          "dis_feat: " + str(['cosine_distance', 'cityblock_distance', 'jaccard_distance', 'canberra_distance',
        'euclidean_distance', 'minkowski_distance', 'braycurtis_distance', 'skew_q1vec',
        'skew_q2vec', 'kur_q1vec']) + '\n' + \
          'dis_feat_char: ' + str(['cosine_distance', 'cityblock_distance', 'jaccard_distance', 'canberra_distance',
         'euclidean_distance', 'minkowski_distance', 'braycurtis_distance', 'skew_q1vec',
         'skew_q2vec', 'kur_q1vec'])
with open(training_data_save_path + 'parameters.txt', 'w') as f:
    f.write(time.strftime("%Y%m%d%H%M%S", time.localtime()) + '_' + STAMP + '\n')
    f.write(features)
    f.write('\n\n')
    f.close()

########################################
## add class weight
########################################
if re_weight:
    class_weight = {0: weight_0, 1: weight_1}
else:
    class_weight = None


########################################
## train the model
########################################
model = decomposable_attention(pretrained_embedding="./dataset/dataset_reform/merged_word_char_embedding_matrix.npy",
                               projection_dim=projection_dim, projection_hidden=projection_hidden,
                               projection_dropout=projection_dropout, compare_dim=compare_dim,
                               compare_dropout=compare_dropout, dense_dim=dense_dim, dense_dropout=dense_dropout,
                               lr=lr, activation=activation, magic_len=Magic_feat_length*2, dis_len=Dis_feat_length*2,
                               maxlen=MAX_SEQUENCE_LENGTH*2)


early_stopping = EarlyStopping(monitor='val_loss', patience=7)
bst_model_path = "./model_files/" + date + "/decomposable_soft_attention_ensembles_embedding" + str(number_of_experiment)
if not os.path.exists(bst_model_path):
    os.makedirs(bst_model_path)
model_checkpoint = ModelCheckpoint(bst_model_path + '/' + '##epoch{epoch:02d}_valloss{val_loss:.4f}_valacc{val_acc:.4f}.h5',
                                   monitor='val_acc', verbose=1, mode='max',
                                   save_best_only=True, save_weights_only=False)

model.fit([data_1_merged, data_2_merged, magic_feat_merged, dis_feat_merged], labels,
          validation_split=0.1, verbose=1,
          epochs=200, batch_size=batch_size, shuffle=True,
          class_weight=class_weight, callbacks=[model_checkpoint, early_stopping])



