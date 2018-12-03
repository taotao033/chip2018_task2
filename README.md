Chip2018_task2 and Biendata competitions links:http://icrc.hitsz.edu.cn/chip2018/Task.html and https://biendata.com/competition/chip2018/
Dataset Download Link:https://drive.google.com/drive/folders/1kDuR24xeIO97DSSxlPQU-F_GvZ6WAaUv?usp=sharing
# Instruction:
## Code instruction:
* data_preprocess.py: Analysis data and generate features we need.
* attention_model.py: Design model structure
* train.py: Train model
* predict.py: Load our pretrain_model and predict the result of testset. 
* label_transform.py : Transform the result that are predicted in predict.py into formal format,
then you can submit.  
* requirements.txt: Related packages environment, before train model you should to execute command 'pip install -r requirements' to install dependency.

* Trainseting: the number of 0: 10000, the number of 1: 10000,  total number:20000
* Testset: total number:10000

## 1.base on word:
* The size of word_embedding.txt: 300
* Total number of question pairs for training: 20000
* Duplicate pairs: 50.0%
* Total number of questions in the training data: 21787
* Number of questions that appear multiple times: 14259

* 8features: q1_word_count, q2_word_count, word_count_diff, word_overlap, uni_BLEU, bi_BLEU,BLEU2, char_unigram_overlap

* feature1: word_match, tfidf_word_match

* feature2: q1_hash, q2_hash, q1_freq, q2_freq

* magic_features: min_freq, common_neighbours, q_len1, q_len2 , diff_len, word_len1, word_len2, common_words, fuzzy_qratio, fuzzy_wratio, fuzzy_partial_ratio, fuzzy_partial_token_set_ratio, fuzzy_partial_token_sort_ratio,fuzzy_token_set_ratio,fuzzy_token_sort_ratio

* questions_distance_features: cosine_distance,cityblock_distance,jaccard_distance,canberra_distance,euclidean_distance,minkowski_distance,braycurtis_distance,skew_q1vec,skew_q2vec,kur_q1vec

* Word count: 
mean-train 7.07, std-train 3.44, min-train 1.00, max-train 43.00, Total word count:7611
mean-test 6.96, std-test 3.41, min-test 1.00, max-test 40.00
(unique)Word_count_total:7611


## 2.base on char:
* The size of char_embedding.txt: 300
* (Base on char) Total number of question pairs for training: 20000
* (Base on char) Duplicate pairs: 50.0%
* (Base on char) Total number of questions in the training data: 21786
* (Base on char) Number of questions that appear multiple times: 14258

* 7features: q1_char_count, q2_char_count, char_count_diff, char_overlap, uni_BLEU, bi_BLEU,BLEU2

* feature1: char_match, tfidf_char_match

* feature2: q1_hash, q2_hash, q1_freq, q2_freq

* magic_features: min_freq, common_neighbours, q_len1, q_len2 , diff_len, char_len1, char_len2, common_chars, fuzzy_qratio, fuzzy_wratio, fuzzy_partial_ratio, fuzzy_partial_token_set_ratio, fuzzy_partial_token_sort_ratio,fuzzy_token_set_ratio,fuzzy_token_sort_ratio

* questions_distance_features: cosine_distance,cityblock_distance,jaccard_distance,canberra_distance,euclidean_distance,minkowski_distance,braycurtis_distance,skew_q1vec,skew_q2vec,kur_q1vec

* Char count:
mean-train 13.73, std-train 5.34, min-train 2.00, max-train 54.00 
mean-test 13.48, std-test 5.40, min-test 2.00, max-test 57.00
(unique)Char_count_total:2085

## Dataset Analysis
![Overall architecture](https://github.com/taotao033/chip2018_task2/blob/master/data_analysis/base_word/Figure_1.png)
* Dataset Analysis Base on word
![Overall architecture](https://github.com/taotao033/chip2018_task2/blob/master/data_analysis/base_word/Figure_2.png)
![Overall architecture](https://github.com/taotao033/chip2018_task2/blob/master/data_analysis/base_word/Figure_3_wordcloud.png)
![Overall architecture](https://github.com/taotao033/chip2018_task2/blob/master/data_analysis/base_word/Figure_4_base_word.png)
![Overall architecture](https://github.com/taotao033/chip2018_task2/blob/master/data_analysis/base_word/Figure_5_base_word.png)
![Overall architecture](https://github.com/taotao033/chip2018_task2/blob/master/data_analysis/base_word/Figure_6_q1_q2_intersect.png)

* Dataset Analysis Base on char
![Overall architecture](https://github.com/taotao033/chip2018_task2/blob/master/data_analysis/base_char/Figure_2.png)
![Overall architecture](https://github.com/taotao033/chip2018_task2/blob/master/data_analysis/base_char/Figure_3.png)
![Overall architecture](https://github.com/taotao033/chip2018_task2/blob/master/data_analysis/base_char/Figure_4_char_cloud.png)
![Overall architecture](https://github.com/taotao033/chip2018_task2/blob/master/data_analysis/base_char/Figure_5_base_char.png)
![Overall architecture](https://github.com/taotao033/chip2018_task2/blob/master/data_analysis/base_char/Figure_6_base_char.png)
![Overall architecture](https://github.com/taotao033/chip2018_task2/blob/master/data_analysis/base_char/Figure_7_q1_q2_intersect.png)
