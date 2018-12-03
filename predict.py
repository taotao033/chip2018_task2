from keras.activations import softmax
import os
from keras.models import load_model
import numpy as np
import pandas as pd

########################################
## make the submission
########################################

print('Start making the submission before fine-tuning')

file_list6 = os.listdir("./model_files/20181125/decomposable_soft_attention_ensembles_embedding6/")
file_list9 = os.listdir("./model_files/20181125/decomposable_soft_attention_ensembles_embedding9/")
file_list10 = os.listdir("./model_files/20181125/decomposable_soft_attention_ensembles_embedding10/")
file_list13 = os.listdir("./model_files/20181125/decomposable_soft_attention_ensembles_embedding13/")
file_list14 = os.listdir("./model_files/20181125/decomposable_soft_attention_ensembles_embedding14/")

print(file_list6)
print(file_list9)
print(file_list10)
print(file_list13)
print(file_list14)

# ['##epoch05_valloss0.3232_valacc0.8640.h5', 'training_data_cache']
# ['##epoch05_valloss0.3296_valacc0.8600.h5', 'training_data_cache']
# ['##epoch07_valloss0.3413_valacc0.8600.h5', 'training_data_cache']
# ['##epoch05_valloss0.3297_valacc0.8605.h5', '##epoch08_valloss0.3268_valacc0.8625.h5', 'training_data_cache']
# ['##epoch12_valloss0.3322_valacc0.8610.h5', 'training_data_cache']

model11 = load_model("./model_files/20181125/decomposable_soft_attention_ensembles_embedding6/" + file_list6[0],
                   custom_objects={'softmax': softmax})
test_data_1 = np.load("./model_files/20181125/decomposable_soft_attention_ensembles_embedding6/"
                      "training_data_cache/test_data1_merged.npy")
test_data_2 = np.load("./model_files/20181125/decomposable_soft_attention_ensembles_embedding6/"
                      "training_data_cache/test_data2_merged.npy")
test_magic_feat = np.load("./model_files/20181125/decomposable_soft_attention_ensembles_embedding6/"
                          "training_data_cache/test_magic_feat_merged.npy")
test_dis_feat = np.load("./model_files/20181125/decomposable_soft_attention_ensembles_embedding6/"
                        "training_data_cache/test_dis_feat_merged.npy")
preds = model11.predict([test_data_1, test_data_2, test_magic_feat, test_dis_feat], batch_size=32, verbose=1)
preds += model11.predict([test_data_2, test_data_1, test_magic_feat, test_dis_feat], batch_size=32, verbose=1)

###
model16 = load_model("./model_files/20181125/decomposable_soft_attention_ensembles_embedding9/" + file_list9[0],
                   custom_objects={'softmax': softmax})
test_data_1 = np.load("./model_files/20181125/decomposable_soft_attention_ensembles_embedding9/"
                      "training_data_cache/test_data1_merged.npy")
test_data_2 = np.load("./model_files/20181125/decomposable_soft_attention_ensembles_embedding9/"
                      "training_data_cache/test_data2_merged.npy")
test_magic_feat = np.load("./model_files/20181125/decomposable_soft_attention_ensembles_embedding9/"
                          "training_data_cache/test_magic_feat_merged.npy")
test_dis_feat = np.load("./model_files/20181125/decomposable_soft_attention_ensembles_embedding9/"
                        "training_data_cache/test_dis_feat_merged.npy")
preds += model16.predict([test_data_1, test_data_2, test_magic_feat, test_dis_feat], batch_size=32, verbose=1)
preds += model16.predict([test_data_2, test_data_1, test_magic_feat, test_dis_feat], batch_size=32, verbose=1)

###
model19 = load_model("./model_files/20181125/decomposable_soft_attention_ensembles_embedding10/" + file_list10[0],
                   custom_objects={'softmax': softmax})
test_data_1 = np.load("./model_files/20181125/decomposable_soft_attention_ensembles_embedding10/"
                      "training_data_cache/test_data1_merged.npy")
test_data_2 = np.load("./model_files/20181125/decomposable_soft_attention_ensembles_embedding10/"
                      "training_data_cache/test_data2_merged.npy")
test_magic_feat = np.load("./model_files/20181125/decomposable_soft_attention_ensembles_embedding10/"
                          "training_data_cache/test_magic_feat_merged.npy")
test_dis_feat = np.load("./model_files/20181125/decomposable_soft_attention_ensembles_embedding10/"
                        "training_data_cache/test_dis_feat_merged.npy")
preds += model19.predict([test_data_1, test_data_2, test_magic_feat, test_dis_feat], batch_size=32, verbose=1)
preds += model19.predict([test_data_2, test_data_1, test_magic_feat, test_dis_feat], batch_size=32, verbose=1)

###
model23 = load_model("./model_files/20181125/decomposable_soft_attention_ensembles_embedding13/" + file_list13[0],
                   custom_objects={'softmax': softmax})
model24 = load_model("./model_files/20181125/decomposable_soft_attention_ensembles_embedding13/" + file_list13[1],
                   custom_objects={'softmax': softmax})
test_data_1 = np.load("./model_files/20181125/decomposable_soft_attention_ensembles_embedding13/"
                      "training_data_cache/test_data1_merged.npy")
test_data_2 = np.load("./model_files/20181125/decomposable_soft_attention_ensembles_embedding13/"
                      "training_data_cache/test_data2_merged.npy")
test_magic_feat = np.load("./model_files/20181125/decomposable_soft_attention_ensembles_embedding13/"
                          "training_data_cache/test_magic_feat_merged.npy")
test_dis_feat = np.load("./model_files/20181125/decomposable_soft_attention_ensembles_embedding13/"
                        "training_data_cache/test_dis_feat_merged.npy")
preds += model23.predict([test_data_1, test_data_2, test_magic_feat, test_dis_feat], batch_size=32, verbose=1)
preds += model23.predict([test_data_2, test_data_1, test_magic_feat, test_dis_feat], batch_size=32, verbose=1)
preds += model24.predict([test_data_1, test_data_2, test_magic_feat, test_dis_feat], batch_size=32, verbose=1)
preds += model24.predict([test_data_2, test_data_1, test_magic_feat, test_dis_feat], batch_size=32, verbose=1)

###
model25 = load_model("./model_files/20181125/decomposable_soft_attention_ensembles_embedding14/" + file_list14[0],
                   custom_objects={'softmax': softmax})
test_data_1 = np.load("./model_files/20181125/decomposable_soft_attention_ensembles_embedding14/"
                      "training_data_cache/test_data1_merged.npy")
test_data_2 = np.load("./model_files/20181125/decomposable_soft_attention_ensembles_embedding14/"
                      "training_data_cache/test_data2_merged.npy")
test_magic_feat = np.load("./model_files/20181125/decomposable_soft_attention_ensembles_embedding14/"
                          "training_data_cache/test_magic_feat_merged.npy")
test_dis_feat = np.load("./model_files/20181125/decomposable_soft_attention_ensembles_embedding14/"
                        "training_data_cache/test_dis_feat_merged.npy")
preds += model25.predict([test_data_1, test_data_2, test_magic_feat, test_dis_feat], batch_size=32, verbose=1)
preds += model25.predict([test_data_2, test_data_1, test_magic_feat, test_dis_feat], batch_size=32, verbose=1)

preds /= 12

submission = pd.DataFrame({'label': preds.ravel()})
csv_save_path = './predicted_output/' + "20181126" + '/decomposable_soft_attention_ensembles_embedding/'
if not os.path.exists(csv_save_path):
    os.makedirs(csv_save_path)

submission.to_csv(csv_save_path + 'decomposable_soft_attention_ensembles_embedding_mean_12_models_results.csv', index=False)
