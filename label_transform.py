import pandas as pd
import os
predict_result_df = pd.read_csv("./predicted_output/20181126/decomposable_soft_attention_ensembles_embedding/"
                                "decomposable_soft_attention_ensembles_embedding_mean_12_models_results.csv",
                                encoding="utf-8")
predict_label = predict_result_df["label"]
label_list = []
for label in predict_label:
    if label > 0.5:
        label_list.append(1)
    else:
        label_list.append(0)

df = pd.read_csv("./dataset/test.csv", encoding="utf-8")
df["label"] = label_list

result_save_path = "./predicted_output/results/"
if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)
df.to_csv(result_save_path + "ultimate_results.csv", index=False, encoding="utf-8")
print("Complete predict !")