import pandas as pd
import torch

from preparation.preprocessing import pre_processing


# csvファイル読み込み
df = pd.read_csv('./tweet_extructor/final_data.csv')

classes = ['Pos_Neg', 'Pos', 'Neg', 'Neu', 'Unr']

# Pos&Neg=1であるデータの数を取得
df_posneg = df[df['Pos_Neg'] == 1]
num = len(df_posneg)
df.loc[df['Pos_Neg'] == 1, 'Class'] = 'Pos_Neg'

# Pos&Neg=1であるデータの数だけ、その他のクラスののデータも抽出、クラス付け
for cls in classes[1:]:
    df_tmp = df[df[cls] == 1]
    df_tmp = df_tmp.sample(n=num)
    df_tmp['Class'] = cls
    df = df[df[cls] == 0]
    df = pd.concat([df, df_tmp])
    # 確認
    # print(len(df[df[cls] == 1]))

# 必要な列のみを抽出
datasets = df[['Class', 'Text']]
datasets.reset_index(drop=True, inplace=True)
print(len(datasets))
# print(datasets.head())


# 単語ID辞書を作成
word2index = {}
for sentence in datasets["Text"]:
    separated_sentence = pre_processing(sentence)
    for word in separated_sentence:
        if word in word2index: continue
        word2index[word] = len(word2index)
print("vocab size : ", len(word2index))


# 文章を単語IDの系列データに変換
# PyTorchのLSTMのインプットになるデータなのでtensor型へ変換
def sentence2index(sentence):
    separated_sentence = pre_processing(sentence)
    return torch.tensor([word2index[w] for w in separated_sentence], dtype=torch.long)
# print(sentence2index('Xperia Z3は今月23日発売？'))


class2index = {}
for cls in classes:
    if cls in class2index: continue
    class2index[cls] = len(class2index)
print(class2index)

def class2tensor(cls):
    return torch.tensor([class2index[cls]], dtype=torch.long)
# print(class2tensor('Pos'))
