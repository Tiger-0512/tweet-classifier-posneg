from model import LSTMClassifier
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

from preparation import data_preparation
from preparation.data_preparation import class2tensor, sentence2index
from model import LSTMClassifier
datasets = data_preparation.datasets
word2index = data_preparation.word2index
classes = data_preparation.classes
#sentence2index = data_preparation.sentence2index
#class2tensor = data_preparation.class2tensor


# 元データを7:3に分割（7->学習、3->テスト）
traindata, testdata = train_test_split(datasets, train_size=0.7)

# 単語のベクトル次元数
EMBEDDING_DIM = 10
# 隠れ層の次元数
HIDDEN_DIM = 128
# データ全体の単語数
VOCAB_SIZE = len(word2index)
# 分類先のカテゴリの数
TAG_SIZE = len(classes)

# モデル宣言
model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, TAG_SIZE)
# 損失関数はNLLLoss()を使用 LogSoftmaxを使う時はNLLLoss
loss_function = nn.NLLLoss()
# 最適化手法 lossの減少に時間がかかるため要検討
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 各エポックの合計loss値を格納
losses = []

for epoch in range(100):
    all_loss = 0
    for text, cls in zip(traindata['Text'], traindata['Class']):
        # モデルが持ってる勾配の情報をリセット
        model.zero_grad()
        # 文章を単語IDの系列に変換（modelに食わせられる形に変換）
        inputs = sentence2index(text)
        # 順伝播の結果を受け取る
        out = model(inputs)
        # 正解カテゴリをテンソル化
        answer = class2tensor(cls)
        # 正解とのlossを計算
        loss = loss_function(out, answer)
        # 勾配をセット
        loss.backward()
        # 逆伝播でパラメータ更新
        optimizer.step()
        # lossを集計
        all_loss += loss.item()
    losses.append(all_loss)
    print("epoch", epoch, "\t" , "loss", all_loss)
print("done.")


# lossのグラフ表示
plt.plot(losses)

# テストデータの母数計算
test_num = len(testdata)
# 正解の件数
a = 0
# 勾配自動計算OFF
with torch.no_grad():
    for text, classes in zip(testdata['Text'], testdata['Class']):
        # テストデータの予測
        inputs = sentence2index(text)
        out = model(inputs)

        # outの一番大きい要素を予測結果をする
        _, predict = torch.max(out, 1)

        answer = class2tensor(classes)
        if predict == answer:
            a += 1
print("predict : ", a / test_num)
