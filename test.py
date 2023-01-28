import torch

from model import LSTMClassifier

# モデル宣言
model = LSTMClassifier()
# モデルの読み込み
# GPUで読み込む場合
model_path = 'model_GPU.pth'
model.load_state_dict(torch.load(model_path))
# CPUで読み込む場合
# model_path = 'model.pth'
# model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


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
