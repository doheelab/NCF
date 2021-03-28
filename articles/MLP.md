---
title: '[Pytorch] Neural Collaborative Filtering - MLP 실험'
date: 2021-03-28 09:00:00 -0400
categories: recommender-system
tags: [recommender-system, machine-learning, pytorch, tensorboard, collaborative-filtering, mlp, neural-network]
---

이번 글에서는 **Pytorch**를 이용하여, Neural Collaborative Filtering논문의 **MLP**(Multi Layer Perceptron) 파트의 실험을 구현해보겠습니다. 참고한 코드는 
`hexiangnan`의 `PyTorch` 구현 [코드](https://github.com/hexiangnan/neural_collaborative_filtering)입니다. 실습을 위한 코드는 [링크](https://github.com/doheelab/NCF)에서 확인하실 수 있습니다.

## 학습 데이터

 저희가 사용할 테이터는 **MovieLens 1 Million (ml-1m)**입니다. 데이터에 대한 자세한 설명은 [링크](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt)에서 확인하실 수 있습니다. 
 
 이 데이터는 6000명의 유저가 4000개의 영화에 대해서 1~5점 사이로 점수를 매긴 데이터이며, 총 100만여개의 평가 데이터로 이루어져 있습니다. 학습과 테스트에 사용할 데이터셋은 NCF 논문의 저자인 `Xiangnan`의 [저장소](https://github.com/guoyang9/NCF)에서 받으실 수 있습니다.

## 라이브러리 불러오기

`PyTorch`를 사용하여 학습시키고, `텐서보드`를 활용하여 학습과정을 시각화하겠습니다.

```python
import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import pandas as pd
import scipy.sparse as sp
```

## `config`, `args` 설정

```python
config = {
    "model": "GMF",
    "model_path": "./models/",
    "train_rating": "../save/NCF/ml-1m.train.rating",
    "test_negative": "../save/NCF/ml-1m.test.negative",
}

args = {
    "batch_size": 256,
    "dropout": 0.0,
    "epochs": 20,
    "factor_num": 32, 
    "gpu": "0",
    "lr": 0.001,
    "num_layers": 3,
    "num_ng": 4,
    "out": True,
    "test_num_ng": 99,
    "top_k": 10,
}
os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu"]
cudnn.benchmark = True
```

## 데이터셋 설명

저희는 `ml-1m.train.rating`, `ml-1m.test.negative` 2개의 파일을 사용할 것입니다. 파일을 다운받으신 후 `config`의 경로를 파일이 있는 저장된 곳으로 바꿔야 합니다. 각 데이터에 대한 설명은 다음과 같습니다. 

### train.rating:

- 학습 데이터
- 형식: `userID\t itemID\t rating\t timestamp (존재할 경우)`

### test.negative

- 테스트 데이터
- 각 라인은 99개의 `negative samples`를 포함
- 형식: `(userID,itemID)\t negativeItemID1\t negativeItemID2 ...`

### 데이터 불러오기

먼저 `ml-1m.train.rating`, `ml-1m.test.negative`에 데이터가 어떤 형식으로 저장되어있는지 확인을 해보겠습니다. `train_data`는 사이즈가 `(994168, 1)`인 DataFrame이며, `test_negative`는 사이즈가 `604000`인 리스트인 것을 확인할 수 있습니다. 

```python
train_data = pd.read_csv(config["train_rating"])
with open(config["test_negative"], "r") as fd:
    lines = fd.readlines()

print(train_data.shape, len(lines))
print(train_data.head(10))
print(lines[:2])
```

<div style="text-align:center"><img src="https://user-images.githubusercontent.com/57972646/112752295-3e5a1e00-900d-11eb-99b1-105a1aae9f28.png"/></div>

<br/>

> 학습 데이터 불러오기

저희는 논문의 실험 방식에 따라 평점 데이터는 활용하지 않을 것이기 때문에, `\t`를 기준으로 0, 1번째 칼럼만 사용할 것입니다. 다음 코드를 사용하여 원하는 정보만 추출할 수 있습니다.

```python
train_data = pd.read_csv(
    config["train_rating"],
    sep="\t",
    header=None,
    names=["user", "item"],
    usecols=[0, 1],
    dtype={0: np.int32, 1: np.int32},
)
```


<div style="text-align:center"><img src="https://user-images.githubusercontent.com/57972646/112752833-de18ab80-900f-11eb-9225-0a0f1122b867.png"/></div>


> 테스트 데이터 불러오기

테스트 데이터 또한 `\t`를 기준으로 분리할 수 있습니다. `(user_id, movie_id)`형식의 튜플의 리스트로 저장하겠습니다. `user_id`가 같은 요소들 중 첫번째 요소만이 `실제로 평점을 매긴 movie_id)`를 의미하며, 이후의 99개의 요소는 `negative samples`를 의미합니다.

```python
test_data = []
with open(config["test_negative"], "r") as fd:
    line = fd.readline()
    while line != None and line != "":
        arr = line.split("\t")
        u = eval(arr[0])[0]
        test_data.append([u, eval(arr[0])[1]])
        for i in arr[1:]:
            test_data.append([u, int(i)])
        line = fd.readline()
```

<div style="text-align:center"><img src="https://user-images.githubusercontent.com/57972646/112753006-d4dc0e80-9010-11eb-8c67-4dd0e2252e3b.png"/></div>


위에서 설명한 내용을 토대로 `load_all` 함수를 정의하겠습니다.

```python
def load_all():
    """ We load all the three file here to save time in each epoch. """
    train_data = pd.read_csv(
        config["train_rating"],
        sep="\t",
        header=None,
        names=["user", "item"],
        usecols=[0, 1],
        dtype={0: np.int32, 1: np.int32},
    )

    user_num = train_data["user"].max() + 1
    item_num = train_data["item"].max() + 1

    # dok matrix 형식으로 저장하기
    train_data = train_data.values.tolist()

    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    test_data = []
    with open(config["test_negative"], "r") as fd:
        line = fd.readline()
        while line != None and line != "":
            arr = line.split("\t")
            u = eval(arr[0])[0]
            test_data.append([u, eval(arr[0])[1]])
            for i in arr[1:]:
                test_data.append([u, int(i)])
            line = fd.readline()
    return train_data, test_data, user_num, item_num, train_mat

# prepare dataset
train_data, test_data, user_num, item_num, train_mat = load_all()
```

## `Data Loader` 정의하기

다음으로는 `torch.utils.data.DataLoader`를 이용하여 `Data Loader`를 정의하겠습니다. 앞서 살펴보았듯이 `train_data`는 전부 `positive sample`만을 포함하고 있으므로, 학습을 시키기 위해서 `negative samples`를 추가해줍니다`(set_ng_sample)`.

```python
class NCFData(data.Dataset):
    def __init__(self, features, num_item, train_mat=None, num_ng=0, is_training=None):
        super(NCFData, self).__init__()
        """ Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
        # self.features_ps = [[0, 121], [0, 199], [1, 456],...]
        self.features_ps = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training
        self.labels = [0] * len(features)

    def set_ng_sample(self):
        assert self.is_training, "no need to sampling when testing"

        # negative sample 더하기
        self.features_ng = []
        for x in self.features_ps:
            # user
            u = x[0]
            for _ in range(self.num_ng):
                j = np.random.randint(self.num_item)
                # train set에 있는 경우 다시 뽑기
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_ng.append([u, j])

        labels_ps = [1] * len(self.features_ps)
        labels_ng = [0] * len(self.features_ng)

        self.features_fill = self.features_ps + self.features_ng
        self.labels_fill = labels_ps + labels_ng

    def __len__(self):
        return (self.num_ng + 1) * len(self.labels)

    def __getitem__(self, idx):
        features = self.features_fill if self.is_training else self.features_ps
        labels = self.labels_fill if self.is_training else self.labels

        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]
        return user, item, label

def prepare_data(train_data, test_data, item_num, train_mat):

    # construct the train and test datasets
    # args = (features, num_item, train_mat=None, num_ng=0, is_training=None)
    train_dataset = NCFData(train_data, item_num, train_mat, args["num_ng"], True)
    test_dataset = NCFData(test_data, item_num, train_mat, 0, False)
    train_loader = data.DataLoader(
        train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=4
    )
    test_loader = data.DataLoader(
        test_dataset, batch_size=args["test_num_ng"] + 1, shuffle=False, num_workers=0
    )

    return train_loader, test_loader

train_loader, test_loader = prepare_data(train_data, test_data, item_num, train_mat)
```

## 모델 정의하기

저희가 사용할 모델을 정의하는 부분입니다. 주목할 부분은 `input_size = 유저 임베딩 벡터 차원 + 아이템 임베딩 벡터 차원`이라는 것입니다. 즉 유저, 아이템의 임베딩 벡터를 합친 벡터를 입력으로 받아서, MLP layers를 통해 유저와 아이템 간의 `interaction`이 있는지 예측하는 모델로 구성하였습니다. loss 함수로서 `nn.BCEWithLogitsLoss()`를 사용하였고, `optimizer`는 `optim.Adam(model.parameters(), lr=args["lr"])`을 사용하였습니다.

```python
class NCF(nn.Module):
    def __init__(
        self, user_num, item_num, factor_num, num_layers, dropout, model,
    ):
        super(NCF, self).__init__()
        """
		user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors;
		num_layers: the number of layers in MLP model;
		dropout: dropout rate between fully connected layers;
		model: 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF-pre';
		"""
        self.dropout = dropout
        self.model = model

        # 임베딩 저장공간 확보; (num_embeddings, embedding_dim)
        self.embed_user_MLP = nn.Embedding(
            user_num, factor_num * (2 ** (num_layers - 1))
        )
        self.embed_item_MLP = nn.Embedding(
            item_num, factor_num * (2 ** (num_layers - 1))
        )

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)
        predict_size = factor_num
        self.predict_layer = nn.Linear(predict_size, 1)
        self._init_weight_()

    def _init_weight_(self):
        # weight 초기화
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity="sigmoid")

        # bias 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, user, item):
        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        # 임베딩 벡터 합치기
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.MLP_layers(interaction)
        concat = output_MLP

        # 예측하기
        prediction = self.predict_layer(concat)
        return prediction.view(-1)

def create_model(user_num, item_num, args):
    model = NCF(
        user_num,
        item_num,
        args["factor_num"],
        args["num_layers"],
        args["dropout"],
        config["model"],
    )
    model.cuda()
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args["lr"])
    return model, loss_function, optimizer

# 모델 생성하기
model, loss_function, optimizer = create_model(user_num, item_num, args)
```


<br/>

## 평가지표 만들기

실험에서 사용할 평가지표는 `hit rate`와 `nDCG(normalized Discounted Cumulative Gain)`입니다. `hit rate`는 `ground truth`가 예측한 아이템 순위 `k` 안에 들어가는 비율을 나타낸 것이고, `nDCG`는 관련성이 높은 결과를 상위권에 노출시켰는지를 평가하는 지표입니다.

```python
def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def metrics(model, test_loader, top_k):
    HR, NDCG = [], []

    for user, item, _ in test_loader:
        user = user.cuda()
        item = item.cuda()

        predictions = model(user, item)
        # 가장 높은 top_k개 선택
        _, indices = torch.topk(predictions, top_k)
        # 해당 상품 index 선택
        recommends = torch.take(item, indices).cpu().numpy().tolist()
        # 정답값 선택
        gt_item = item[0].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))

    return np.mean(HR), np.mean(NDCG)
```

<br/>

## 학습하기

학습 과정에서의 `loss`, `HR`, `NDCG`를 기록하기 위해, 다음과 같이 `Tensorboard`의 command를 사용합니다.

- `writer = SummaryWriter()`: writer 초기화
- `writer.add_scalar("data/loss", loss.item(), count)`: 매 count마다 loss를 기록
- `writer.add_scalar("test/HR", np.mean(HR), epoch)`: 매 epoch마다 HR의 평균을 기록
- `writer.add_scalar("test/NDCG", np.mean(NDCG), epoch)`: 매 epoch마다 NDCG의 평균을 기록


```python
if __name__ == "__main__":
    count, best_hr = 0, 0
    writer = SummaryWriter()  # for visualization
    for epoch in range(args["epochs"]):
        model.train()  # Enable dropout (if have).
        start_time = time.time()
        train_loader.dataset.set_ng_sample()

        for user, item, label in train_loader:
            user = user.cuda()
            item = item.cuda()
            label = label.float().cuda()

            # gradient 초기화
            model.zero_grad()
            prediction = model(user, item)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()
            writer.add_scalar("data/loss", loss.item(), count)
            count += 1

        model.eval()
        HR, NDCG = metrics(model, test_loader, args["top_k"])

        elapsed_time = time.time() - start_time
        print(
            "The time elapse of epoch {:03d}".format(epoch)
            + " is: "
            + time.strftime("%H: %M: %S", time.gmtime(elapsed_time))
        )
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

        if HR > best_hr:
            best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
            if args["out"]:
                if not os.path.exists(config["model_path"]):
                    os.mkdir(config["model_path"])
                torch.save(
                    model, "{}{}.pth".format(config["model_path"], config["model"])
                )

    print(
        "End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
            best_epoch, best_hr, best_ndcg
        )
    )
```

## 실험결과

20 epoch 동안 학습을 한 결과, 최고 점수는 `HR = 0.690`, `NDCG = 0.414`로 관찰되었습니다. 이는 논문에 나온 결과인 `HR = 0.692`, `NDCG = 0.425`에 약간 못 미치지만, `initialization` 결과에 따라 값은 달라질 수 있으므로 대체로 구현이 잘 되었다고 생각됩니다.

```
The time elapse of epoch 000 is: 00: 03: 12
HR: 0.553       NDCG: 0.308
The time elapse of epoch 001 is: 00: 03: 15
HR: 0.610       NDCG: 0.351
The time elapse of epoch 002 is: 00: 03: 16
HR: 0.644       NDCG: 0.377
The time elapse of epoch 003 is: 00: 03: 13
HR: 0.658       NDCG: 0.386
The time elapse of epoch 004 is: 00: 03: 14
HR: 0.671       NDCG: 0.398
The time elapse of epoch 005 is: 00: 03: 12
HR: 0.672       NDCG: 0.403
The time elapse of epoch 006 is: 00: 03: 07
HR: 0.680       NDCG: 0.408
The time elapse of epoch 007 is: 00: 03: 09
HR: 0.680       NDCG: 0.407
The time elapse of epoch 008 is: 00: 03: 08
HR: 0.687       NDCG: 0.411
The time elapse of epoch 009 is: 00: 03: 07
HR: 0.687       NDCG: 0.413
The time elapse of epoch 010 is: 00: 03: 07
HR: 0.687       NDCG: 0.411
The time elapse of epoch 011 is: 00: 03: 08
HR: 0.688       NDCG: 0.415
The time elapse of epoch 012 is: 00: 03: 07
HR: 0.681       NDCG: 0.415
The time elapse of epoch 013 is: 00: 03: 10
HR: 0.689       NDCG: 0.414
The time elapse of epoch 014 is: 00: 03: 16
HR: 0.682       NDCG: 0.409
The time elapse of epoch 015 is: 00: 03: 16
HR: 0.682       NDCG: 0.413
The time elapse of epoch 016 is: 00: 03: 14
HR: 0.682       NDCG: 0.412
The time elapse of epoch 017 is: 00: 03: 14
HR: 0.684       NDCG: 0.415
The time elapse of epoch 018 is: 00: 03: 14
HR: 0.680       NDCG: 0.410
The time elapse of epoch 019 is: 00: 03: 09
HR: 0.690       NDCG: 0.414
End. Best epoch 019: HR = 0.690, NDCG = 0.414
```

![image](https://user-images.githubusercontent.com/57972646/112757172-94d25700-9023-11eb-9cb0-133c284b487f.png)

## 참고자료

[1] [실습 코드 링크](https://github.com/doheelab/NCF)

[2] [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)

[3] [A pytorch GPU implementation of He et al.](https://github.com/guoyang9/NCF)

[4] [movielens 데이터 셋](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt)

