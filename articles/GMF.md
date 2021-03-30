---
title: 'Pytorch를 이용한 협업 필터링(MF) 구현'
date: 2021-03-28 09:00:00 -0400
categories: recommender-system
tags: [recommender-system, machine-learning, pytorch, tensorboard, collaborative-filtering, mlp, neural-network]
---


이번 글에서는 **Pytorch**와 **MovieLens** 데이터셋을 이용하여, 협업필터링을 구현하겠습니다. 협업 필터링의 여러 기법 중에서 **Matrix Factorization**을 사용하겠습니다. 또한 학습과정은 **Tensorboard**을 이용하여 시각화할 것입니다. 마지막으로 `Neural Collaborative Filtering` 논문에서 제안한 Generalized Matrix Factorization 모델에 대해서 알아보고, 기존 알고리즘과의 성능 비교 실험을 해보겠습니다.

## 1. Matrix Factorization 소개

유저 벡터($p_u$)와 아이템 벡터($p_i$)가 주어졌을 때, 유저와 아이템의 `상호작용(interaction)`을 다음과 같이 내적으로 정의합니다.

> *Matrix Factorization Model* 
>
> $$ y_{ui} = p_u \cdot p_i $$ 
>

이를 바탕으로 모델을 정의하면 다음과 같습니다.

```python
class MF(nn.Module):
    def __init__(
        self, user_num, item_num, factor_num, model,
    ):
        super(MF, self).__init__()
        self.dropout = dropout
        self.model = model
        self.factor_num = factor_num

        # 임베딩 저장공간 확보; (num_embeddings, embedding_dim)
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        predict_size = factor_num
        # 상수 Tensor 생성
        self.predict_layer = torch.ones(predict_size, 1).cuda()
        self._init_weight_()

    def _init_weight_(self):
        # weight 초기화
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

        # bias 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, user, item):
        embed_user = self.embed_user(user)
        embed_item = self.embed_item(item)
        # Tensor의 원소별 곱셈
        output_GMF = embed_user * embed_item
        prediction = torch.matmul(output_GMF, self.predict_layer)
        return prediction.view(-1)
```

## Generalized Matrix Factorization

**GMF**는 이러한 상호작용을 일반화하여 모델의 성능을 향상시키기 위한 모델입니다. 이를 위해 위식 에서 **1. 내적($\cdot$)을 일반화하고**, **2. 활성함수(activation function)를 추가합니다.** 

> *Generalized Matrix Factorization (GMF)*
>
> $$ \hat{y}_{ui} = a_{out}(h^T(p_u \odot p_i)) $$ 
>

이 모델을 구현한 코드는 다음과 같습니다.

```python
class GMF(nn.Module):
    def __init__(
        self, user_num, item_num, factor_num, num_layers, dropout, model,
    ):
        super(GMF, self).__init__()
        self.dropout = dropout
        self.model = model

        # 임베딩 저장공간 확보; num_embeddings, embedding_dim
        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        predict_size = factor_num

        self.predict_layer = nn.Linear(predict_size, 1)
        self._init_weight_()

    def _init_weight_(self):
        # weight 초기화
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity="sigmoid")

        # bias 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, user, item):
        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        output_GMF = embed_user_GMF * embed_item_GMF
        concat = output_GMF

        prediction = self.predict_layer(concat)
        return prediction.view(-1)
```

손실함수는 `binary cross entropy`를, `optimizer`는 `Adam`을 사용하였습니다. 
이 외의 설정은 다음과 같습니다.

```python
args = {
    "batch_size": 256,
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
```

전체코드는 다음 링크에서 확인하실 수 있습니다.

- [Matrix Factorization (github)]()
- [Generalized Matrix Factorization (github)]()

## 실험결과

20 epoch 동안 학습을 한 결과, `MF`의 최고 점수는 `HR = 0.704`, `NDCG = 0.422`이고, `GMF`의 치고 점수는 `HR = 0.706, NDCG = 0.423`이 나왔습니다. `GMF`의 스코어가 약간 더 높지만, 크게 의미있는 차이는 아닌 것 같습니다. 다만 `GMF`는 딥러닝 모델이므로 더 큰 데이터에 대해서 테스트를 하면 의미있는 차이가 나올 수도 있습니다.

> 실험결과 (MF)
```
The time elapse of epoch 000 is: 00: 03: 05
HR: 0.514       NDCG: 0.290
The time elapse of epoch 001 is: 00: 03: 01
HR: 0.600       NDCG: 0.342
The time elapse of epoch 002 is: 00: 02: 59
HR: 0.644       NDCG: 0.372
The time elapse of epoch 003 is: 00: 02: 58
HR: 0.669       NDCG: 0.392
The time elapse of epoch 004 is: 00: 03: 05
HR: 0.680       NDCG: 0.401
The time elapse of epoch 005 is: 00: 03: 01
HR: 0.688       NDCG: 0.408
The time elapse of epoch 006 is: 00: 02: 58
HR: 0.694       NDCG: 0.415
The time elapse of epoch 007 is: 00: 03: 00
HR: 0.699       NDCG: 0.418
The time elapse of epoch 008 is: 00: 03: 01
HR: 0.702       NDCG: 0.418
The time elapse of epoch 009 is: 00: 03: 08
HR: 0.698       NDCG: 0.420
The time elapse of epoch 010 is: 00: 03: 04
HR: 0.704       NDCG: 0.422
The time elapse of epoch 011 is: 00: 03: 04
HR: 0.701       NDCG: 0.422
The time elapse of epoch 012 is: 00: 02: 56
HR: 0.704       NDCG: 0.423
The time elapse of epoch 013 is: 00: 02: 52
HR: 0.702       NDCG: 0.421
The time elapse of epoch 014 is: 00: 02: 53
HR: 0.703       NDCG: 0.423
The time elapse of epoch 015 is: 00: 02: 53
HR: 0.701       NDCG: 0.424
The time elapse of epoch 016 is: 00: 02: 52
HR: 0.699       NDCG: 0.419
The time elapse of epoch 017 is: 00: 02: 46
HR: 0.699       NDCG: 0.419
The time elapse of epoch 018 is: 00: 02: 45
HR: 0.697       NDCG: 0.420
The time elapse of epoch 019 is: 00: 02: 46
HR: 0.698       NDCG: 0.421
End. Best epoch 010: HR = 0.704, NDCG = 0.422
```

> 실험결과 (GMF)

```
The time elapse of epoch 000 is: 00: 03: 11
HR: 0.572       NDCG: 0.320
The time elapse of epoch 001 is: 00: 03: 07
HR: 0.625       NDCG: 0.360
The time elapse of epoch 002 is: 00: 03: 08
HR: 0.651       NDCG: 0.382
The time elapse of epoch 003 is: 00: 03: 06
HR: 0.665       NDCG: 0.393
The time elapse of epoch 004 is: 00: 03: 03
HR: 0.681       NDCG: 0.404
The time elapse of epoch 005 is: 00: 02: 50
HR: 0.695       NDCG: 0.411
The time elapse of epoch 006 is: 00: 02: 55
HR: 0.699       NDCG: 0.413
The time elapse of epoch 007 is: 00: 02: 55
HR: 0.701       NDCG: 0.418
The time elapse of epoch 008 is: 00: 03: 09
HR: 0.705       NDCG: 0.420
The time elapse of epoch 009 is: 00: 03: 07
HR: 0.703       NDCG: 0.419
The time elapse of epoch 010 is: 00: 03: 03
HR: 0.700       NDCG: 0.420
The time elapse of epoch 011 is: 00: 03: 04
HR: 0.701       NDCG: 0.421
The time elapse of epoch 012 is: 00: 03: 10
HR: 0.702       NDCG: 0.421
The time elapse of epoch 013 is: 00: 03: 06
HR: 0.705       NDCG: 0.423
The time elapse of epoch 014 is: 00: 03: 04
HR: 0.701       NDCG: 0.425
The time elapse of epoch 015 is: 00: 03: 03
HR: 0.703       NDCG: 0.425
The time elapse of epoch 016 is: 00: 03: 07
HR: 0.704       NDCG: 0.425
The time elapse of epoch 017 is: 00: 03: 11
HR: 0.706       NDCG: 0.423
The time elapse of epoch 018 is: 00: 03: 09
HR: 0.702       NDCG: 0.424
The time elapse of epoch 019 is: 00: 03: 10
HR: 0.703       NDCG: 0.426
End. Best epoch 017: HR = 0.706, NDCG = 0.423
```

## 참고자료

[1] [실습 코드 링크](https://github.com/doheelab/NCF)

[2] [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)

[3] [A pytorch GPU implementation of He et al.](https://github.com/guoyang9/NCF)

[4] [movielens dataset](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt)

