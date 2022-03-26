## Deep Learning based Collaborative Filtering

MF의 inner product는 linearity로 인해 표현력에 한계가 있기 때문에 User term과 Item term의 관계를 모델링할 때 inner product 대신에 DNN을 사용한다

## Autoencoder based CF

Autoencoder는 추천시스템의 sparse한 데이터의 상황에 적합하도록 Formulation이 가능하다 입력값(rating)을 reconstruction 할 수 있게끔 학습함으로써 rating이 가지고 있는 잠재적인 패턴이 latent code에 encoding된다 Rating Prediction의 경우 Rating 값을 직접적으로 reconstruction하며 Top-K ranking의 경우 interaction이 발생할 확률을 reconstruction한다

## DL-based CF for Rating Prediction

U/I-RBM, AutoRec

## DL-based CF for Top-K Ranking

NeuMF, CDAE, Multi-VAE, EASE 