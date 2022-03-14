## Graph Neural Network
1. Graph

    꼭짓점들과 그 노드들을 잇는 변으로 구성된 자료 구조이다 일반적으로 그래프 G = (V,E)로 정의되며

    Graph를 사용하는 이유는 다음과 같다

    관계, 상호작용과 같은 추상적인 개념을 다루기에 적합하다 (유저-아이템 소비 관계)

    Non-Euclidean Space (ex. sns, molecule)의 표현 및 학습이 가능 여기서 Euclidean Space란 2차원 평면이나 3차원 공간을 일반화 시킨 공간, 즉 유한한 실수로 표현 가능한 공간을 이야기한다

2. Graph Neural Network (GNN)
   
    그래프를 표현하는 인접 행렬을 구성하여 그대로 Neural Network의 Input으로 활용하기도 한다

    하지만 인접 행렬을 구성하여 그대로 사용하는 Naive Approach에는 한계가 존재하는데,

    노드의 수가 증가하게 된다면 인접행렬의 차원은 계속 증가하게 되어 모델의 복잡도 또한 증가하게 된다 (+ Input data의 sparse 또한 증가)

    또 그래프에는 노드의 순서에 의미가 없지만 인접행렬을 그대로 Input layer에 사용하게 된다면 노드의 순서가 반영되어 의미가 달라질 수 있다

3. Grpah Convolution Network (GCN)

    Graph Convolution 또한 2D-Covolution와 유사해 물리적으로 가까운 것을 의미하는 edge로 연결되어있는 node들을 모아 사용한다 (local connectivity)

    2D-Convolution 연산과 Graph Convolution 연산 모두 파라미터는 공유한다 (shared weights)

    convolution layer를 3개 이상 쌓게 되면 바로 옆 데이터 포인트 외에 더 멀리있는 정보까지 참고하여 사용할 수 있다 (multi-layer)

    위와 같이 convolution 효과를 만들면 연산량을 줄이고 깊은 네트워크를 통해 간접적인 관계 특징 추출이 가능해진다

## Neural Graph Collaborative Filtering (NGCF)

유저-아이템 상호작용을 GNN으로 임베딩 과정에서 인코딩하는 접근법을 처음 제시한 논문

1. NGCF 등장배경
    
    CF 모델의 두가지 키 포인트

        유저와 아이템의 임베딩
        상호작용 모델링
    
    MF 모델들은 p_u와 q_i를 각각 따로 구해 내적하기 때문에 임베딩과 상호작용은 분리되어있다고 볼 수 있으며
    
    신경망을 적용한 CF 모델들은 유저-아이템 상호작용을 임베딩 단계에서 접근하지 못했다 임베딩이 일어난 이후에 concatenate하기 때문에 임베딩과 상호작용이 분리되어있다

    각각의 임베딩이 상호작용에 sub-optimal하게 학습되기 때문에 모델이 더 정확한 표현력을 가지지 못하고 추천의 성능이 떨어졌다

2. NGCF 기본 아이디어
   
    유저와 아이템 상호작용이 임베딩 단에서부터 반영될 수 있도록 모델 설계했다

    유저, 아이템 개수가 많아질수록 모든 상호작용을 표현하기에 한계가 존재했기 때문에

    하나의 노드를 기준으로 경로가 1보다 큰 High-order Connectivity를 사용하여 유저의 다양한 표현력을 Embedding하고자 하였다

3. NGCF 전체 구조
   
   - 임베딩 레이어
    
        유저-아이템 초기 임베딩 제공하는 레이어 (기존 CF와 동일)

        기존의 MF, Neural CF 모델에서는 임베딩이 곧바로 상호작용에 입력되었지만 NGCF에서는 임베딩을 GNN상에서 propagation시켜 **refine**한다


   - **임베딩 전파 레이어**
  
        high-order connectivity를 학습하는 레이어

        유저-아이템의 collaborative signal을 담을 'message'를 구성하고 결합한다

        Message Construction: 유저-아이템 간 affinity를 고려할 수 있도록 메시지를 구성한다
        
        Message Aggregation: u의 이웃 노드로부터 전파된 message들을 결합하면 1-hop 전파를 통한 임베딩 완료

        1-hop이 아닌 2-hop, 3-hop까지 쌓으면 L차 이웃으로부터 전파된 Message 이용 가능하다

   - 유저-아이템 선호도 예측 레이어

        각각 다르게 임베딩된 값을 concat하여 예측하는 레이어

        L차까지의 임베딩 벡터를 concatnate하여 최종 임베딩 벡터를 계산해 유저-아이템 벡터를 내적해 최종 선호도 예측값을 계산한다
  

## LightGCN

GCN의 가장 핵심적인 부분만 사용하여 더 정확하고 가벼운 추천모델을 제시한 논문

1. LightGCN 아이디어
   
    기존의 NGCF 모델은 Convolution 수행시 학습 파라미터에 Embedding을 곱하고 비선형 함수를 사용했다

    하지만 LightGCN은 단순하게 임베딩을 가중합하여 convolution 연산량을 줄였다

    또 레이어가 깊어질수록 강도가 약해질 것이라는 아이디어를 적용해 모델을 단순화하였다

2. LightGCN Propagation Rule
    
    feature transformation이나 non-linear activation을 제거하고 가중합으로 GCN 적용하였으며

    연결된 노드만을 사용해 self-connection이 존재하지 않는다

    학습 파라미터는 초기 Input을 Embedding vector로 전환하는 첫번째 layer에만 존재한다 (가중 평균을 사용한 전파)

3. Prediction
   
    각 레이어의 임베딩을 결합하는 방법은 가중치(하이퍼파라미터 or 학습 파라미터 둘 다 사용가능)를 사용해 가중합으로 최종 임베딩 벡터를 계산하였다
    


