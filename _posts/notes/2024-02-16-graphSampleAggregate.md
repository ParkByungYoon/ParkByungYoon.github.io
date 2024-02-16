# 1. Introduction

<aside>
💡 The basic idea behind node embedding approaches is to use dimensionality reduction techniques to distill the high-dimensional information about a node’s graph neighborhood into a dense vector embedding However, previous works have focused on embedding nodes from a single fixed graph, and many real-world applications require embeddings to be quickly generated for unseen nodes, or entirely new (sub)graphs.

</aside>

노드 임베딩을 통해 그래프 이웃 정보가 들어있는 고차원적 데이터에 대한 차원 축소를 이뤄 dense vector embedding을 만들고자 한다. 그런데, 새로운 노드에 대해 대응하지 못한다는 문제가 존재하여 비슷한 형태의 그래프 전반에 걸쳐 노드 임베딩을 진행할 수 있는 Inductive한 approach가 필요하다. (generalization)

Inductive node embedding은 transductive와는 달리 새로 관찰된 subgraph들을 이미 최적화를 마친 알고리즘에 맞게 정렬해야한다. 또,  node의 이웃들 각각이 가지는 지역적인 역할 뿐만 아니라 전역적인 위치에서 구조적인 특성을 고려할 수 있어야한다. 이 논문에서는 기존에 single/fixed graph를 활용하여 transductive한 GCN에 학습 가능한 agg func를 넣음으로써 inductive하게 변형시키는 것이 목표이다.

Matrix Factorization 기반의 embedding approach들과는 달리 새로운 node에 대응 가능한 embedding function을 학습하기 위하여 text attributes, node profile information, node degrees 와 같은 node feature들을 활용한다. 저 친구가 내 근처에 있는가 아닌가 (topological structure) 뿐만 아니라, 저 친구가 가진 feature 정보까지 얻어와서 내 주변 친구들의 feature distribution까지 학습 할 수 있게 된다.

GraphSAGE(SAmple and aggreGatE)는 Node 각각이 가지는 embedding vector를 학습하는 것이 아닌, node의 local neighborhood로부터 feature 정보를 aggreate하는 **aggregator function들의 집합을 학습**하였다. **각 aggregator function은 각자 다른 길이의 hop으로부터 오는 정보를 aggregate** 한다. 해당 aggregator function들은 새로운 노드들을 embedding하는데 사용된다.

# 2. Related work

Factorization-based embedding approaches (random walk, Spectral Clustering, PageRank, multi-dimensional scaling)

각 노드별로 Node Embedding을 직접적으로 진행하기에 새로운 node에 대한 embedding을 구하기 위해선 부가적인 프로세스 필요하다.

Supervised learning over graphs (recent neural network approaches)

Node-Embedding Approach가 아닌 graph-structure 전체에 대한 supervised learning을 진행한다. 즉, 전체 그래프를 분류하려는 시도들이지만 본 논문에서는 각 노드별 representation을 구하는 것이 목표이다.

Graph convolutional networks

Graph 데이터에 CNN을 활용하고자 한 움직임은 자주 있었으나 Large Graph에 적합하지 않거나, 전체 그래프에 대한 분류 수행을 목적으로 두고 등장하였다. 그 중 GCN은 graph Laplacian을 훈련 과정에 이용하는 transductive setting을 사용하고 있어 GraphSAGE는 inductive setting을 사용할 수 있도록 확장하였다.

# 3. Proposed method: GraphSAGE

## 3.1 Embedding generation algorithm

주변 노드가 가지는 feature 정보를 어떻게 Aggregate 할지 학습

1. GraphSAGE 모델이 학습되어있다고 가정할 때 Embedding 생성 과정에 대해 설명
2. 이후에 SGD 기반으로 어떻게 GraphSAGE  parameter 학습되는지 설명

GraphSAGE 모델이 학습되어있다고 가정한다면, 

K개의 aggregator function (주변 노드로부터 정보를 집계)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/5ff7e13f-04e3-4ace-95d7-b1d964b10c87/9e7420f9-4db1-4de8-a963-acf2c97ccf11/Untitled.png)

K개의 Weight Matrix (서로 다른 hop간 정보를 전달)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/5ff7e13f-04e3-4ace-95d7-b1d964b10c87/eb7172c6-9aeb-4477-96d8-26e53e2d3e54/Untitled.png)

Embedding Generation Algorithm

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/5ff7e13f-04e3-4ace-95d7-b1d964b10c87/fed02564-18ef-4798-83c0-ae6edeb14163/Untitled.png)

2: K-hop 만큼 반복

3: 모든 node에 대해 진행

4: 주변 노드들의 embedding을 aggregate

5: 4 과정을 통해 얻은 embedding과 현 노드의 이전 embedding을 concat하여 embedding을 구한다

Neighborhood definition

위 Algorithm 속 4번 과정에서 주변 노드는 각 배치 별로 계산량을 동일하게 가져가기 위해 sampling을 통해 fixed-size로 구해온다. (속도 뿐만 아니라 성능까지 좋아짐을 확인함)

## 3.2 Learning the parameters of GraphSAGE

Unsupervised setting을 위해 output representation에 대해 graph-based loss function 적용

graph-based loss function

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/5ff7e13f-04e3-4ace-95d7-b1d964b10c87/5ee2d52b-915f-4066-a4bf-d2e7df9123c6/Untitled.png)

가까운 노드일수록 비슷한 representation, 먼 노드일수록 구분되는 representation을 가지도록 학습

v는 u로부터 고정된 길이의 random-walk에 의해 발생한 노드

Pn은 negative sampling distribution이고, Q는 negative sample들의 숫자이다.

**여기서 중요한 것은 representation zu 가 embedding look-up을 통해 뽑아온 각 노드 별 고유 embedding vector가 아니라, 주변 노드의 feature으로부터 생성된 representation이라는 것이다.**

## 3.3 Aggregator Architecture

일반적인 N-D차원의 데이터들 (text, image) 과는 다르게 node의 이웃들은 순서에 영향을 받아서는 안된다. 이러한 이유로 aggregator function은 symmetric ( input의 순열에 영향 받지 않음 ) 해야한다. 

### 1. Mean Aggregator

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/5ff7e13f-04e3-4ace-95d7-b1d964b10c87/01899ee3-f1c8-4e1f-977a-359a2984a439/Untitled.png)

Algorithm 1의 4,5번 과정을 위 수식으로 대체한다면, GCN의 inductive한 propagation rule을 유도해낼 수 있다. (modified mean-based aggregator convolution, Localized Spectral Convolution을 linear approximation한 과정)

뒤에 소개할 aggregator들과 다른 이 aggregator만의 특징은 concat 과정이 없다는 점이 있는데, concat 과정은 skip connection과 같은 역할을 하는데, (이전 노드 상태를 그대로 반영한다는 점이 비슷) 큰 성능을 이끌어준다.

### 2. LSTM Aggregator

LSTM을 aggregator로 사용한다면 표현력이 풍부하다는 장점이 있지만, symmetric하지 못하다는 문제점이 존재한다. 이를 해결하기 위해 논문에서는 주변 노드들의 순서를 랜덤하게 부여하여 학습시킨다.

### 3. Pooling Aggregator

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/5ff7e13f-04e3-4ace-95d7-b1d964b10c87/36d05f59-34d3-4951-b3f0-07fde93b1dd2/Untitled.png)

마지막으로 Pooling 기법은 symmetric하고, 학습 가능하다. 위 수식에서는 single layer로 구성되었지만 multi-layer로도 구성할 수 있다. (Wpool: neighbor set에 존재하는 노드의 feature들을 표현하는 역할)

max-pooling operator는 각 노드의 이웃 별로 계산된 representation들에서 효율적으로 주변 노드들의 중요한 특징을 추출해낼 수 있다.