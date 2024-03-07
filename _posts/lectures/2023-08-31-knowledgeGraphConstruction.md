---
layout: single
title: 'Knowledge Graph Constuction'
categories:
  - Knowledge Graph
---

# Knowledge Graph Construction

## 1. Introduction

일반적인 텍스트 데이터 -> Information Extraction System -> Knowledge Base/Graph

Knowledge Base와 Knowledge Graph는 같은 것을 다르게 표현한 것이기에 기본적인 notation은 동일하다

관계의 주체/객체를 entity라 하며, 각각을 head tail 이라 부른다.

추가로 fact는 subject, relation, object 이 3가지를 triple 형태로 정형화한 정보를 의미한다.


## 2. Overview

### 2-1. Named Entity Recognition (NER)

주체와 객체를 찾는 과정

주체와 객체가 될 수 있는 요소들의 위치 뿐만 아니라 종류까지도 찾는다

### 2-2. Relation Extraction

주체와 객체 간 존재하는 관계를 추출

NER과 합쳐 Information Extraction이라 하며, 이는 knowledge graph 분야만이 아닌 nlp task에서도 자주 사용된다.

### 2-3. Event Extraction

주체/객체 사이 일어난 event를 추출하고 주체/객체는 해당 event의 참여자로 표현

### 2-4. Entity linking

NER을 통해 얻은 entity에 대한 추가적인 정보를 얻기 위함

Information Extraction의 경우 데이터에 없는 정보는 얻을 수 없다. 그러나 Entity Linking을 통해 주체/객체에 대한 추가적인 정보들을 얻을 수 있다.

NER을 통해 추출한 entity들을 외부 데이터베이스에서 검색하여 관련된 문서를 연결한다 

### 2-5. Coreference Resolution

다른 문장들 사이에서 동일한 entity를 지정하는 부분을 찾는 과정

entity들을 하나로 표현하지 않고 전부 각각의 노드로 표현하게 될 경우 대부분 문장마다 새로운 그래프를 그리게 될 것 이러한 현상은 문장의 단위로 일종의 바운더리를 가지게 되어 knowledge graph를 통한 정보 추론을 어렵게 한다. 이러한 한계를 Coreference Resolution을 통해 해결하고 이 과정을 Graph의 merge라고 보는데, 이는 여러 문장 속 entity들을 모두 하나의 거대한 그래프 안에서 표현이 가능케 하는 것을 의미한다. 정보의 출연 과정이 계속해서 확장되는 Multi-Hop 문제 또한 해결한다.

### 2-6. Knowledge Graph Completion

앞선 과정들을 통해 얻어진 정보들로부터 새로운 정보를 추론

Knowledge Graph이 구축되어 있을 때 시도할 수 있는 과정으로 Knowledge Graph에 표현되어 있지 않은 관계를 그래프 상에서 추론

## 3. Subtask Details

### 3-1. Named Entity Recognition 

#### Named Entity

Entity가 될 수 있는 고유명사 등의 것으로 목적(Class)에 따라 정의

클래스를 나누는 기준에 따라 Entity에 포함되지 않는 등 NER 과정이 바뀔 수 있다.

#### NER의 적용 및 평가

적용은 NER tagging을 통해 수행

태그는 B(begin) I(inside) E(end) S(singleton) O(outside) 가 존재하고, 해당 태그들은 각각 Named Entity를 단어 단위로 표현한다.

평가의 경우 단어들이 올바르게 태깅 되었는지 확인하는 classification의 형태이기에 Precision Recall 그리고 F1 score를 통해 토큰 단위로 수행한다.

### 3-2. Relation Extraction

#### Semi-Supervised Approaches

token embedding 상에서 소수의 labeled data와 유사한 unlabeled data를 찾아 새로 labeling해서 data의 수를 증가시킨다.

#### Discriminative Approaches

- Classification Model: s,r,o 쌍에 대해 가장 높은 likelihood를 구해 결정 (MLE)

- Sequential Tagging Model: 사전 학습된 언어 모델의 인코딩을 활용하여 Input text에 대해 Relation을 포함하는 올바른 tagging을 수행

#### Generative Approaches

seq2seq 모델을 통해 autoregressive하게 s,r,o 쌍을 생성

### 3-3. Coreference Resolution

- mention detection: entity를 찾는 과정 (NER)

- mention clustering: 임베딩을 바탕으로 클러스터링하여 유사도가 높은 단어를 찾는 과정

해당 과정들은 end-to-end로 일어나고 이를 정리하면 언어 모델을 통해 embedding하고 산출된 entity들의 representation을 활용해 clustering을 진행하는 것으로 동일한 entity를 표현하는 mention을 탐색한다고 볼 수 있다.

### 3-4. Knowledge Graph Completion

#### Closed World Assumption (CWA)
KG 상에는 모든 facts가 표현되고 있고 특정 facts가 존재하지 않는다면 해당 관계는 거짓이다

#### Open World Assumption (OWA)
KG 상에서 특정한 facts가 존재하지 않아도 해당 관계의 진위 여부는 알 수 없다

#### Knowledge Graph Completion Subtasks

- Entity prediction: (s,r,?) 혹은 (?,r,o)를 input으로 해서 entity를 예측해 KG에 표현된 fact 이외에 새로운 관계를 얻는다

- Relation prediction: (s,?,o)를 input으로 하여 적절한 relation을 예측

- Link prediction: Node 간 Missing Edge를 예측

#### Embedding based Approaches

(s,r,o) 각각의 Embedding vector를 구해 s+r=o 형태로 KGC를 수행

- TransE: 모두 동일한 차원 임베딩
- TransR: Entity와 relation은 서로 다른 차원이기에 projection 진행

#### Relation Path Reasoning

KG에서는 Composition Relation이 성립한다고 가정

여기서 Composition Relation이란 x,y가 연결되어있고 y,z가 연결되어있다면 x,z 또한 연결되어있다 (multi-hop을 통해 복잡한 관계를 추론) 일반적으로 random-walk를 수행하여 조금씩 확장해 나가며 새로운 path를 찾는다.

#### Rule-based Approaches

logical rule을 활용해서 새로운 relation을 탐색 (rule mining tool)

#### Triple classification based Approaches

GNN을 활용해 특정한 fact의 embedding을 구해서 참,거짓인지 이진 분류를 수행


4. GraphRel 



5. Conclusion