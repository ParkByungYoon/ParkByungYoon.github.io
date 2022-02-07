---
layout: single
title: boostcamp AI tech Daily Report - 09

---

## 1. 새로 알게된 내용

- [Bias & Variance Trade off 증명과정](https://ko.wikipedia.org/wiki/%ED%8E%B8%ED%96%A5-%EB%B6%84%EC%82%B0_%ED%8A%B8%EB%A0%88%EC%9D%B4%EB%93%9C%EC%98%A4%ED%94%84#%EC%9C%A0%EB%8F%84)

- Optimizer
  - Gradient Descent
    - step size(learning rate)를 잡는게 어렵다
  - Momentum
    - 관성을 이용해 이전 배치에서 일어난 gradient를 반영한다
  - Nesterov Accelerated Gradient
    - 이전 관성을 그대로 반영해 local minima에 도달하기 어려운 Momentum update를 고려해 lookahead gradient를 이용한다
  - Adagrad
    - 각 파라미터가 변한 정도를 저장하고 많이 변한 파라미터는 많게, 적게 변한 파라미터는 적게 update를 적용한다
    - 학습이 길어질수록 적은 변화가 생김(학습이 멈춘다)
  - Adadelta
    - Adagrad 문제를 해결하기 위해 accumulation window를 통해 일정 시간에 대한 gradient만을 반영
    - learning rate가 존재하지 않아 우리가 바꿀 수 있는 요소는 적다.
  - RMSprop
    - Adagrad에서 step size를 반영하였다
  - Adam
    - 이전 gradient 정보의 momentum과 여러 파라미터의 squared gradient를 모두 활용

## 2. 내일 해야할 공부

- CNN 마무리, RNN, Generative Model 강의 듣기
- CNN 심화과제

## 3. 회고

- 정해진 시간 잘 지키기
- 내가 무엇을 얻어 갈 것인가 고민해보기
  - 내가 잘 할 수 있는 것
  - 내가 하고싶은 것
