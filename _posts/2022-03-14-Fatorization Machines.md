## Factorization Machines

SVM과 Factorization Model의 장점을 결합한 FM을 처음 소개한 논문

1. FM 등장 배경
    
    딥러닝 등장 이전에는 커널 공간을 활용하여 비선형 데이터셋에 높은 성능을 보이는 SVM이 가장 많이 사용되는 모델이었다 하지만 CF 환경에서는 SVM보다 MF 계열의 모델이 더 높은 성능을 내왔고 MF는 특별한 환경 혹은 데이터에만 적용할 수 있었기 때문에 둘의 장점을 결합하고자 하였다

2. FM 공식
   
   두 피쳐의 상호작용을 K 차원의 Factorization Parameter로 표현

3. FM 활용
   
   유저의 영화에 대한 평점 데이터는 대표적인 High Sparsity data이다. 평점 데이터를 일반적인 입력 데이터로 바꾸면(One-Hot encoding) 입력 차원이 전체 유저와 아이템 수만큼 증가하기 때문

   
   