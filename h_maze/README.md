# 실험 개요

1. H 모양 맵에서 골 지점이 매번 바뀌고, test시에는 본적 없던 장소에 골이 나오는 세팅으로 generalization 측정 실험 세팅 돌려보기.
2. Baseline은 (1) only sparse,  (2) only dense,  (3) 3 stage free exploration (블록 단위 거리로 계산한 보너스 주기, 즉 멀리가면 갈수록 더 보너스 주는
   거) -to- Sparse (골에 도달했을때만 주어짐)-to-Dense ( 골 지점 으로부터 일정 유닛 블록 몇개까지만 더 Potential-based dense rewards 주기)
3. 2번을 바탕으로 최적화된 reward transition time도 ablation 해보기.

# Baseline 상세

## 1. Only Sparse

Goal에 도달했을 때 (거리 2 이내) 에만 보상을 주고, 에피소드 종료

- Reward: 고정값 1

## 2. Only Dense

Goal 지점으로부터 일정 거리까지 (거리 5 이내) dense reward를 주고, 에피소드 종료

- Reward 계산: Potential Based
    - Taxicab distance로 계산한 거리 이용
    - 스텝 당 리워드: 멀어졌으면 -0.01, 가까워졌으면 0.01

## 3. Free Exploration

시작 지점으로부터 많이 떨어질수록 dense reward를 주는 방식

- Reward 계산: Potential Based
    - Taxicab distance로 계산한 거리에 비례하여 보상을 줌
    - 스텝 당 리워드: 멀어졌으면 +0.01, 가까워졌으면 -0.01

# TODO

- 로깅 래퍼와 리워드 래퍼 분리
