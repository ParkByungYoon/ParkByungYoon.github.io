---
layout: single
title: 'Go-Back-N / Selective Repeat'
categories:
  - Computer Network
---

Go-Back-N / Selective Repeat / TCP 들이 가진 공통점과 차이점을 찾으며 비교해보고자 한다. 각각 Sender와 Receiver가 상황에 따른 대응 방식에 대해 살펴보면 이해가 쉬울 것 같다.

# Go-Back-N (GBN)

## GBN 특징
- Sender Buffer 보유
- Cumulative ACK 방식
- discard or buffer 구현 측면에서 결정

## GBN Sender
- 가장 오래된 in-flight(yet-to-be-acknowledged) 패킷을 기준으로 Timer
- Timeout 시 해당 packet보다 seq 높은 패킷 모두 재전송
- msg에 corrupt가 일어났을 때는 재전송이 ACK를 통해 요청됨
- ACK msg에 corrupt가 일어났을 때는 Timeout으로 처리
- ACK가 missing 되더라도 높은 seq의 ACK를 받으면 missing된 ACK 모두 처리 (Cumulative ACK)

## GBN Receiver
- 어떤 이벤트 간에 expected(가장 높은 수) seqnum을 가진 ACK를 제대로 된 패킷을 받을때까지 보낸다 (ACK seq# 업데이트 X)

# Selective Repeat (SR)

## SR 특징
- Sender Buffer / Receiver Buffer 보유
- Cumulative ACK X
- Selective repeat 딜레마 문제 보유
  - sequence numbering이 base N counting 형식 (ex. 0 1 2 3 0 1 ...)일 때 생기는 문제
  - Window Size 보다 충분히 큰 counting 필요 (>= 2 * Window Size)

## SR Sender
- 패킷 각각 Timer를 관리한다
- Timeout 시 해당 packet만을 재전송
- ACK를 순서대로 안받아도 상관 없다 (이름에 Selective가 들어간 이유로 추정됨...)
- Receiver가 보내서 받았지만 처리되지 않은 ACK는 record 되었다가 낮은 seq의 packet이 처리되었을 때 한번에 처리된다(Window base를 옮긴다)

## SR Receiver
- 재전송된 packet이 도착할 때까지 buffer에 해당 packet 보다 높은 seq의 packet이 있을 경우 deliver 되지 않고 기다렸다가 순차적으로 upper layer에 전달된다.