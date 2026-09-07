Matcher를 의료영상에 맞게 transfer 시킬거임

Matcher는 기본적으로 랜덤하게 하나의 reference 이미지, reference GT, Target을 input으로 가짐

의료영상 특히 BUSI에서는 클래스(benign, malignant)에 따라 특징이 달라 문제 생김

따라서 모든 클래스에서 하나의 reference와 그에 맞는 GT를 통해 Target 이미지가 어떤 클래스에 속하는지를 결정(busi에서는 benign reference,GT, malignant reference,GT, Target 총 5개의 input)

그후에 해당하는 reference를 통해 Target 이미지 segmentation 할거임

추후에 다른 방법론적인 것들을 추가하여 성능 향상 목표


현재의 경우 클래스의 reference를 내가 넣어줌 후에는 클래스별로 이미지를 모두 비교하여 최종 reference 이미지를 정할 것임


26/09/01
extract_anchor.py
를 통해서 내가 정해 놓은 클래스별 이미지 tensor를 save_image에 저장함.

matcher/Matcher.py에서 route_target 함수를 통해 target 이미지가 어디 클래스에 가까운가를 비교하여 반환하도록 함

predict 함수에서 기존의 patch_level_matching 전에 route_target 함수를 호출함으로 흐름을 바꿔줌

유사도가 전부 악성 유사도가 높게 나옴 

why? malignant가 기본적인 노이즈와 비슷한 성질을 띄워서 그런거로 예상됨

how? 코사인 유사도가 아닌 다른 수학적 모델링이 필요할것으로 보임


26/09/02
what? Top-K 라우팅을 통해 전체 이미지가 아닌 유사도 높은 K개의 패치만을 비교

matcher/Matcher.py 에서 route_target 함수를 변경

K를 50으로 50개의 패치를 봤을때 양성: 178, 악성: 469로 악성이 압도적 실제 데이터는 양성2 : 악성1  비율을 가짐 

K를 30으로 줄여서 다시 해봄 결과: 양성: 203, 악성: 443 으로 여전히 악성이 많음


26/09/03
이를 해결하기 위해 대푯값 추출로 변경 20~30개의 이미지를 뽑아 벡터를 평균내어 앵커를 다시 만듬
결과 누적 양성: 220, 누적 악성: 427

26/09/07
pca를 통한 차원 축소로 노이즈 제거를 사용해봄
fit_pca.py 추가
Matcher.py -> route_target 변경