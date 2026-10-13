# ArTaxOr 절지동물 객체 탐지

ArTaxOr 데이터셋을 활용해 절지동물의 위치와 분류군을 탐지하는 개인 프로젝트입니다. 바운딩 박스 전처리와 객체 탐지 모델 학습·추론 실험을 진행했습니다.

- 시작: 2024.05.03
- 담당: 권혁원
- 데이터: ArTaxOr의 절지동물 7개 목(order) 이미지·바운딩 박스
- 기술: Python, PyTorch, torchvision, Faster R-CNN, SSD

## 담당 작업

- 이미지와 JSON 어노테이션을 읽고 바운딩 박스 정보를 분석용 표로 정리
- 이미지 크기 변경과 바운딩 박스 처리 과정 점검
- MobileNet 기반 Faster R-CNN 학습 및 탐지 결과 시각화
- ResNet 기반 Faster R-CNN과 SSD 적용·비교 시도
- 잘못 분류되거나 과도하게 탐지된 결과를 검토하고 후속 실험 방향 정리

## 자료 및 코드

- [발표자료](docs/artaxor-presentation.pdf): 전처리·모델 실험은 PDF 8–23쪽
- [remake_annot.ipynb](remake_annot.ipynb), [resize_img.ipynb](resize_img.ipynb): 어노테이션 정리 및 이미지·박스 확인
- [main_source.ipynb](main_source.ipynb): 초기 데이터 확인
- [Android 이미지 입력 프로토타입](https://github.com/one5469/ArTaxOr_Android_JAVA)

현재 저장소에는 전처리·초기 확인 코드가 주로 남아 있으며, 발표자료에 제시된 학습·추론 코드 전체는 보존되어 있지 않습니다. ResNet 및 SSD 실험에는 학습 실패·저장 결과 유실 등의 제한이 있었습니다.
