---
layout: page
mathjax: true
permalink: assignments2016/assignment3/
---

이번 과제에서는 회귀신경망(Recurrent Neural Network, RNN)을 구현하고, Microsoft COCO 데이터셋의 이미지 캡셔닝(captionint) 문제에 적용해볼 것입니다. 또한, TinyImageNet 데이터셋을 소개하고, 이 데이터셋에 대해 미리 학습된 모델을 사용하여 이미지 그라디언트에 대한 다양한 어플리케이션에 대해 알아볼 것입니다.

이번 과제의 목표는 다음과 같습니다.

- *회귀신경망(Recurrent Neural Network, RNN)* 구조에 대해 이해하고 시간축 상에서 파라미터 값을 공유하면서 어떻게 시퀀스 데이터에 대해 동작하는지 이해하기
- 기본 RNN 구조와 Long-Short Term Memory (LSTM) RNN 구조의 차이점 이해하기
- 테스트 시 RNN에서 어떻게 샘플을 뽑는지 이해하기
- 이미지 캡셔닝 시스템을 구현하기 위해 컨볼루션 신경망(CNN)과 회귀신경망(RNN)을 결합하는 방법 이해하기
- 학습된 CNN이 입력 이미지에 대한 그라디언트를 계산할 때 어떻게 활용되는지 이해하기
- 이미지 그라디언트의 여러 가지 응용법들 구현하기 (saliency 맵, 모델 속이기, 클래스 시각화, 특징 추출의 역과정, DeepDream 등 포함)

## 설치
다음 두가지 방법으로 숙제를 시작할 수 있습니다: Terminal.com을 이용한 가상 환경 또는 로컬 환경.

### Termianl에서의 가상 환경.
Terminal에는 우리의 수업을 위한 서브도메인이 만들어져 있습니다. [www.stanfordterminalcloud.com](https://www.stanfordterminalcloud.com) 계정을 등록하세요. 이번 숙제에 대한 스냅샷은 [여기](https://www.stanfordterminalcloud.com/snapshot/49f5a1ea15dc424aec19155b3398784d57c55045435315ce4f8b96b62819ef65)에서 찾아볼 수 있습니다. 만약 수업에 등록되었다면, TA(see Piazza for more information)에게 이 수업을 위한 Terminal 예산을 요구할 수 있습니다. 처음 스냅샷을 실행시키면, 수업을 위한 모든 것이 설치되어 있어서 바로 숙제를 시작할 수 있습니다. [여기](/terminal-tutorial)에 Terminal을 위한 간단한 튜토리얼을 작성해 뒀습니다.

### 로컬 환경
[여기](http://cs231n.stanford.edu/winter1516_assignment3.zip)에서 압축파일을 다운받으세요.
Dependency 관련:

**[Option 1] Use Anaconda:**
과학, 수학, 공학, 데이터 분석을 위한 대부분의 주요 패키지들을 담고있는 [Anaconda](https://www.continuum.io/downloads)를 사용하여 설치하는 것이 흔히 사용하는 방법입니다. 설치가 다 되면 모든 요구사항(dependency)을 넘기고 바로 숙제를 시작해도 좋습니다.

**[Option 2] 수동 설치, virtual environment:**
만약 Anaconda 대신 좀 더 일반적이면서 까다로운 방법을 택하고 싶다면 이번 과제를 위한 [virtual environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/)를 만들 수 있습니다. 만약 virtual environment를 사용하지 않는다면 모든 코드가 컴퓨터에 전역적으로 종속되게 설치됩니다. Virtual environment의 설정은 아래를 참조하세요.

~~~bash드
cd assignment3
sudo pip install virtualenv      # This may already be installed
virtualenv .env                  # Create a virtual environment
source .env/bin/activate         # Activate the virtual environment
pip install -r requirements.txt  # Install dependencies
# Work on the assignment for a while ...
deactivate                       # Exit the virtual environment
~~~

**데이터셋 다운로드:**
시작 코드를 받은 후, 전처리 과정이 수행된 MS-COCO 데이터셋, TinyImageNet 데이터셋, 미리 학습된 TinyImageNet 모델을 다운받아야 합니다. `assignment3` 디렉토리에서 다음 명령어를 입력하세요.

~~~bash
cd cs231n/datasets
./get_coco_captioning.sh
./get_tiny_imagenet_a.sh
./get_pretrained_model.sh
~~~

**Compile the Cython extension:** 컨볼루션 신경망은 매우 효율적인 구현이 필요합니다. [Cython](http://cython.org/)을 사용하여 필요한 기능들을 구현해 두어서, 코드를 돌리기 전에 Cython extension을 컴파일해 주어야 합니다. `cs231n` 디렉토리에서 다음 명령어를 입력하세요.

~~~bash
python setup.py build_ext --inplace
~~~

**IPython 시작:**
데이터를 모두 다운받은 뒤, `assignment3`에서 IPython notebook 서버를 시작해야 합니다. IPython에 익숙하지 않다면 [IPython tutorial](/ipython-tutorial)을 먼저 읽어보는 것을 권장합니다.

**NOTE:** OSX에서 virtual environment를 실행하면, matplotlib 에러가 날 수 있습니다([이 문제에 관한 이슈](http://matplotlib.org/faq/virtualenv_faq.html)).  IPython 서버를 `assignment3`폴더의 `start_ipython_osx.sh`로 실행하면 이 문제를 피해갈 수 있습니다; 이 스크립트는 virtual environment가 `.env`라고 되어있다고 가정하고 작성되었습니다.


### 과제 제출:
로컬 환경이나 Terminal에서 숙제를 마쳤다면 `collectSubmission.sh`스크립트를 실행하세요. 이 스크립트는 `assignment3.zip`파일을 만듭니다. 이 파일을 [the coursework](https://coursework.stanford.edu/portal/site/W15-CS-231N-01/) 페이지의 Assignments 탭 아래에 업로드하세요.


### Q1: 기본 RNN 구조로 이미지 캡셔닝 구현 (40 points)
IPython notebook `RNN_Captioning.ipynb`에서 기본 RNN 구조를 사용하여 MS COCO 데이터셋에서 이미지 캡셔닝 시스템을 구현하는 방법을 설명합니다.

### Q2: LSTM 구조로 이미지 캡셔닝 구현 (35 points)
IPython notebook `LSTM_Captioning.ipynb`에서 Long-Short Term Memory (LSTM) RNN 구조의 구현에 대해 설명하고, 이를 MS COCO 데이터셋의 이미지 캡셔닝 문제에 적용해 봅니다.

### Q3: 이미지 그라디언트: Saliency 맵과 Fooling Images (10 points)
IPython notebook `ImageGradients.ipynb`에서 TinyImageNet 데이터셋을 소개합니다. 이 데이터셋에 대해 미리 학습된 모델(pretrained model)을 활용하여 이미지에 대한 그라디언트를 계산하고, 이를 사용해서 saliency 맵과 fooling image들을 생성하는 법에 대해 설명합니다.

### Q4: 이미지 생성: 클래스, 역 과정(Inversion), DeepDream (15 points)
IPython notebook `ImageGeneration.ipynb`에서는 미리 학습된 TinyImageNet 모델을 활용하여 이미지를 생성해볼 것입니다. 특히, 클래스들을 시각화 해보고 특징(feature) 추출의 역과정과 DeepDream을 구현할 것입니다.

### Q5: 추가 과제: 뭔가 더 해보세요! (+10 points)
이번 과제에서 제공된 것들을 활용해서 무언가 멋있는 것들을 시도해볼 수 있을 것입니다. 과제에서 구현하지 않은 다른 방식으로 이미지들을 생성하는 방법이 있을 수도 있어요!
