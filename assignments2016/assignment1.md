---
layout: page
mathjax: true
permalink: /assignments2016/assignment1/
---
이번 숙제에서 여러분은 간단한 이미지 분류 파이프라인을 k-Nearest neighbor 또는 SVM/Softmax 분류기에 기반하여 넣는 방법을 연습할 수 있습니다. 이번 숙제의 목표는 다음과 같습니다.

- **이미지 분류 파이프라인**의 기초와 데이터기반 접근법에 대해 이해합니다.
- 학습/확인/테스트의 분할과 **hyperparameter 튜닝**를 위해 검증 데이터를 사용하는 것에 관해 이해합니다.
- 효율적으로 작성된 **벡터화**된 numpy 코드로 proficiency을 나타나게 합니다.
- k-Nearest Neighbor (**kNN**) 분류기를 구현하고 적용해봅니다.
- Multiclass Support Vector Machine (**SVM**) 분류기를 구현하고 적용해봅니다.
- **Softmax** 분류기를 구현하고 적용해봅니다.
- **Two layer neural network** 분류기를 구현하고 적용해봅니다.
- 위 분류기들의 장단점과 차이에 대해 이해합니다.
- 성능향상을 위해 단순히 이미지 픽셀(화소)보다 더 고차원의 표현(**higher-level representations**)을 사용하는 이유에 관하여 이해합니다. (색상 히스토그램, 그라데이션의 히스토그램(HOG) 특징)

## 설치
여러분은 다음 두가지 방법으로 숙제를 시작할 수 있습니다: Terminal.com을 이용한 가상 환경 또는 로컬 환경.

### Termianl에서의 가상 환경.
Terminal에는 우리의 수업을 위한 서브도메인이 만들어져 있습니다. [www.stanfordterminalcloud.com](https://www.stanfordterminalcloud.com) 계정을 등록하세요. 이번 숙제에 대한 스냅샷은 [여기](https://www.stanfordterminalcloud.com/snapshot/49f5a1ea15dc424aec19155b3398784d57c55045435315ce4f8b96b62819ef65)에서 찾아볼 수 있습니다. 만약 수업에 등록되었다면, TA(see Piazza for more information)에게 이 수업을 위한 Terminal 예산을 요구할 수 있습니다. 처음 스냅샷을 실행시키면, 수업을 위한 모든 것이 설치되어 있어서 바로 숙제를 시작할 수 있습니다. [여기](/terminal-tutorial)에 Terminal을 위한 간단한 튜토리얼을 작성해 뒀습니다.

### 로컬 환경
[여기](http://vision.stanford.edu/teaching/cs231n/winter1516_assignment1.zip)에서 압축파일을 다운받고 다음을 따르세요.

**[선택 1] Use Anaconda:**
과학, 수학, 공학, 데이터 분석을 위한 대부분의 주요 패키지들을 담고있는 [Anaconda](https://www.continuum.io/downloads)를 사용하여 설치하는 것이 흔히 사용하는 방법입니다. 설치가 다 되면 모든 요구사항(dependency)을 넘기고 바로 숙제를 시작해도 좋습니다.

**[선택 2] 수동 설치, virtual environment:**
만약 Anaconda 대신 좀 더 일반적이면서 까다로운 방법을 택하고 싶다면 이번 과제를 위한 [virtual environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/)를 만들 수 있습니다. 만약 virtual environment를 사용하지 않는다면 모든 코드가 컴퓨터에 전역적으로 종속되게 설치됩니다. Virtual environment의 설정은 아래를 참조하세요.

~~~bash
cd assignment1
sudo pip install virtualenv      # 아마 먼저 설치되어 있을 겁니다.
virtualenv .env                  # virtual environment를 만듭니다.
source .env/bin/activate         # virtual environment를 활성화 합니다.
pip install -r requirements.txt  # dependencies 설치합니다.
# Work on the assignment for a while ...
deactivate                       # virtual environment를 종료합니다.
~~~

**데이터셋 다운로드:**
먼저 숙제를 시작하기전에 CIFAR-10 dataset를 다운로드해야 합니다. 아래 코드를 `assignment1` 폴더에서 실행하세요:

~~~bash
cd cs231n/datasets
./get_datasets.sh
~~~

**IPython 시작:**
CIFAR-10 data를 받았다면, `assignment1` 폴더의 IPython notebook server를 시작할 수 있습니다. IPython에 친숙하지 않다면 작성해둔 [IPython tutorial](/ipython-tutorial)를 읽어보는 것을 권장합니다.

**NOTE:** OSX에서 virtual environment를 실행하면, matplotlib 에러가 날 수 있습니다([이 문제에 관한 이슈](http://matplotlib.org/faq/virtualenv_faq.html)).  IPython 서버를 `assignment1`폴더의 `start_ipython_osx.sh`로 실행하면 이 문제를 피해갈 수 있습니다; 이 스크립트는 virtual environment가 `.env`라고 되어있다고 가정하고 작성되었습니다.로

### 과제 제출:
로컬 환경이나 Terminal에 상관없이, 이번 숙제를 마쳤다면 `collectSubmission.sh`스크립트를 실행하세요. 이 스크립트는 `assignment1.zip`파일을 만듭니다. 이 파일을 [the coursework](https://coursework.stanford.edu/portal/site/W16-CS-231N-01/)에 업로드하세요.


### Q1: k-Nearest Neighbor 분류기 (20 points)

IPython Notebook **knn.ipynb**이 kNN 분류기를 구현하는 방법을 안내합니다.

### Q2: Support Vector Machine 훈련 (25 points)

IPython Notebook **svm.ipynb**이 SVM 분류기를 구현하는 방법을 안내합니다.

### Q3: Softmax 분류기 실행하기 (20 points)

IPython Notebook **softmax.ipynb**이 Softmax 분류기를 구현하는 방법을 안내합니다.

### Q4: Two-Layer Neural Network (25 points)

IPython Notebook **two_layer_net.ipynb**이 two-layer neural network 분류기를 구현하는 방법을 안내합니다

### Q5: 이미지 특징을 고차원으로 표현하기 (10 points)

IPython Notebook **features.ipynb**을 사용하여 단순한 이미지 픽셀(화소)보다 고차원의 표현이 효과적인지 검사해 볼 것입니다.

### Q6: 추가 과제: 뭔가 더 해보세요! (+10 points)
이번 과제와 관련된 다른 것들을 작성한 코드로 분석하고 연구해보세요. 예를 들어, 질문하고 싶은 흥미로운 질문이 있나요? 통찰력 있는 시각화를 작성할 수 있나요? 아니면 다른 재미있는 살펴볼 거리가 있나요? 또는 손실 함수(loss function)을 조금씩 변형해가며 실험해볼 수도 있을 것입니다. 만약 다른 멋있는 것을 시도해본다면 추가로 10 points를 얻을 수 있고 강의에 수행한 결과가 실릴 수 있습니다.
