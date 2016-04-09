---
layout: page
title: AWS Tutorial
permalink: /aws-tutorial/
---

GPU 인스턴스를 사용할경우, 아마존 EC2에 GPU 인스턴스를 사용할 수 있는 아마존 머신 이미지 (AMI)가 있습니다. 이 튜토리얼은 제공된 AMI를 통해 자신의 EC2 인스턴스를 설정하는 방법에 대해서 설명합니다. **현재 CS231N 학생들에게 AWS크레딧을 제공하지 않습니다. AWS 스냅샷을 사용하기 위해 여러분의 예산을 사용하기 권장합니다.**

**요약** AWS가 익숙한 분들: 사용할 이미지는
`cs231n_caffe_torch7_keras_lasagne_v2` 입니다., AMI ID: `ami-125b2c72` region은 US WEST(N. California)입니다. 인스턴스는 `g2.2xlarge`를 사용합니다. 이 이미지에는 Caffe, Torch7, Theano, Keras 그리고 Lasagne가 설치되어 있습니다. 그리고 caffe의 Python binding을 사용할 수 있습니다. 생성한 인스턴스는 CUDA 7.5 와 CuDNN v3를 포함하고 있습니다.

첫째로, AWS계정이 아직 없다면 [AWS홈페이지](http://aws.amazon.com/)에 접속하여 "가입"이라고 적혀있는 노란색 버튼을 눌러 계정을 생성합니다. 버튼을 누르면 가입페이지가 나오며 아래 그림과 같이 나타납니다.

<div class='fig figcenter fighighlight'>
  <img src='{{site.baseurl}}/assets/aws-signup.png'>
</div>

이메일 또는 휴대폰 번호를 입력하고 "새 사용자입니다."를 선택합니다, "보안서버를 사용하여 로그인"을 누르면  세부사항을 입력하는 페이지들이 나오게 됩니다. 이 과정에서 신용카드 정보입력과 핸드폰 인증절차를 진행하게 됩니다. 가입을 위해서 핸드폰과 신용카드를 준비해주세요.

가입을 완료했다면 [AWS 홈페이지](http://aws.amazon.com)로 돌아가 "콘솔에 로그인" 버튼을 클릭합니다. 그리고 이메일과 비밀번호를 입력해 로그인을 진행합니다.

<div class='fig figcenter fighighlight'>
  <img src='{{site.baseurl}}/assets/aws-signin.png'>
</div>

로그인을 완료했다면 다음과 같은 페이지가 여러분을 맞아줍니다.

<div class='fig figcenter fighighlight'>
  <img src='{{site.baseurl}}/assets/aws-homepage.png'>
</div>

오른쪽 상단의 region이 N. California로 설정되어있는지 확인합니다. 만약 제대로 설정되어 있지 않다면 드롭다운 메뉴에서 N. California로 설정합니다.

(그 다음으로 진행하기 위해서는 여러분의 계정이 "인증"되어야 합니다. 인증에 소요되는 시간은 약 2시간이며 인증이 완료되기 전까지는 인스턴스를 실행할 수 없을 수도 있습니다.)

다음으로 EC2링크를 클릭합니다. (Compute 카테고리의 첫번째 링크) 그러면 다음과 같은 대시보드 페이지로 이동합니다.

<div class='fig figcenter fighighlight'>
  <img src='{{site.baseurl}}/assets/ec2-dashboard.png'>
</div>

"Launch Instace"라고 적혀있는 파란색 버튼을 클릭합니다. 그러면 다음과 같은 페이지로 이동하게 됩니다.

<div class='fig figcenter fighighlight'>
  <img src='{{site.baseurl}}/assets/ami-selection.png'>
</div>

왼쪽의 사이드바 메뉴에서 "Community AMIs"를 클릭합니다. 그리고 검색창에 "cs231n"를 입력합니다. 검색결과에 `cs231n_caffe_torch7_keras_lasagne_v2`(AMI ID: `ami-125b2c72`)가 나타납니다. 이 AMI를 선택하고 다음 단게에서 인트턴스 타입을 선택합니다.

<div class='fig figcenter fighighlight'>
  <img src='{{site.baseurl}}/assets/community-AMIs.png'>
</div>

인스턴스 타입`g2.2xlarge` 를 선택하고 "Review and Launch"를 클릭합니다.

<div class='fig figcenter fighighlight'>
  <img src='{{site.baseurl}}/assets/instance-selection.png'>
</div>

다음 화면에서 Launch를 클릭합니다.

<div class='fig figcenter fighighlight'>
  <img src='{{site.baseurl}}/assets/launch-screen.png'>
</div>

클릭하게 되면 기존에 사용하던 key-pair를 사용할 것인지 새로 key-pair를 만들것인지 묻는 창이 뜨게됩니다. 만약 AWS를 이미 사용하고 있다면 사용하던 key를 사용할 수 있습니다. 혹은 드롭다운 메뉴에서 "Create a new key pair"를 선택하여 새로 key를 생성할 수 있습니다. 그리고 key 를 다운로드해야합니다. 다운로드한 key를 실수로 삭제하지 않도록 각별한 주의를 기울여야합니다. 만약 key를 잃어버릴 경우 인스턴스에 **접속할 수 없습니다.**

<div class='fig figcenter fighighlight'>
  <img src='{{site.baseurl}}/assets/key-pair.png'>
</div>

<div class='fig figcenter fighighlight'>
  <img src='{{site.baseurl}}/assets/key-pair-create.png'>
</div>

key 다운로드가 완료되면 key의 권한을 user-only RW로 바꿉니다. Linux/OSX 사용자는 다음 명령어로 권한을 수정할 수 있습니다.

~~~
$ chmod 600 PEM_FILENAME
~~~

여기서 `PEM_FILENAME`은 방금전에 다운로드한 .pem 파일의 이름입니다.

권한수정을 마쳤다면 "Launch Instace"를 클릭합니다. 그럼 생성한 인스턴스가 지금 작동중(Your instance are now launching)이라는 메시지가 나타납니다.


<div class='fig figcenter fighighlight'>
  <img src='{{site.baseurl}}/assets/launching-screen.png'>
</div>

"View Instance"를 클릭하여 인스턴스의 상태를 확인합니다. "2/2 status checks passed"상태가 지나면 "Running"으로 상태가 변하게 됩니다. "Running"상태가 되면 ssh를 통해 생성한 인스턴스에 접속 할 수 있습니다.

<div class='fig figcenter fighighlight'>
  <img src='{{site.baseurl}}/assets/instances-page.png'>
</div>

먼저, 인스턴스 리스트에서 인스턴스의 Public IP를 기억해 둡니다. 그리고 다음을 진행합니다.

~~~
ssh -i PEM_FILENAME ubuntu@PUBLIC_IP
~~~

이제 인스턴스에 로그인이 됩니다. 다음 명령어를 통해 Caffe가 작동중인지 확인할 수 있습니다.

~~~
$ cd caffe
$ ./build/tools/caffe time --gpu 0 --model examples/mnist/lenet.prototxt
~~~

생성한 인스턴스에는 Caff3, Theano, Torch7, Keras 그리고 Lasagne이 설치되어 있습니다. 또한 Caffe Python bindings를 기본적으로 사용할 수 있게 설정되어 있습니다. 그리고 인스턴스에는 CUDA 7.5 와 CuDNN v3가 설치되어 있습니다.

만약 아래와 같은 에러가 발생한다면

~~~
Check failed: error == cudaSuccess (77 vs.  0)  an illegal memory access was encountered
~~~

생성한 인스턴스를 terminate하고 인스턴스 생성부터 다시 시작해야합니다. 오류가 발생하는 정확한 이유는 알 수 없지만 이런현상이 드물게 일어난다고 합니다.

생성한 인스턴스를 사용하는 방법:

- root directory는 총 12GB 입니다. 그리고 ~ 3GB 정도의 여유공간이 있습니다.
- model checkpoins, model들을 저장할 수 있는 60GB의 공간이 `/mnt`에 있습니다.
- 인스턴스를 reboot/terminate 하면 `/mnt` 디렉토리의 자료는 소멸됩니다.
- 추가 비용이 발생하지 않도록 작업이 완료되면 인스턴스를 stop해야합니다. GPU 인스턴스는 사용료가 높습니다. 예산을 현명하게 사용하는것을 권장합니다. 여러분의 작업이 완전히 끝났다면 인스턴스를 Terminate합니다. (디스크 공간 또한 과금이 됩니다. 만약 큰 용량의 디스크를 사용한다면 과금이 많이 될 수 있습니다.)
- 'creating custom alarms'에서 인스턴스가 아무 작업을 하지 않을때 인스턴스를 stop하도록 설정할 수 있습니다.
- 만약 인스턴스의 큰 데이터베이스에 접근할 필요가 없거나 데이터베이스를 다운로드 하기위해서 인스턴스 작동을 원하지 않는다면 가장 좋은 방법은 AMI를 생성하고 인스턴스를 설정할 때 당신의 기기에 AMI를 연결하는 것 일것입니다. (이 작업은 AMI를 선택한 후에 인스턴스를 실행(launching) 하기 전에 설정해야합니다.) 