# Neural ODE
* Reproduction of Neural ODE with PyTorch (+Autograd)
* 대부분의 코드는 <a href="https://github.com/msurtsukov/neural-ode/tree/master">Reference(2)</a>를 참고했습니다.
* 또한, Neural ODE 자체가 어려운 알고리즘을 감안하여 본 작업들의 주석은 한국어로 작성하였습니다.
* OfficialNeuralODE는 <code>torchdiffeq</code>라는 Neural ODE 공식 repo를 사용했습니다. (Implicit Adam's method를 사용하기 위해)
* 그리고, 다른 NeuralODE는 <a href="https://github.com/msurtsukov/neural-ode/tree/master">Reference (2)</a>에서 가져온 것으로 간단한 ODE Solver인 Euler's method로 구현된 것이고, Adjoint sensitivity method의 구현, <code>torch.autograd.Function</code>. <code>torch.autograd.grad</code>를 이해하고자 했습니다.

# Environments
```
conda create -n node python=3.12
conda activate node
pip install -r requirements.txt
```

# Experiement (25.12.25)
* 논문의 3번 Section (Replacing Residual Networks with ODEs for supervised learning)을 재현하고자 했다.

![Section3expr](assets/result.png)

* 결과는 위 테이블과 같다. 먼저 논문에서 보여준 성능과 비슷하게 재현되었는지 조사했다. ResNet(6 Blocks)의 성능은 유사하게 도출되었다. 하지만, Neural ODE 기반 모델의 성능은 논문 대비 감소하였다. (RK-Net 포함) 하지만, 그 차이 폭이 실험 별로 대략 0.2X%를 나타낸다. 이에 따라 잘못된 구현은 아님을 시사한다. (다만, 차이점이 하나 있다. ResNet을 구현하면서 <code>Residual Blocks -> Normalization -> ReLU</code>를 거쳐야 하는데, 실수로 ReLU를 넣지 않았다. 실험 중에 발견한 일이라 나머지 Neural ODE 기반 모델 또한 결과의 동등성을 위해 <code>ODE Block -> Normalization -> ReLU</code> 구조에서 ReLU를 뺐다. -> 그래서, 결과 간 공정성은 유지하도록 했다.)

![Section3losscurve](assets/loss_curve.jpg)

* 그리고, ODE-Net 실험을 하면서 모호했던 부분이 ODE Solver 선택이었다. 논문에서는 Implicit Adam's method, 공식 레포지토리에서는 Explicit Runge-Kunta method (<code>dopri5</code>) 사용했다. 그래서, 두 ODE Solver를 기반으로 모두 학습하여 재현했다. 하지만, 이 과정에서 Implicit Adams method 학습에 문제가 있었다. 위 그림과 같이 Implicit Adams method (초록색) 기반 학습이 상당히 불안정했으며, 전역 극소값이 아닌 곳에서 수렴한 것으로 추측된다. 이를 완화하기 위해 Optimizer를 Adam으로 바꿔 다시 학습을 했으며, adaptive learning rate를 통해 안정화된 학습을 유도했다. 그 결과, Runge-Kunta 기반 ODE Solver를 사용한 결과가 0.06% 정도 개선되었다. 이 차이는 소폭의 차이로 큰 차이점이 없음을 시사한다. 하지만, 각 ODE Solver에 따라 안정된 학습 및 불안정한 학습이 유도되기 때문에, 이를 통해 Explicit Runge-Kunta method와 Implicit Adams method의 차이를 분석해볼 필요가 있었다.
    
    ➡️ <b>ODE Solver 간 차이를 공부해볼 것.</b>

* 마지막으로, 논문과 달라진 실험 결과를 기반으로 해석을 해봤다. 내가 수행한 실험의 결과가 맞다고 가정한다. 우선, 거의 차이가 없는 소폭의 차이인 점을 인정하겠으나 이 부분을 주요하게 다루었다. Neural ODE 기반 모델은 ResNet보다 성능이 낮은 것으로 나타났다. 즉, 'Residual Networks를 무한에 가까운 레이어 수로 구성한 듯한 착각을 일으킴으로써, 성능의 개선을 유도한다.' 이건 틀렸다고 볼 수 있다. 이를 더 강하게 주장하기 위해 Neural ODE 기반 네트워크 실험이 Block을 하나만 사용한 것에 기인하여 Residual Block을 하나로만 구성한 ResNet 실험을 수행했다. 그 결과, ResNet (1 Block)과 ODE-Net(RK, explicit)의 성능이 유사하며, 앞서 언급한 주장이 틀렸음을 뒷받침한다. 그래서, 만약에 어떤 연구를 수행하던 도중 성능 개선을 deeper nerual network로 이끌어내기 위해 Neural ODE를 사용하는 전략은 적절하지 않을 수도 있다. (물론, 이런 naive한 접근법도 벌써 10년이 넘었다.) 그래서, Neural ODE는 기존 문제에 대한 접근법의 대안을 제안하는 게 contribution은 아니다. 다만, Neural ODE를 통해서 기존에 풀지 못 했던 문제 (irregular time-series, dynamic systems modeling, continuous-time modeling)를 딥러닝의 영역으로 가져온 것이 메인이라 볼 수 있다.

    ➡️ <b>Neural ODE의 응용을 단순한 task에만 두지는 않을 것, 접근할 수 있는 영역의 확정성을 갖게 된 것.</b>

* 실험과 별개로 Neural ODE는 직접 구현해보는 방법(클론 코딩 수준, <a href="https://github.com/msurtsukov/neural-ode/tree/master">Reference (2)</a> 참고 / Euler's method)과 공식 레포지토리인 <a href="https://github.com/rtqichen/torchdiffeq"><code>torchdiffeq</code></a>를 사용하여 구현한 방법 (Adams, RK), 2가지로 접근해봤다. 위 실험을 통해 느낀 것이나 다른 Neural ODE 응용 논문을 봤을 때, 더 편리한 <code>torchdiffeq</code>를 적극적으로 사용하는 것이 효율적이라는 결론을 내렸다. 하지만, 만약 내가 Neural ODE의 이론 쪽을 연구하는 날이 온다면, 그 때는 <code>torchdiffeq</code>의 코드를 더 깊게 파봐야 한다. 물론, 논문에 대해 이해하는 것은 기반 연구를 하지 않더라도 필수적이다.

    ➡️ <b>Neural ODE 기반 응용 연구 시에는 <code>torchdiffeq</code>를 적극적으로 사용할 것.</b>


# References

1. Reference (1): https://github.com/rtqichen/torchdiffeq

2. Reference (2): https://github.com/msurtsukov/neural-ode/tree/master