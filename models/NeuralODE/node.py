'''
# 하나의 block(NeuralNetwork)을 선언한다.
func = ConvODEF(64) 

# derivative로 사용할 NeuralNetwork를 인자로 받는다.
ode = NeuralODE(func) 

# 리턴된 ode는 하나의 block(NerualNetwork)으로 취급하면 된다.
model = ContinuousNeuralMNISTClassifier(ode)
'''
from .odefunc import ODEFunc
from .odeadjoint import ODEAdjoint

import torch
from torch import nn

class NeuralODE(nn.Module):
    def __init__(self, func):
        super().__init__()

        # 인자로 받은 func가 ODEFunc를 상속받아 만들어진 클래스인지 확인
        assert isinstance(func, ODEFunc)

        self.func = func

    def forward(self, z0, t=torch.tensor([0, 1], dtype=torch.float32), return_whole_sequence=False):
        t = t.to(z0)

        '''
        torch.autograd.Funtion의 apply method
        커스텀한 autograd 연산
        '''
        z = ODEAdjoint.apply(z0, t, self.func.flatten_parameters(), self.func)

        if return_whole_sequence: return z
        else: return z[-1]