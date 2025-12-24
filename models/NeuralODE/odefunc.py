import torch
from torch import nn

class ODEFunc(nn.Module):
    """
        Compute f and a df/dz, a df/dp, a df/dt
        위 내용이 클론하고 있는 repo의 original docstring인데, 아마 Appendix B의 Eq.50에서
        Sensitivity Adjoint Method를 위해 계산하는 것으로 보인다.

        그리고, 이 클래스는 무언가를 상속해주려는 템플릿(?) 클래스로 보인다.
        NODE를 가지고 하는 실험에서 forward_with_grad(), flatten_parameters()가 모두 필수인 듯 보인다.
    """
    def forward_with_grad(self, z, t, grad_outputs):

        batch_size = z.shape[0]

        # 여기서 forward는 이 클래스를 상속받게 되는 Neural network or Block의 forward
        out = self.forward(z, t)

        a = grad_outputs
        adfdz, adfdt, *adfdp = torch.autograd.grad(
            (out,), (z, t) + tuple(self.parameters()), grad_outputs=(a),
            allow_unused=True, retain_graph=True
        )
        # grad method automatically sums gradients for batch items, we have to expand them back 
        if adfdp is not None:
            adfdp = torch.cat([p_grad.flatten() for p_grad in adfdp]).unsqueeze(0)
            adfdp = adfdp.expand(batch_size, -1) / batch_size
        if adfdt is not None:
            adfdt = adfdt.expand(batch_size, 1) / batch_size
        return out, adfdz, adfdt, adfdp

    def flatten_parameters(self):
        p_shapes = []
        flat_parameters = []
        for p in self.parameters():
            p_shapes.append(p.size())
            flat_parameters.append(p.flatten())
        return torch.cat(flat_parameters)