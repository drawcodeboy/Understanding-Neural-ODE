from .odefunc import ODEFunc
from .odesolver import ode_solve

import torch
import numpy as np

class ODEAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z0, t, flat_parameters, func):
        '''
        Docstring for forward
        
        :param ctx: ODEAdjoint.apply(...)할 때, 내부적으로 연산 노드에 저장하는 forward의 정보
        :param z0: initial value
        :param t: 기본 값 사용, torch.tensor([0, 1], dtype=torch.float32) = (2,)
        :param flat_parameters: Description
        :param func: Description
        '''

        # ODEFunc의 그 함수가 필요한가보다. 왜지? 
        assert isinstance(func, ODEFunc)
        bs, *z_shape = z0.size() # (B, z1, z2, z3, ...) -> B, (z1, z2, z3, ...)
        time_len = t.size(0)

        # 연산 그래프 없이 forward만 계산
        # Backward는 Adjoint sensitivity method로 계산
        with torch.no_grad(): 
           z = torch.zeros(time_len, bs, *z_shape).to(z0)
           z[0] = z0
           for i_t in range(time_len - 1):
               z0 = ode_solve(z0, t[i_t], t[i_t+1], func)
               z[i_t+1] = z0 

        ctx.func = func # f를 backward에 넘겨서 df/dz, df/dp, df/dt 만들기 위해서 (p = theta)

        # t는 역방향으로 적분해야 하고
        # z는 adjoint 구간별로 업데이트해야 해서 Figure 2에서 adjoint state 흐르는 거 (파란색 점선) 구하려고
        # 아무튼, t, z, p는 df/dz, df/dp, df/dt를 구하기 위해 사용되기 때문에 저장한다
        ctx.save_for_backward(t, z.clone(), flat_parameters) # t는 역행해야 하니까 필요한 거 알겠는데, z랑 p는 왜?

        return z

    @staticmethod
    def backward(ctx, dLdz):
        '''
        Docstring for backward
        
        :param ctx: loss.backward() 시에 상응하는 연산 노드의 ctx를 본 함수에 넣음
        :param dLdz: operation 결과에 대한 L의 미분, grad_y | (time_len, batch_size, *z_shape)
        '''

        func = ctx.func
        t, z, flat_parameters = ctx.saved_tensors
        time_len, bs, *z_shape = z.size()
        n_dim = np.prod(z_shape)
        n_params = flat_parameters.size(0)

        def augmented_dynamics(aug_z_i, t_i):
            z_i, a = aug_z_i[:, :n_dim], aug_z_i[:, n_dim:2*n_dim]

            z_i = z_i.view(bs, *z_shape)
            a = a.view(bs, *z_shape)

            with torch.set_grad_enabled(True): # torch.no_grad()의 반대
                # 이미 gradient가 누적되있지는 않았을테지만, 혹시 모를 위험을 위해
                # 기존의 연산 그래프가 있었든 아니든 .detach()를 통해 떼어내줬다.
                # 그리고, 이 시점부터 df/dz, df/dt를 계산하기 위해 다시 autograd를 켜서 연산 그래프를 그리기 시작
                t_i = t_i.detach().requires_grad_(True)
                z_i = z_i.detach().requires_grad_(True)

                func_eval, adfdz, adfdt, adfdp = func.forward_with_grad(z_i, t_i, grad_outputs=a)
                adfdz = adfdz.to(z_i) if adfdz is not None else torch.zeros(bs, *z_shape).to(z_i)
                adfdp = adfdp.to(z_i) if adfdp is not None else torch.zeros(bs, n_params).to(z_i)
                adfdt = adfdt.to(z_i) if adfdt is not None else torch.zeros(bs, 1).to(z_i)

            # Flatten f and adfdz
            func_eval = func_eval.view(bs, n_dim)
            adfdz = adfdz.view(bs, n_dim) 
            return torch.cat((func_eval, -adfdz, -adfdp, -adfdt), dim=1)

        # dLdz 이거 왜 time length 고려하면서 들어오는가? -> z_i가 직접 Loss에 영향을 미친 case도 고려하는 것
        # 그러므로 dLdz는 adjoint가 아니다. adjoint는 loss에 직접 영향을 준 gradient + 미래 시간에서 흘러들어온 gradient
        dLdz = dLdz.view(time_len, bs, n_dim)
        with torch.no_grad():
            adj_z = torch.zeros(bs, n_dim).to(dLdz)
            adj_p = torch.zeros(bs, n_params).to(dLdz)
            adj_t = torch.zeros(time_len, bs, 1).to(dLdz)

            for i_t in range(time_len-1, 0, -1):
                z_i = z[i_t]
                t_i = t[i_t]
                f_i = func(z_i, t_i).view(bs, n_dim)

                # Compute gradient w.r.t t_1 (page.16 - Algorithm 2 - line 2)
                # direct gradients (z_i가 직접 Loss에 영향을 미친 경우)
                dLdz_i = dLdz[i_t] # (time_len, bs, n_dim) -> (bs, n_dim)

                # dLdz_i: {(bs, n_dim) -> (bs, n_dim, 1) -> (bs, 1, n_dim)} / f_i: {(bs, n_dim) -> (bs, n_dim, 1)}
                # bmm: (bs, 1, 1) -> [:, 0]: (bs, 1)
                dLdt_i = torch.bmm(torch.transpose(dLdz_i.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0] 

                # 직접적인 영향을 주는 gradient 계산
                adj_z += dLdz_i
                adj_t[i_t] = adj_t[i_t] - dLdt_i

                # Define initial augmented state (page.16 - Algorithm 2 - line 3)
                aug_z = torch.cat((z_i.view(bs, n_dim), adj_z, torch.zeros(bs, n_params).to(z), adj_t[i_t]), dim=-1)

                # Solve reverse-time ODE (page.16 - Algorithm 2 - line 6)
                aug_ans = ode_solve(aug_z, t_i, t[i_t-1], augmented_dynamics)

                adj_z[:] = aug_ans[:, n_dim:2*n_dim]
                adj_p[:] += aug_ans[:, 2*n_dim:2*n_dim + n_params]
                adj_t[i_t-1] = aug_ans[:, 2*n_dim + n_params:]

                del aug_z, aug_ans

            ## Adjust 0 time adjoint with direct gradients
            # Compute direct gradients 
            dLdz_0 = dLdz[0]
            dLdt_0 = torch.bmm(torch.transpose(dLdz_0.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

            # Adjust adjoints
            adj_z += dLdz_0
            adj_t[0] = adj_t[0] - dLdt_0
        return adj_z.view(bs, *z_shape), adj_t, adj_p, None