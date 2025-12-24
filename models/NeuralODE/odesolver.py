import math

def ode_solve(z0, t0, t1, f):
    """
    ODE 초기값 문제를 풀기 위한 가장 단순한 방법: Euler's Method
    다른 Explicit (Runge-Kunta), Implicit (Adams)를 해도 된다.
    :param z0: 
    """

    h_max = 0.05
    n_steps = math.ceil((abs(t1 - t0)/h_max).max().item()) # 이걸 왜 이렇게 했는지는 좀 의문이네

    h = (t1 - t0)/n_steps
    t = t0
    z = z0

    for i_step in range(n_steps):
        z = z + h * f(z, t)
        t = t + h
    return z