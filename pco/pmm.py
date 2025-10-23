import torch
from .utils import jacobian, hessian, project_to_simplex


class PMM:
    def __init__(self, F, f0, x0, beta0, lr_outer=0.1, lr_inner=0.1, iters_inner=1):
        """
        Pareto Majorization-Minimization using (projected) gradient descent
        
        Args:
          F: [n objectives] vector-valued function x -> (f_1(x),...,f_n(x)) 
          f0: [preference] scalar-valued function x -> f_0(x) 
          x0: initial decision vector
          beta0: initial scalarization
          lr_outer: step size for projected gradient descent for beta
          lr_inner: step size for gradient descent for x
          iters_inner: number of steps of gradient on x per inner solve

        Usage:
           > pmm = PMM(F, f0, x0, beta0)
           > pmm.solve()
           > X = torch.stack(pmm.history)

          X is a tensor of shape (T, d) where T is the number of iterations
          in used in pmm.solve() and d is the dimension of a decision vector
        """
        self.F, self.f0 = F, f0

        # learning rates
        self.lr_inner, self.lr_outer = lr_inner, lr_outer
        self.iters_inner = iters_inner

        # intial iterate
        self.x0 = torch.as_tensor(x0, dtype=torch.get_default_dtype())
        self.beta0 = torch.as_tensor(beta0,dtype=torch.get_default_dtype())

        # store trajectory
        self.history = [self.x0.detach().clone()]
        self.history_beta = [self.beta0.detach().clone()]

    def step(self):
        x = self.history[-1].detach().clone()
        beta = self.history_beta[-1].detach().clone()

        # compute Hf_beta, dF, dx^*, df0
        H_beta = hessian(lambda z: beta @ self.F(z), x)
        dF = jacobian(self.F, x)
        dx = - torch.linalg.solve(H_beta, dF.T)
        df0 = jacobian(self.f0, x)

        # gradient step in beta
        beta -= self.lr_outer * df0 @ dx
        beta = project_to_simplex(beta)
        
        # inner solve: gradient steps in x
        for _ in range(self.iters_inner):
            dfbeta = jacobian(lambda z: beta @ self.F(z), x)
            x -= self.lr_inner * dfbeta

        self.history_beta.append(beta.detach().clone())
        self.history.append(x.detach().clone())

    def solve(self, iters=1500):
        from tqdm import tqdm
        for _ in tqdm(range(iters)):
            self.step()