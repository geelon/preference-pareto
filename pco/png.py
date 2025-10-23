import torch
from .utils import jacobian, hessian, project_to_simplex

def minimize_norm_over_simplex(T, *, iters=500, tol=1e-8, w0=None, device=None, dtype=None):
    """
    Solve  min_{w in simplex} || T^T w ||^2   with T ∈ R^{n×d}.
    Returns w (n,), obj value, and a log dict.

    Frank–Wolfe with exact line search (projection-free).
    """
    if not isinstance(T, torch.Tensor):
        T = torch.as_tensor(T, dtype=dtype or torch.get_default_dtype(), device=device)
    if not T.is_floating_point():
        T = T.to(dtype or torch.get_default_dtype())
    device = T.device
    dtype = T.dtype
    n, d = T.shape

    # init on simplex
    if w0 is None:
        w = torch.full((n,), 1.0 / n, dtype=dtype, device=device)
    else:
        w = torch.as_tensor(w0, dtype=dtype, device=device).clone()
        # project tiny negatives to 0 and renormalize (just to be safe)
        w = torch.clamp(w, min=0)
        s = w.sum()
        w = (w / s) if s > 0 else torch.full_like(w, 1.0/n)

    # helpers
    def obj_and_grad(w):
        v = T.t().mv(w)            # v = T^T w  (d,)
        f = v.dot(v)               # ||v||^2
        grad = 2.0 * T.mv(v)       # ∇f(w) = 2 T v  (n,)
        return f, grad, v

    history = []
    f, grad, v = obj_and_grad(w)

    for t in range(iters):
        # Linear minimization oracle on simplex → vertex e_k
        k = torch.argmin(grad)     # index with smallest gradient component
        s = torch.zeros_like(w)
        s[k] = 1.0
        dvec = s - w               # search direction in the simplex
        u = T.t().mv(dvec)         # u = T^T (s - w)

        # FW gap (duality gap) = (w - s)^T ∇f(w) = - dvec^T ∇f(w)
        gap = -(dvec.dot(grad)).item()

        # exact line search in [0,1] on quadratic along dvec
        denom = u.dot(u).item()
        if denom <= 0:             # already optimal along this direction (u≈0)
            if gap <= tol:
                break
            gamma = 0.0
        else:
            gamma = - (v.dot(u).item()) / denom
            gamma = float(max(0.0, min(1.0, gamma)))

        # update
        w = w + gamma * dvec
        # recompute for next iter
        f, grad, v = obj_and_grad(w)

        history.append({"iter": t, "obj": float(f), "gap": gap, "gamma": gamma, "k": int(k)})
        if gap <= tol:
            break

    return w, f.item(), {"iters": len(history), "last_gap": history[-1]["gap"] if history else None, "history": history}

@torch.no_grad()
def _power_iter_spectral_square(T, iters=50):
    """
    Estimate L = ||T||_2^2 = largest eigenvalue of S = T T^T
    via power iteration on T^T T (usually smaller: d x d).
    """
    d = T.shape[1]
    z = torch.randn(d, device=T.device, dtype=T.dtype)
    z = z / (z.norm() + 1e-12)
    for _ in range(iters):
        z = T.t().mv(T.mv(z))       # (T^T T) z
        nz = z.norm()
        if nz == 0: return 0.0
        z = z / nz
    # Rayleigh quotient gives ||T||_2^2
    L = z.dot(T.t().mv(T.mv(z))).item()
    return max(L, 1e-12)

def solve_max_over_orthant(T, v, u, *, iters=2000, tol=1e-8, use_fista=True):
    """
    Solve: max_{w>=0} -0.5 ||v + T^T w||^2 + u^T w
         = min_{w>=0} 0.5 ||v + T^T w||^2 - u^T w

    Returns: w (n,), objective (max form), info dict.
    """
    # Ensure tensors & dtypes
    T = torch.as_tensor(T, dtype=torch.get_default_dtype())
    v = torch.as_tensor(v, dtype=T.dtype, device=T.device)
    u = torch.as_tensor(u, dtype=T.dtype, device=T.device)
    n, d = T.shape

    # Precompute helpers that avoid forming S explicitly:
    # S w = T (T^T w),  grad = S w - b,  b = u - T v
    b = u - T.mv(v)                  # (n,)

    # 0) Try the unconstrained solution Sw = b
    # Solve in the smaller space using normal eq: T T^T w = b.
    # Use CG on the linear operator S(w)=T(T^T w).
    def S_mv(w):                     # linear operator for S
        return T.mv(T.t().mv(w))

    w = torch.zeros(n, dtype=T.dtype, device=T.device)
    r = b - S_mv(w)
    p = r.clone()
    rs_old = r.dot(r)
    for k in range(min(200, n)):     # a few CG steps to get close
        Sp = S_mv(p)
        denom = p.dot(Sp)
        if denom.abs() < 1e-20: break
        alpha = rs_old / denom
        w = w + alpha * p
        r = r - alpha * Sp
        rs_new = r.dot(r)
        if rs_new.sqrt() < 1e-10:
            break
        p = r + (rs_new/rs_old) * p
        rs_old = rs_new

    if (w >= -1e-12).all():          # essentially nonnegative → done
        w = w.clamp_min(0)
        obj_max = -0.5 * (v + T.t().mv(w)).dot(v + T.t().mv(w)).item() + u.dot(w).item()
        return w, obj_max, {"method": "unconstrained-CG", "iters": k+1}

    # 1) FISTA on the constrained problem
    L = _power_iter_spectral_square(T)    # Lipschitz of ∇(0.5||v+T^T w||^2) is ||T||_2^2
    step = 1.0 / L if L > 0 else 1.0      # safe step size

    x = w.clamp_min(0)                    # feasible start
    y = x.clone()
    t = 1.0

    history = []
    for it in range(1, iters + 1):
        # Gradient: ∇(0.5||v + T^T y||^2 - u^T y) = T T^T y - b
        Ty = T.t().mv(y)
        grad = T.mv(Ty) - b               # (n,)

        # FISTA update + projection onto w>=0
        x_next = (y - step * grad).clamp_min(0)

        # Nesterov momentum
        t_next = 0.5 * (1 + (1 + 4 * t * t) ** 0.5)
        y = x_next + ((t - 1) / t_next) * (x_next - x)

        # Prepare next iter & log
        x, t = x_next, t_next

        # Stopping: projected gradient norm
        # (distance to feasibility-adjusted stationary condition)
        pg = (x - (x - grad).clamp_min(0)).norm().item()
        history.append((it, pg))
        if pg < tol:
            break

    w = x
    obj_max = -0.5 * (v + T.t().mv(w)).dot(v + T.t().mv(w)).item() + u.dot(w).item()
    return w, obj_max, {"method": "FISTA", "iters": it, "pg_norm": history[-1][1] if history else None}



class PNG:
    def __init__(self, F, f0, x0, threshold=1e-4, lr=1e-1, regs=None):
        """
        An implementation of the PNG algorithm from Ye and Liu 2022,
        see https://arxiv.org/abs/2110.08713v2

        This algorithm takes parameters:
          threshold: threshold (e in the paper)
          lr: learning rate (xi in the paper)
          regs: regularization schedule (alpha_t in paper)
        """
        self.F = F
        self.f0 = f0
        self.threshold = threshold
        self.lr = lr

        if regs is None:
            regs = torch.ones(1) * 0.01
        self.regs = regs

        self.t = 0
        self.T = len(regs)

        self.x0 = torch.as_tensor(x0, dtype=torch.get_default_dtype())
        self.history = [self.x0.detach().clone()]

    def step(self):
        x = self.history[-1].detach().clone()
        J = jacobian(self.F, x)
        g0 = jacobian(self.f0, x)
        w, g, _ = minimize_norm_over_simplex(J)

        if g <= self.threshold:
            lambdas = torch.zeros_like(w)
        else:
            phi = self.regs[self.t] * g * torch.ones_like(w)
            lambdas = solve_max_over_orthant(J, g0, phi)[0]
        
        v = g0 + lambdas @ J
        x -= self.lr * v
        self.history.append(x.detach().clone())

        if self.t < self.T - 1:
            self.t += 1

    def solve(self, iters=1500):
        from tqdm import tqdm
        for _ in tqdm(range(iters)):
            self.step()