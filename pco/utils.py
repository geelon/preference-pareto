import torch
from torch.func import jacrev


def jacobian(f, x):
    """
    Computes the Jacobian of a possibly vector-valued function f at x.
    
    Args:
      f: callable mapping x -> tensor
      x: Tensor (any shape); will be flattened to a vector of size n

    Returns:
      J: (m, n) tensor, where n = x.numel()
    """
    # Ensure a clean float leaf
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.get_default_dtype())
    if not x.is_floating_point():
        x = x.to(torch.get_default_dtype())
    x = x.detach().clone().requires_grad_(True)

    return jacrev(f, argnums=0, has_aux=False)(x)


def hessian(f, x):
    """
    Compute the Hessian H of a scalar function f at x.

    Args:
      f: callable mapping x -> scalar tensor
      x: Tensor (any shape); will be flattened to a vector of size n

    Returns:
      H: (n, n) tensor, where n = x.numel()
    """
    # Make x a float leaf with grad
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.get_default_dtype())
    if not x.is_floating_point():
        x = x.to(torch.get_default_dtype())
    x = x.detach().clone().requires_grad_(True)

    # Forward + first gradient
    y = f(x)
    if y.ndim != 0:
        raise ValueError("f(x) must return a scalar.")
    (g,) = torch.autograd.grad(y, x, create_graph=True)   # shape = x.shape

    n = x.numel()
    g_flat = g.reshape(-1)

    rows = []
    for i in range(n):
        ei = torch.zeros_like(g_flat)
        ei[i] = 1.0
        # Row i of Hessian: ∂g_i/∂x = J_g[i,:]
        (Hi,) = torch.autograd.grad(
            g, x,
            grad_outputs=ei.view_as(g),
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
        rows.append(Hi.reshape(-1))
    H = torch.stack(rows, dim=0)

    # symmetrize
    H = 0.5 * (H + H.t())

    return H


def project_to_simplex(z, s=1.0, dim=-1):
    """
    Euclidean projection of z onto the simplex { w >= 0, sum(w)=s }.

    Args:
      z   : Tensor (..., n)
      s   : float, simplex sum (>=0); default 1.0 for probability simplex
      dim : axis along which to project (default: last)

    Returns:
      w   : Tensor projected onto the simplex, same shape as z
    """
    if s < 0:
        raise ValueError("Simplex sum s must be >= 0")

    z = torch.as_tensor(z, dtype=torch.get_default_dtype())
    # Move target dim to last for convenience
    if dim != -1:
        z = z.movedim(dim, -1)

    # Sort in descending order
    z_sorted, _ = torch.sort(z, dim=-1, descending=True)
    # Cumulative sum minus s
    cssv = z_sorted.cumsum(dim=-1) - s

    # Find rho = max { j : z_sorted[j] - cssv[j]/(j+1) > 0 }
    # Build 1,2,...,n as float to broadcast
    n = z.shape[-1]
    arange = torch.arange(1, n + 1, device=z.device, dtype=z.dtype)
    t = cssv / arange  # threshold candidates
    cond = z_sorted > t
    # Number of True along last dim gives rho+1
    rho = cond.sum(dim=-1, keepdim=True) - 1  # shape (...,1), int indices

    # theta = t[rho]
    theta = t.gather(-1, rho.clamp(min=0).long()).squeeze(-1)  # shape (...)

    # Projection
    w = (z - theta.unsqueeze(-1)).clamp_min(0)

    # Restore original dim
    if dim != -1:
        w = w.movedim(-1, dim)
    return w