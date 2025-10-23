import torch
import matplotlib.pyplot as plt


@torch.no_grad()
def evaluate_on_grid(f, xs, ys, device=None, dtype=torch.float32, try_vmap=True):
    xs = xs.to(dtype=dtype, device=device)
    ys = ys.to(dtype=dtype, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing='xy')
    pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    try:
        out = f(pts)
    except Exception:
        try:
            out = f(X, Y)
            if out.ndim == 2:
                return X, Y, out.to(dtype)
            out = out.reshape(X.shape)
            return X, Y, out.to(dtype)
        except Exception:
            if try_vmap:
                try:
                    from torch.func import vmap
                    out = vmap(lambda p: f(p))(pts)
                except Exception:
                    out = torch.stack([f(p) for p in pts], dim=0)
            else:
                out = torch.stack([f(p) for p in pts], dim=0)

    if out.ndim > 1:
        out = out.squeeze(-1)
    Z = out.reshape(X.shape).to(dtype)
    return X, Y, Z

def plot_heatmap(
    f,
    xlim=(-2.0, 2.0), ylim=(-2.0, 2.0),
    nx=400, ny=400,
    mode="imshow",                 # "imshow" or "contourf"
    levels=10,                     # used if mode == "contourf"
    show_contours=True,            # overlay contour lines
    contour_kwargs=None,           # e.g., dict(linewidths=0.8)
    cmap=None,
    alpha=0.3,
    device=None, dtype=torch.float32
):
    xs = torch.linspace(xlim[0], xlim[1], nx)
    ys = torch.linspace(ylim[0], ylim[1], ny)
    X, Y, Z = evaluate_on_grid(f, xs, ys, device=device, dtype=dtype)

    Znp = Z.cpu().numpy()
    Xnp, Ynp = X.cpu().numpy(), Y.cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 5))

    if mode == "contourf":
        hm = ax.contourf(Xnp, Ynp, Znp, levels=levels, cmap=cmap, alpha=alpha)
        ax.set_aspect("equal", adjustable="box")
    else:
        hm = ax.imshow(
            Znp, origin='lower',
            extent=[xs.min().item(), xs.max().item(), ys.min().item(), ys.max().item()],
            aspect='equal', alpha=alpha, cmap=cmap
        )   

    if show_contours:
        ck = dict(levels=levels,linewidths=0.7, colors='gray')
        if contour_kwargs:
            ck.update(contour_kwargs)
        cs = ax.contour(Xnp, Ynp, Znp, **ck)

    ax.set_xlabel('$x_1$', fontsize=16)
    ax.set_ylabel('$x_2$', fontsize=16, rotation=0,labelpad=10)
    plt.tight_layout()
    return fig, ax


# Pareto Visual

class Quadratic:
    def __init__(self, A, z):
        self.A = self.as_tensor(A)
        self.z = self.as_tensor(z) 

    def eval(self, x):
        return 0.5 * torch.einsum('ij,i,j', self.A, x - self.z, x - self.z)
    
    def grad(self, x):
        return torch.einsum('ij,i', self.A, x - self.z)

    @staticmethod
    def as_tensor(tensor):
        if torch.is_tensor(tensor):
            return tensor
        else:
            return torch.tensor(tensor, dtype=torch.float)
        
        
def pareto(q1, q2, res=20):
    A1, A2 = q1.A, q2.A 
    z1 = torch.einsum('ij,j', A1, q1.z)
    z2 = torch.einsum('ij,j', A2, q2.z)

    xs = []
    ys = []
    for w in torch.arange(0, 1 + 1/res, 1/res):
        A_inv = torch.linalg.inv(w * A1 + (1 - w) * A2)
        z = w * z1 + (1 - w) * z2 
        u = torch.einsum('ij,j', A_inv, z)
        xs.append(u[0].item())
        ys.append(u[1].item())

    return xs, ys

q1 = Quadratic([[1,0],[0,1.0]], [0,0.])
q2 = Quadratic([[0.25,0.0],[0.0, 1.]], [1.,0.5])
q3 = Quadratic([[1,0],[0,0.25]], [0.5,1])
q0 = Quadratic([[1,0],[0.,1]], [])

xs1, ys1 = pareto(q1, q2)
xs2, ys2 = pareto(q2, q3)
xs3, ys3 = pareto(q3, q1)