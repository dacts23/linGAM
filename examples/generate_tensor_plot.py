"""Generate 2D Tensor Product Interaction visual for documentation."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from magli import GAMCore
from magli.eilers._bspline import b_spline_basis


def create_tensor_product_figure():
    fig = plt.figure(figsize=(15, 4.8), dpi=150)
    
    # ── 1. Left Panel: 1D Marginal Bases ──────────────────────────────────────
    ax1 = fig.add_subplot(1, 3, 1)
    grid_1d = np.linspace(0, 1, 200)
    bases_1d = b_spline_basis(grid_1d, n_splines=6, spline_order=3, edge_knots=np.array([0.0, 1.0]))
    
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, 6))
    for j in range(6):
        ax1.plot(grid_1d, bases_1d[:, j], color=colors[j], lw=2, label=f"$B_{j+1}(x_1)$" if j < 3 else None)
    
    # Highlight one specific 1D bump (j=2)
    ax1.plot(grid_1d, bases_1d[:, 2], color="crimson", lw=2.8, ls="--", label="$B_3(x_1)$ (selected)")
    ax1.set_title("1. Marginal 1-D Basis Functions\n$B_i(x_1)$ and $B_j(x_2)$", fontsize=11, fontweight="bold")
    ax1.set_xlabel("$x_1$ or $x_2$", fontsize=10)
    ax1.set_ylabel("Basis Value $B_j(x)$", fontsize=10)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)
    
    # ── 2. Middle Panel: 2D Tensor Product Basis Bump ─────────────────────────
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    x1_g = np.linspace(0, 1, 60)
    x2_g = np.linspace(0, 1, 60)
    X1, X2 = np.meshgrid(x1_g, x2_g)
    
    B1 = b_spline_basis(x1_g, n_splines=6, spline_order=3, edge_knots=np.array([0.0, 1.0]))
    B2 = b_spline_basis(x2_g, n_splines=6, spline_order=3, edge_knots=np.array([0.0, 1.0]))
    
    # 2D basis bump: B_3(x1) * B_3(x2)
    Z_bump = np.outer(B2[:, 2], B1[:, 2])
    
    surf = ax2.plot_surface(X1, X2, Z_bump, cmap="coolwarm", edgecolor="k", linewidth=0.2, alpha=0.85)
    ax2.set_title("2. 2-D Tensor Basis Function\n$B_{3,3}(x_1, x_2) = B_3(x_1) \\times B_3(x_2)$", fontsize=11, fontweight="bold")
    ax2.set_xlabel("$x_1$", fontsize=9)
    ax2.set_ylabel("$x_2$", fontsize=9)
    ax2.set_zlabel("Height", fontsize=9)
    ax2.view_init(elev=28, azim=-55)
    
    # ── 3. Right Panel: Fitted 2D Surface te(0, 1) ───────────────────────────
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    np.random.seed(42)
    n_pts = 200
    x1_data = np.random.uniform(0, 1, n_pts)
    x2_data = np.random.uniform(0, 1, n_pts)
    # Interactive function: saddle surface
    y_true_surf = np.sin(np.pi * x1_data) * np.cos(np.pi * x2_data)
    y_data = y_true_surf + np.random.normal(0, 0.08, n_pts)
    
    X_train = np.column_stack([x1_data, x2_data])
    model = GAMCore("te(0, 1, k=5, bs='cr')", backend="wood")
    model.fit(X_train, y_data)
    
    # Evaluate model on 2D grid
    grid_pts = np.column_stack([X1.ravel(), X2.ravel()])
    y_pred_grid = model.predict(grid_pts).reshape(X1.shape)
    
    surf_fit = ax3.plot_surface(X1, X2, y_pred_grid, cmap="viridis", edgecolor="none", alpha=0.82)
    ax3.scatter(x1_data, x2_data, y_data, color="crimson", s=12, alpha=0.6, label="Data $(y_i)$")
    ax3.set_title("3. Fitted Interaction Surface\n$\\hat{f}(x_1, x_2)$ via `te(0, 1)` (REML)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("$x_1$", fontsize=9)
    ax3.set_ylabel("$x_2$", fontsize=9)
    ax3.set_zlabel(r"$\hat{y}$", fontsize=9)
    ax3.view_init(elev=28, azim=-55)
    ax3.legend(loc="upper left", fontsize=8)
    
    fig.suptitle("Tensor-Product Splines: From 1-D Bases to 2-D Interaction Surfaces", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    out_dir = Path(__file__).resolve().parent.parent / "docs" / "_static" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tensor_product_interaction.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    create_tensor_product_figure()
