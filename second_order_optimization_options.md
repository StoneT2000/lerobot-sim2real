# Second-Order Optimization Options for RBSolver Camera Calibration

## Overview

The current RBSolver implementation uses first-order optimization (likely Adam) to optimize camera extrinsics via mask overlap loss. With only 6 DOF parameters (3 translation + 3 rotation in se(3) Lie algebra), we can leverage second-order information for significantly faster convergence.

## Current Setup Analysis

- **Parameter Space**: 6 DOF (`self.dof` - se(3) Lie algebra representation)
- **Loss Function**: Mean squared error between rendered and reference masks
- **Differentiable Renderer**: NVDiffrastRenderer provides gradients through rendering

## Second-Order Optimization Options

### 1. **Gauss-Newton Method**

**Pros:**

- Specifically designed for least-squares problems (which MSE loss is)
- Avoids computing full Hessian by approximating with J^T J
- Often converges in fewer iterations than first-order methods

**Implementation Strategy:**

```python
# Pseudo-code
J = compute_jacobian(loss_per_pixel)  # Shape: [H*W, 6]
H_approx = J.T @ J  # Gauss-Newton approximation
g = J.T @ residuals
update = -solve(H_approx, g)
```

**Considerations:**

- Need to flatten mask differences to get per-pixel residuals
- May need damping (Levenberg-Marquardt) for stability

### 2. **L-BFGS (Limited-memory BFGS)**

**Pros:**

- PyTorch native support via `torch.optim.LBFGS`
- Approximates inverse Hessian using gradient history
- Good balance between memory usage and convergence speed
- No explicit Hessian computation needed

**Implementation:**

```python
optimizer = torch.optim.LBFGS(
    [self.dof],
    lr=1.0,
    max_iter=20,
    history_size=10,
    line_search_fn='strong_wolfe'
)
```

**Considerations:**

- Requires closure function for line search
- May need careful tuning of max_iter per step

### 3. **Natural Gradient Descent with Fisher Information Matrix**

**Pros:**

- Fisher Information Matrix (FIM) provides geometry-aware updates
- Particularly effective for probabilistic models
- Can interpret mask overlap as Bernoulli likelihood

**Implementation Strategy:**

```python
# Approximate FIM for Bernoulli mask prediction
p = rendered_masks  # predicted probabilities
F = sum_over_pixels(p * (1 - p) * (grad_p @ grad_p.T))
natural_grad = solve(F + λI, gradient)  # λ for regularization
```

**Considerations:**

- FIM can be approximated empirically or analytically
- Regularization crucial for numerical stability

### 4. **Trust Region Methods**

**Pros:**

- Robust convergence guarantees
- Automatically adapts step size
- Can use either exact or approximate Hessian

**Options:**

- **Dogleg Method**: Good for exact Hessian computation
- **Steihaug-CG**: Efficient for large-scale problems using Hessian-vector products

### 5. **Hessian-Free Optimization (Truncated Newton)**

**Pros:**

- Uses conjugate gradient to solve Newton system
- Only requires Hessian-vector products (no explicit Hessian)
- Can be implemented using automatic differentiation

**Implementation Strategy:**

```python
def hessian_vector_product(v):
    # Compute Hv using double backward pass
    g = torch.autograd.grad(loss, dof, create_graph=True)[0]
    Hv = torch.autograd.grad(g @ v, dof)[0]
    return Hv

# Solve Hd = -g using CG with HVP
```

## Recommended Approach

### Phase 1: L-BFGS (Immediate Implementation)

- Easiest to implement with PyTorch
- Likely to provide significant speedup over Adam
- Good baseline for comparison

### Phase 2: Gauss-Newton with Levenberg-Marquardt

- Natural fit for least-squares mask loss
- Can leverage problem structure
- Add adaptive damping for robustness

### Phase 3: Hybrid Approach

- Start with L-BFGS for initial convergence
- Switch to Gauss-Newton for final refinement
- Use trust region for stability

## Implementation Considerations

### 1. **Lie Group Constraints**

- Updates should respect SE(3) manifold structure
- Consider using exponential map for updates
- May need to adjust step sizes accordingly

### 2. **Numerical Stability**

- Add regularization to condition number
- Use Cholesky decomposition with pivoting
- Monitor condition number of approximate Hessian

### 3. **Computational Efficiency**

- Cache rendered masks gradients
- Exploit sparsity in Jacobian (many pixels don't overlap)
- Consider mini-batching over frames

### 4. **Convergence Criteria**

- Monitor both parameter and loss convergence
- Check gradient norm
- Use relative tolerance for robustness

## Practical Implementation Tips

1. **Start Simple**: Begin with L-BFGS as it requires minimal code changes
2. **Profile First**: Measure where computation time is spent (rendering vs optimization)
3. **Adaptive Methods**: Consider switching optimizers based on convergence phase
4. **Regularization**: Add small ridge term to Hessian approximations for stability
5. **Line Search**: Critical for second-order methods - use backtracking or Wolfe conditions

## Expected Benefits

- **Convergence Speed**: 5-10x fewer iterations typical for second-order methods
- **Final Accuracy**: Better convergence to true optimum
- **Robustness**: Less sensitive to learning rate tuning
- **Predictable Behavior**: Convergence in 10-50 iterations vs 100s-1000s

## Code Sketch for L-BFGS Integration

```python
def optimize_extrinsic(self, data_loader, num_epochs=10):
    def closure():
        optimizer.zero_grad()
        total_loss = 0
        for batch in data_loader:
            output = self.forward(batch)
            loss = output['mask_loss']
            total_loss += loss
        total_loss.backward()
        return total_loss

    optimizer = torch.optim.LBFGS(
        [self.dof],
        lr=1.0,
        max_iter=20,
        history_size=10,
        tolerance_grad=1e-5,
        tolerance_change=1e-9,
        line_search_fn='strong_wolfe'
    )

    for epoch in range(num_epochs):
        loss = optimizer.step(closure)
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
```

## References

- Nocedal & Wright, "Numerical Optimization" - Comprehensive treatment of second-order methods
- Martens, "Deep learning via Hessian-free optimization" - Application to neural networks
- Absil et al., "Optimization Algorithms on Matrix Manifolds" - For SE(3) considerations
