# **Gaussian Process Bayesian Optimization (GPBO) Model**

Note: At the time of this project i was at the start of my third year chemical engineering degree and in a lab decided to apply Bayesian Optimisation. Initialy due to lack of time I was not able to develop my own bayesian optimisation tool so this work is an expansion on my previous work (which i believe deserves its own repository) which allows for a good model comparison to the gp minimize from scikit learn. For the original repository where details of the optimisation process are explained refer to the link below:

https://github.com/barbacci-marco/Optimization-of-Methanol-Water-Distillation-Process-Using-Bayesian-Optimization/tree/main 

---

Welcome to the documentation of the Gaussian Process Bayesian Optimization (GPBO) model. This repository provides an implementation of Bayesian Optimization using Gaussian Processes, with a focus on optimizing functions where evaluations are expensive. The model includes a Trust Region approach to balance exploration and exploitation efficiently.

---

## **Introduction**

Bayesian Optimization is a powerful technique for optimizing objective functions that are expensive to evaluate. It is particularly useful when:

- The function lacks an analytical expression.
- The function evaluation is costly (e.g., requires running complex simulations or experiments).
- Derivatives of the function are unavailable.

Gaussian Processes (GP) provide a probabilistic approach to modeling the objective function, capturing both the mean and uncertainty of predictions. By leveraging GPs within Bayesian Optimization, we can make informed decisions about where to sample next, balancing the trade-off between exploration (sampling where uncertainty is high) and exploitation (sampling where the mean prediction is optimal).

This implementation enhances the standard Bayesian Optimization by incorporating a **Trust Region** method, which dynamically adjusts the search space based on the progress of the optimization.

---

## **Prerequisites**

- Python 3.6 or higher
- NumPy
- Matplotlib (for visualization)
- SciPy (optional, for advanced optimization techniques)
- scikit-learn (for comparison purposes)

---

## **Installation**

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/barbacci_marco/gpbo.git
cd gpbo
```

Install the required packages:

```bash
pip install numpy matplotlib scipy scikit-learn
```

---

## **Implementation Details**

The GPBO model consists of several key components:

### **Kernel Function**

The kernel function defines the covariance between points in the input space. We use the Radial Basis Function (RBF) kernel, also known as the squared exponential kernel.

```python
def rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    sqdist = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2)
    return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)
```

- **Parameters**:
  - `length_scale`: Controls the smoothness of the function.
  - `sigma_f`: Signal variance; scales the overall variance of the function.

### **Gaussian Process Posterior**

Computes the posterior mean and covariance of the GP given the training data and test points.

```python
def gp_posterior(X_train, y_train, X_test, kernel, sigma_y):
    K = kernel(X_train, X_train) + sigma_y**2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_test)
    K_ss = kernel(X_test, X_test) + sigma_y**2 * np.eye(len(X_test))
    
    K += 1e-8 * np.eye(len(K))  # Add jitter for numerical stability
    
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    
    mu_s = K_s.T.dot(alpha)
    v = np.linalg.solve(L, K_s)
    cov_s = K_ss - v.T.dot(v)
    
    return mu_s.flatten(), cov_s
```

- **Parameters**:
  - `sigma_y`: Noise level in the observations.

### **Acquisition Function**

We use the **Expected Improvement (EI)** acquisition function to determine the next sampling point.

```python
def expected_improvement(X, X_train, y_train, mu, sigma, f_best, xi=0.01):
    sigma = np.maximum(sigma, 1e-8)
    Z = (mu - f_best - xi) / sigma
    phi = standard_normal_pdf(Z)
    Phi = standard_normal_cdf(Z)
    ei = (mu - f_best - xi) * Phi + sigma * phi
    return ei
```

- **Parameters**:
  - `xi`: Exploitation-exploration trade-off parameter.

### **Trust Region Method**

The trust region adjusts the search space dynamically based on the optimization progress.

```python
def update_trust_region(trust_region_center, trust_region_radius, X_new, y_new, y_best, bounds, shrink_factor=0.8, expand_factor=1.3):
    if y_new < y_best:
        trust_region_radius *= expand_factor
    else:
        trust_region_radius *= shrink_factor
    
    # Ensure the radius stays within reasonable bounds
    min_radius = 0.01 * np.ptp(bounds, axis=1)
    max_radius = 0.5 * np.ptp(bounds, axis=1)
    trust_region_radius = np.maximum(np.minimum(trust_region_radius, max_radius), min_radius)
    
    # Update the center to the new point
    trust_region_center = X_new
    return trust_region_center, trust_region_radius
```

- **Parameters**:
  - `shrink_factor` and `expand_factor`: Control how the trust region radius adjusts.

---

## **Usage**

### **Example: Optimizing the Rosenbrock Function**

The Rosenbrock function is a common test function for optimization algorithms. It has a global minimum at \((1, 1)\).

**Define the Objective Function:**

```python
def sample_loss(X):
    x, y = X
    return (1 - x)**2 + 100 * (y - x**2)**2
```

**Set the Bounds:**

```python
bounds = np.array([[-2, 2], [-1, 3]])
```

**Run the Optimization:**

```python
# Set the random seed for reproducibility
np.random.seed(42)

# Perform Bayesian Optimization
X_res, y_res, iterations = bayesian_optimization_with_trust_region(
    n_iters=30,
    sample_loss=sample_loss,
    bounds=bounds,
    n_pre_samples=5,
    alpha=1e-6,
    initial_trust_radius=0.5
)
```

**Results:**

```python
# Find the best result
best_idx = np.argmin(y_res)
optimal_parameters = X_res[best_idx]
optimal_x = optimal_parameters[0]
optimal_y = optimal_parameters[1]
minimum_value = y_res[best_idx]

print(f"Optimal parameters found: x = {optimal_x:.4f}, y = {optimal_y:.4f}")
print(f"Minimum value of the objective function: {minimum_value:.4f}")
```

**Visualization:**

```python
import matplotlib.pyplot as plt

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(iterations, y_res[n_pre_samples:], marker='o')
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.title('Convergence of Bayesian Optimization')
plt.grid(True)
plt.show()
```
---

## **References**

- Rasmussen, C. E., & Williams, C. K. I. (2006). **Gaussian Processes for Machine Learning**. MIT Press.
- Brochu, E., Cora, V. M., & de Freitas, N. (2010). **A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning**. arXiv preprint arXiv:1012.2599.
- Nocedal, J., & Wright, S. J. (2006). **Numerical Optimization**. Springer.
- E. A. del Rio Chanona, P. Petsagkourakis, E. Bradford, J. E. Alves Graciano, B. Chachuat,
Real-time optimization meets Bayesian optimization and derivative-free optimization: A tale of modifier adaptation, Computers & Chemical Engineering, Volume 147, 2021, 107249, ISSN 0098-1354, https://doi.org/10.1016/j.compchemeng.2021.107249.
(https://www.sciencedirect.com/science/article/pii/S0098135421000272)
- Shi, Y. (2021) Gaussian processes, not quite for dummies, The Gradient. Available at: https://thegradient.pub/gaussian-process-not-quite-for-dummies/ (Accessed: 21 November 2024).
- Louppe, G. and Kumar, M. (2017) Bayesian optimization with SKOPT, scikit. Available at: https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html (Accessed: 21 November 2024).
- Dürholt, J.P. et al. (2024) BoFire: Bayesian Optimization Framework intended for real experiments, arXiv.org. Available at: https://arxiv.org/abs/2408.05040 (Accessed: 01 November 2024). 
---

## **License**

This project is licensed under the MIT License - see the license file for details.

---

**Note:** For any questions or contributions, please open an issue or submit a pull request.
