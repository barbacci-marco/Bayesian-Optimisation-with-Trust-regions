import numpy as np
import math
np.random.seed(16)
# RBF Kernel Function
def rbf_kernel(X1, X2, length_scale=2, sigma_f=1.0):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    sqdist = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2)
    return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)

# Standard Normal PDF
def standard_normal_pdf(z):
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z ** 2)

# Error Function Approximation
def erf(z):
    
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p  = 0.3275911
    
    sign = np.sign(z)
    z = np.abs(z)
    
    t = 1.0 / (1.0 + p * z)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-z * z)
    
    return sign * y

# Standard Normal CDF
def standard_normal_cdf(z):
    return 0.5 * (1 + erf(z / np.sqrt(2)))

# GP Posterior Function
def gp_posterior(X_train, y_train, X_test, kernel, sigma_y):
    K = kernel(X_train, X_train) + sigma_y**2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_test)
    K_ss = kernel(X_test, X_test) + sigma_y**2 * np.eye(len(X_test))
    
    K += 1e-8 * np.eye(len(K))
    
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    
    mu_s = K_s.T.dot(alpha)
    v = np.linalg.solve(L, K_s)
    cov_s = K_ss - v.T.dot(v)
    
    return mu_s.flatten(), cov_s

# Expected Improvement Function
def expected_improvement(X, X_train, y_train, mu, sigma, f_best, xi=0.01):
    sigma = np.maximum(sigma, 1e-8)
    Z = (mu - f_best - xi) / sigma
    phi = standard_normal_pdf(Z)
    Phi = standard_normal_cdf(Z)
    ei = (mu - f_best - xi) * Phi + sigma * phi
    return ei

# Initialize Trust Region
def initialize_trust_region(bounds, initial_radius):
    trust_region_center = None
    trust_region_radius = initial_radius
    return trust_region_center, trust_region_radius

def update_trust_region(trust_region_center, trust_region_radius, X_new, y_new, y_best, bounds, shrink_factor=0.8, expand_factor=1.3):
    '''Update the trust region based on the new observation for multi-dimensional input.'''
    if y_new < y_best:
        trust_region_radius *= expand_factor
    else:
        trust_region_radius *= shrink_factor

    # Ensure the radius stays within bounds
    min_radius = 0.01 * np.ptp(bounds, axis=1)
    max_radius = 0.5 * np.ptp(bounds, axis=1)
    trust_region_radius = np.maximum(np.minimum(trust_region_radius, max_radius), min_radius)

    # Update the center to the new point
    trust_region_center = X_new
    return trust_region_center, trust_region_radius


def random_acquisition_maximization(acquisition, X_train, y_train, kernel, trust_region_center, trust_region_radius, bounds, num_samples=1000, sigma_y=1e-8):
    '''Randomly sample points within the trust region and select the one that maximizes the acquisition function.'''
    # Generate random samples within the trust region
    lower_bounds = np.maximum(trust_region_center - trust_region_radius, bounds[:, 0])
    upper_bounds = np.minimum(trust_region_center + trust_region_radius, bounds[:, 1])
    samples = np.random.uniform(lower_bounds, upper_bounds, size=(num_samples, bounds.shape[0]))
    
    # Compute GP posterior for samples
    mu, cov = gp_posterior(X_train, y_train, samples, kernel, sigma_y)
    sigma = np.sqrt(np.diag(cov))
    f_best = np.min(y_train)
    
    # Compute acquisition values
    acquisition_values = np.array([
        expected_improvement(x.reshape(1, -1), X_train, y_train, mu_i, sigma_i, f_best)
        for x, mu_i, sigma_i in zip(samples, mu, sigma)
    ])
    
    # Select the point with the highest acquisition value
    idx_max = np.argmax(acquisition_values)
    X_next = samples[idx_max]
    
    return X_next


def bayesian_optimization_with_trust_region(n_iters, sample_loss, bounds, n_pre_samples=5, alpha=1e-8, initial_trust_radius=0.1):
    bounds = np.array(bounds)
    X_train = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_pre_samples, bounds.shape[0]))
    y_train = np.array([sample_loss(x) for x in X_train])
    
    y_best_idx = np.argmin(y_train)
    trust_region_center = X_train[y_best_idx]
    trust_region_radius = initial_trust_radius * np.ones(bounds.shape[0])
    
    iteration_numbers = []
    
    for i in range(n_iters):
        mu_s, cov_s = gp_posterior(X_train, y_train, X_train, rbf_kernel, sigma_y=alpha)
        
       
        X_next = random_acquisition_maximization(
            expected_improvement, X_train, y_train, rbf_kernel,
            trust_region_center, trust_region_radius, bounds, num_samples=1000, sigma_y=alpha
        )
        
        y_next = sample_loss(X_next)
        
        X_train = np.vstack((X_train, X_next.reshape(1, -1)))
        y_train = np.append(y_train, y_next)
        
        y_best = y_train[np.argmin(y_train)]
       
        trust_region_center, trust_region_radius = update_trust_region(
            trust_region_center, trust_region_radius, X_next, y_next, y_best, bounds
        )
        
        # Record the iteration number
        iteration_numbers.append(i + 1)
        
        print(f"Iteration {i+1}/{n_iters}, X_next = {X_next}, y_next = {y_next}, Trust Region Center = {trust_region_center}, Radius = {trust_region_radius}")
    
    return X_train, y_train, iteration_numbers
