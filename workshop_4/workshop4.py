import gpflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from gpflow.utilities import print_summary



n = 100
x = np.random.uniform(0, 20, n)
y = 3*x + np.random.normal(0, 3, n)
plt.scatter(x, y)
plt.show()



n = 100
x = np.random.uniform(0, 2*np.pi, n)
y = np.sin(x) + np.random.normal(0, .3, n)
plt.scatter(x, y)
plt.show()



# Motivate: I have some data I want to model... I don't care about the
# functional form of the model, but I want a mean and covariance for prediction
# points.

# First talk about Linear Regression
# y = f(x) = \theta_1 x_1 + \dots + \theta_p x_p + \epsilon = \mathbf{\theta}^T \mathbf{x} + \epsilon
# vector form:
# \mathbf{Y} = \mathbf{f}(\mathbf{x}) = \mathbf{X}^T \mathbf{\theta} + \mathbf{\epsilon}
# Can solve via:
# - maximum likelihood
# - Maximum a posteriori (MAP) estimation
# - Loss optimization (Least Squares)
# - maximum likelihood

# Maximum Likelihood:
# p(w | y) = \prod_{i = 1}^n p(y_i | x)
# arg max_{w} p(w|x)


# Least Squares (Loss Optimization)
# Min_w || \epsilon ||
# Min_w || y - \hat y ||
# put plot here

# Next talk about linear classifiers












## Gaussian Processes

# The lines below are specific to the notebook format
# %matplotlib inline
# plt.rcParams["figure.figsize"] = (12, 6)


# data = np.genfromtxt("data/regression_1D.csv", delimiter=",")
X = [0.01, 0.02, 0.025, 0.18, 0.19, 0.3, 0.67, .76, .79, 0.88, 0.88, 0.9]
Y = [3.75, 3.80, 3.70, 3.30, 3.45, 3.80, 3.65, 3.75, 3.00, 1.60, 1.55, 1.40]

X = np.array(X).reshape(-1, 1)
Y = np.array(Y).reshape(-1, 1)

# X = data[:, 0].reshape(-1, 1)
# Y = data[:, 1].reshape(-1, 1)

raw_plot = plt.plot(X, Y, "kx", mew=2)
plt.show()

k = gpflow.kernels.Matern52()

print_summary(k)

m = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=None)

print_summary(m)

# m.likelihood.variance.assign(0.01)
# m.kernel.lengthscales.assign(0.3)


opt = gpflow.optimizers.Scipy()


opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
print_summary(m)


## generate test points for prediction
xx = np.linspace(-0.1, 1.1, 100).reshape(100, 1)  # test points must be of shape (N, D)

xx.shape
## predict mean and variance of latent GP at test points
mean, var = m.predict_f(xx)

## generate 10 samples from posterior
tf.random.set_seed(1)  # for reproducibility
samples = m.predict_f_samples(xx, 10)  # shape (10, 100, 1)

## plot
plt.figure(figsize=(12, 6))
plt.plot(X, Y, "kx", mew=2)
plt.plot(xx, mean, "C0", lw=2)
plt.fill_between(
    xx[:, 0],
    mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
    mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
    color="C0",
    alpha=0.2,
)

plt.plot(xx, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
_ = plt.xlim(-0.1, 1.1)
plt.show()

