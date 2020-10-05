import gpflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# from gpflow.utilities import print_summary
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


# Setup path to save figures to
figure_path = "/home/jordan/school/rmds/fall_2020_workshops/workshop_4/figures/"

# Generate data and fit linear regression model
# for linear regression example plot
n = 100
x = np.random.uniform(0, 20, n).reshape((n,1))
y = 3*x + np.random.normal(0, 3, n).reshape((n,1))
lr = LinearRegression()
lr.fit(x, y)

# Plot linear regression model/data and save
line_X = np.arange(x.min(), x.max())[:, np.newaxis]
line_y = lr.predict(line_X)
plt.figure()
plt.scatter(x, y)
plt.plot(line_X, line_y)
plt.show()
plt.savefig(figure_path + 'ols.png')

# Generate nonlinear data (sine function) and
# estimate model using k-nearest neighbors.
# Save figure of data and estimated model.
n = 100
x = np.random.uniform(0, 2*np.pi, n).reshape((n,1))
y = np.sin(x) + np.random.normal(0, .3, n).reshape((n,1))
plt.figure()
plt.scatter(x, y)
plt.show()
plt.savefig(figure_path + 'sin.png')

neigh = KNeighborsRegressor(n_neighbors=5)
neigh.fit(x, y)
line_X = np.arange(x.min(), x.max(), .1)
line_X = line_X.reshape((len(line_X), 1))
line_Y = neigh.predict(line_X)

plt.figure()
plt.scatter(x, y)
plt.plot(line_X, line_Y)
# plt.show()
plt.savefig(figure_path + 'sin-knn.png')


# Now fit a Gaussian Process to the data
import gstools as gs

# Simulate a 1d GRF with variance 1 and length scale 10
x = range(100)
model = gs.Gaussian(dim=1, var=1, len_scale=10)
srf = gs.SRF(model)
srf((x), mesh_type='structured')
srf.plot()


# Simulate a 2d GRF with variance 1 and length scale 10
x = y = range(100)
model = gs.Gaussian(dim=2, var=1, len_scale=1)
srf = gs.SRF(model)
srf((x, y), mesh_type='structured')
srf.plot()

########################
## Gaussian Processes ##
########################

# Fit a GP Regression model to data

# Fake data:
X = [0.01, 0.02, 0.025, 0.18, 0.19, 0.3, 0.67, .76, .79, 0.88, 0.88, 0.9]
Y = [3.75, 3.80, 3.70, 3.30, 3.45, 3.80, 3.65, 3.75, 3.00, 1.60, 1.55, 1.40]
X = np.array(X).reshape(-1, 1)
Y = np.array(Y).reshape(-1, 1)

# Plot data:
plt.figure()
plt.plot(X, Y, "kx", mew=2)
plt.xlim(-.1, 1.1)
plt.ylim(-5, 7)
plt.savefig(figure_path + 'gp-data.png')
#plt.show()

# Model using RBF kernel
k = gpflow.kernels.SquaredExponential()
m = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=None)

# Fit model by minimizing training loss
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

## generate test points for prediction
xx = np.linspace(-0.1, 1.1, 100).reshape(100, 1)  # test points must be of shape (N, D)

## predict mean and variance of latent GP at test points
mean, var = m.predict_f(xx)

# generate 20 samples from posterior and plot
tf.random.set_seed(1)  # for reproducibility
samples = m.predict_f_samples(xx, 20)  # shape (20, 100, 1)

# plot predictions with credible intervals
#plt.figure(figsize=(12, 6))
plt.figure()
plt.xlim(-.1, 1.1)
plt.ylim(-5, 7)
plt.plot(X, Y, "kx", mew=2)
plt.plot(xx, mean, "red", lw=2)
plt.fill_between(xx[:, 0],
    mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
    mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
    color="C0", alpha=0.2)
plt.plot(xx, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
_ = plt.xlim(-0.1, 1.1)
plt.savefig(figure_path + 'gp-model.png')
plt.show()


# Play around with what the model would look like with different estimates
# for variance and length scales in the kernel
tf.random.set_seed(1)  # for reproducibility
m.likelihood.variance.assign(0.1)
m.kernel.lengthscales.assign(10)
mean, var = m.predict_f(xx)
samples = m.predict_f_samples(xx, 20)  # shape (10, 100, 1)

## plot
#plt.figure(figsize=(12, 6))
plt.figure()
plt.title("sigma = 0.1, ell = 10")
plt.xlim(-.1, 1.1)
plt.ylim(-5, 7)
plt.plot(X, Y, "kx", mew=2)
plt.plot(xx, mean, "red", lw=2)
plt.fill_between(
    xx[:, 0],
    mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
    mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
    color="C0",
    alpha=0.2,
)

plt.plot(xx, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
_ = plt.xlim(-0.1, 1.1)
plt.savefig(figure_path + 'gp-model-0_1-10.png')
plt.show()


m.likelihood.variance.assign(0.1)
m.kernel.lengthscales.assign(1)
mean, var = m.predict_f(xx)
samples = m.predict_f_samples(xx, 20)  # shape (20, 100, 1)

plt.figure()
plt.title("sigma = 0.1, ell = 1")
plt.xlim(-.1, 1.1)
plt.ylim(-5, 7)
plt.plot(X, Y, "kx", mew=2)
plt.plot(xx, mean, "red", lw=2)
plt.fill_between(
    xx[:, 0],
    mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
    mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
    color="C0",
    alpha=0.2,
)
plt.plot(xx, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
_ = plt.xlim(-0.1, 1.1)
plt.savefig(figure_path + 'gp-model-0_1-1.png')
plt.show()



m.likelihood.variance.assign(1)
m.kernel.lengthscales.assign(.1)
mean, var = m.predict_f(xx)
samples = m.predict_f_samples(xx, 20)  # shape (20, 100, 1)


plt.figure()
plt.title("sigma = 1, ell = .1")
plt.xlim(-.1, 1.1)
plt.ylim(-5, 7)
plt.plot(X, Y, "kx", mew=2)
plt.plot(xx, mean, "red", lw=2)
plt.fill_between(
    xx[:, 0],
    mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
    mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
    color="C0",
    alpha=0.2,
)

plt.plot(xx, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
_ = plt.xlim(-0.1, 1.1)
plt.savefig(figure_path + 'gp-model-1-0_1.png')
plt.show()


m.likelihood.variance.assign(.1)
m.kernel.lengthscales.assign(.1)
mean, var = m.predict_f(xx)
samples = m.predict_f_samples(xx, 20)  # shape (10, 100, 1)

plt.figure()
plt.title("sigma = .1, ell = .1")
plt.xlim(-.1, 1.1)
plt.ylim(-5, 7)
plt.plot(X, Y, "kx", mew=2)
plt.plot(xx, mean, "red", lw=2)
plt.fill_between(
    xx[:, 0],
    mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
    mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
    color="C0",
    alpha=0.2,
)

plt.plot(xx, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
_ = plt.xlim(-0.1, 1.1)
plt.savefig(figure_path + 'gp-model-0_1-0_1.png')
plt.show()
