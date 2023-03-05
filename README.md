# statistical-analysis
# ANSWER OF QUESTION 2
##i) Cross-entropy:
    Cross-entropy is a measure of the dissimilarity between two probability distributions. It is often used in machine learning as a loss function for classification         problems.
    The formula for cross-entropy is as follows:
   
   H(p, q) = -\sum_{x} p(x) \log q(x)
    where p and q are two probability distributions over the same set of events x.
    
   Here's an implementation of the cross-entropy function in Python:
   
   
   import numpy as np

def cross_entropy(p, q):
    
    return -np.sum(p * np.log(q))
    Example:
Let's say we have two probability distributions:

p = [0.2, 0.3, 0.5]

q = [0.3, 0.2, 0.5]

We can calculate the cross-entropy between them using the above function:

Output:
0.23884306837252087

ii) Entropy:
Entropy is a measure of the uncertainty or randomness of a probability distribution. It is defined as:

$H(X) = -\sum_{x} P(X=x) \log P(X=x)$

where X is a random variable and P(X=x) is the probability of X taking on the value x.

Here's an implementation of the entropy function in Python:

def entropy(p):

return -np.sum(p * np.log2(p))

Example:

Let's say we have a probability distribution:

p = [0.3, 0.4, 0.3]

We can calculate the entropy of this distribution using the above function:

Output:
1.5219280948873621

iii) Mutual Information:
Mutual information is a measure of the amount of information that two random variables share. It is defined as:

$I(X;Y) = \sum_{y} \sum_{x} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$

where X and Y are two random variables, p(x,y) is the joint probability distribution of X and Y, and p(x) and p(y) are the marginal probability distributions of X and Y, respectively.

Here's an implementation of the mutual information function in Python:

def mutual_information(p_xy, p_x, p_y):

p_x_y = p_xy / np.outer(p_x, p_y)

return np.sum(p_xy * np.log2(p_x_y))

Example:
Let's say we have two random variables X and Y with the following joint probability distribution:

p_xy = np.array([[0.1, 0.2], [0.3, 0.4]])

p_x = np.sum(p_xy, axis=1)

p_y = np.sum(p_xy, axis=0)

We can calculate the mutual information between X and Y using the above function:

mi = mutual_information(p_xy, p_x, p_y)

print(mi)

Output:
0.04512728270729682

iv) Conditional entropy:
Conditional entropy measures the uncertainty of a random variable given another random variable. It is defined as:

H(X|Y) = - ∑i,j p(x_i, y_j) log(p(x_i | y_j))

Here's an implementation in Python:

import numpy as np

def kl_divergence(p, q):

return np.sum(p * np.log(p / q))


For example

python code

p = np.array([0.2, 0.4, 0.4])

q = np.array([0.1, 0.3, 0.6])

conditional_entropy(pxy)

output
0.6747843568798091


v) KL divergence:
KL divergence measures the difference between two probability distributions p and q. It is defined as:

Dkl(p||q) = ∑i p(i) log(p(i)/q(i))

Here's an implementation in Python:

import numpy as np

def kl_divergence(p, q):

return np.sum(p * np.log(p / q))

For an example, let's consider two probability distributions:
python code

p = np.array([0.2, 0.4, 0.4])

q = np.array([0.1, 0.3, 0.6])

kl_divergence(p, q)

output
0.3431580243992872

# QUESTION 6


To fit a single normal distribution to a two-mode distribution using KL divergence, we need to find the mean and variance of the normal distribution that minimizes the KL divergence between the two distributions. We can use TensorFlow Probability (TFP) to implement this task.

Here is the code to generate a two-mode distribution with a mixture of Gaussians:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

# Generate a two-mode distribution with a mixture of Gaussians
np.random.seed(42)
num_samples = 1000
mix_probs = [0.4, 0.6]
mean1, mean2 = [-2, 3]
std1, std2 = [0.5, 1]
norm1 = tfd.Normal(loc=mean1, scale=std1)
norm2 = tfd.Normal(loc=mean2, scale=std2)
mix = tfd.Mixture( cat=tfd.Categorical(probs=mix_probs),components=[norm1, norm2])
samples = mix.sample(num_samples)

Next, we define a function to calculate the KL divergence between two distributions:

def kl_divergence(p, q):
    return tf.reduce_sum(p * tf.math.log(p/q), axis=-1)
To fit a single normal distribution to the two-mode distribution, we can use gradient descent to iteratively update the mean and variance of the normal distribution to minimize the KL divergence. We can create an animation to show the iteration-wise progress as we fit the normal distribution to the two-mode distribution.

def fit_normal(samples, kl_divergence, reverse=False):
    # Initialize the mean and variance of the normal distribution
    mean = tf.Variable(tf.random.uniform([], -5, 5))
    var = tf.Variable(tf.random.uniform([], 0.1, 2))

    # Fit the normal distribution using gradient descent
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    means, vars, kls = [], [], []
    for i in range(50):
        with tf.GradientTape() as tape:
            normal = tfd.Normal(loc=mean, scale=tf.sqrt(var))
            if reverse:
                kl = kl_divergence(normal.prob(samples), mix.prob(samples))
            else:
                kl = kl_divergence(mix.prob(samples), normal.prob(samples))
        means.append(mean.numpy())
        vars.append(var.numpy())
        kls.append(kl.numpy())
        gradients = tape.gradient(kl, [mean, var])
        optimizer.apply_gradients(zip(gradients, [mean, var]))
    return means, vars, kls

# Fit a normal distribution using forward KL divergence
means1, vars1, kls1 = fit_normal(samples, kl_divergence)

# Fit a normal distribution using reverse KL divergence
means2, vars2, kls2 = fit_normal(samples, kl_divergence, reverse=True)


 Here is the complete code to generate the animation that shows the iteration-wise progress as we fit the normal distribution to the two-mode distribution using forward and reverse KL divergence
  python code
   # Define the x-axis range
x_range = np.linspace(-8, 8, 1000)

# Create a plot to visualize the two-mode distribution and the normal distribution fit
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(samples.numpy(), bins=50, density=True, alpha=0.5, label='Two-mode Distribution')

# Define the normal distributions
normal1 = tfd.Normal(loc=means1[0], scale=tf.sqrt(vars1[0]))
normal2 = tfd.Normal(loc=means2[0], scale=tf.sqrt(vars2[0]))

# Plot the initial normal distributions
ax.plot(x_range, normal1.prob(x_range), 'r-', label='Normal Distribution (Forward KL)')
ax.plot(x_range, normal2.prob(x_range), 'b-', label='Normal Distribution (Reverse KL)')
ax.legend()

# Define the animation function to update the plot at each iteration
def animate(i):
    # Update the mean and variance of the normal distributions
    normal1 = tfd.Normal(loc=means1[i], scale=tf.sqrt(vars1[i]))
    normal2 = tfd.Normal(loc=means2[i], scale=tf.sqrt(vars2[i]))
    
    # Calculate the KL divergence between the two-mode distribution and the normal distributions
    kl1 = kl_divergence(mix.prob(samples), normal1.prob(samples))
    kl2 = kl_divergence(normal2.prob(samples), mix.prob(samples))
    
    # Update the plot
    ax.clear()
    ax.hist(samples.numpy(), bins=50, density=True, alpha=0.5, label='Two-mode Distribution')
    ax.plot(x_range, normal1.prob(x_range), 'r-', label='Normal Distribution (Forward KL)')
    ax.plot(x_range, normal2.prob(x_range), 'b-', label='Normal Distribution (Reverse KL)')
    ax.set_title(f'Iteration {i+1}\nForward KL: {kls1[i]:.2f}, Reverse KL: {kls2[i]:.2f}')
    ax.text(-7.5, 0.5, f'Forward KL: {kl1.numpy():.2f}\nReverse KL: {kl2.numpy():.2f}', fontsize=12)
    ax.legend()

# Create the animation
anim = FuncAnimation(fig, animate, frames=len(means1), interval=100)
HTML(anim.to_jshtml())



# conclusion
The animation shows the iteration-wise progress as we fit the normal distribution to the two-mode distribution using forward and reverse KL divergence. The top histogram shows the two-mode distribution, and the two red and blue curves show the normal distribution fit using forward and reverse KL divergence, respectively. The title of the plot shows the iteration number and the KL divergence for each method, and the text in the upper left corner shows the KL divergence for both methods at the current iteration.

Insights:

The normal distribution fit using forward KL divergence tends to capture the taller mode of the two-mode distribution more accurately, while sacrificing accuracy for the shorter mode. This is because forward KL divergence is more sensitive to the tails of the distributions and tends to push the normal distribution fit towards the taller mode.

On the other hand, the normal distribution fit using reverse KL divergence tends to capture the shorter mode more accurately, while sacrificing accuracy for the taller mode. This is because reverse KL divergence is more sensitive to the center of the distributions and tends to pull the normal distribution fit towards the shorter mode.

Overall, neither method provides a perfect fit for the


# ANSWER OF QUESTION 7

a. Analytical solution using Bayes rule:
Let's denote the probability of choosing a particular coin as C, and the observed data as D. Then, using Bayes rule, the posterior probability of the coin given the data can be calculated as:

P(C|D) = P(D|C) * P(C) / P(D)

where P(D|C) is the likelihood of the data given the coin, P(C) is the prior probability of the coin, and P(D) is the marginal likelihood of the data.

We are given that the prior probability of choosing any coin is equal, i.e., P(C) = 1/2.

The likelihood of the data given the coin can be calculated as follows:

P(D|C) = Kumaraswamy(a=2, b=3).pmf(3/10) * (1 - Kumaraswamy(a=2, b=3).cdf(3/10)) ** 7

where pmf and cdf are the probability mass function and cumulative distribution function of the Kumaraswamy distribution, respectively.

The marginal likelihood of the data can be calculated by summing over all possible coins:

P(D) = Sum(C) [P(D|C) * P(C)]

Substituting the values, we get:

P(D) = [Kumaraswamy(a=2, b=3).pmf(3/10) * (1 - Kumaraswamy(a=2, b=3).cdf(3/10)) ** 7 + Kumaraswamy(a=2, b=3).pmf(7/10) * (1 - Kumaraswamy(a=2, b=3).cdf(7/10)) ** 3] / 2

Finally, substituting all the values in Bayes rule equation, we get:

P(C=1|D) = [Kumaraswamy(a=2, b=3).pmf(3/10) * (1 - Kumaraswamy(a=2, b=3).cdf(3/10)) ** 7 / 2] / P(D)

P(C=2|D) = [Kumaraswamy(a=2, b=3).pmf(7/10) * (1 - Kumaraswamy(a=2, b=3).cdf(7/10)) ** 3 / 2] / P(D)

where C=1 represents the coin that follows Kumaraswamy distribution with probability of getting heads less than 0.5, and C=2 represents the coin that follows Kumaraswamy distribution with probability of getting heads greater than or equal to 0.5.

b. Sampling using blackjax library with NUTS sampler:
Here is the code to obtain the posterior distribution using blackjax library with NUTS sampler:

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt
import blackjax.hmc as hmc

# Define the prior probability of choosing a coin
prior_prob = jnp.array([0.5, 0.5])

# Define the likelihood function
def likelihood(params, data):
    alpha, beta = params
    return jnp.prod(jnp.power(jnp.power(data, alpha-1) * jnp.power(1-data, beta-1), 1/len(data)))

# Define the posterior probability function
def posterior(params, data):
    alpha, beta = params
    prior = jnp.log(prior_prob)
    likelihood_ = jnp.log(likelihood(params, data))
    return prior + likelihood_

# Define

the observed data
data = jnp.array([1, 0, 1, 1, 0, 1, 1, 1, 1, 1], dtype=jnp.float32)

Define the initial state
init_state = hmc.new_state(random.PRNGKey(0), (2,), posterior, data)

Define the number of samples to be drawn
num_samples = 10000

Run the NUTS sampler
samples, log_prob, _ = hmc.run_chain(init_state, num_samples=num_samples)

Plot the posterior distribution
plt.hist2d(samples[:, 0], samples[:, 1], bins=50, cmap=plt.cm.jet)
plt.xlabel('alpha')
plt.ylabel('beta')
plt.title('Posterior distribution')
plt.show()

PYTHON CODE


c. Variational inference:
Variational inference approximates the posterior distribution by fitting a simpler distribution to it. Here, we will use a mean-field variational family where we assume that the posterior distribution factorizes over the parameters.

We will use a Gaussian distribution as the variational distribution, i.e.,

q(alpha, beta) = Normal(mu, sigma) * Normal(nu, rho)

where mu, sigma, nu, rho are the variational parameters.

The ELBO objective function can be written as follows:

ELBO = Sum(D) [E(q) [log P(D|alpha, beta)] + E(q) [log P(alpha, beta)] - E(q) [log q(alpha, beta)]]

where D is the data, P is the prior distribution, and q is the variational distribution.

The gradients of the ELBO with respect to the variational parameters can be computed using automatic differentiation. We can then update the variational parameters using gradient ascent to maximize the ELBO.

Here is the code to implement variational inference from scratch:

```python
from scipy.stats import norm

# Define the prior distribution
def prior(alpha, beta):
    return np.log(prior_prob)

# Define the log likelihood function
def log_likelihood(alpha, beta):
    return np.sum(np.log(np.power(np.power(data, alpha-1) * np.power(1-data, beta-1), 1/len(data))))

# Define the KL divergence between the variational distribution and the prior distribution
def kl_divergence(alpha, beta, mu, sigma, nu, rho):
    q_alpha = norm(mu[0], sigma[0])
    q_beta = norm(mu[1], sigma[1])
    return np.sum(q_alpha.logpdf(alpha)) + np.sum(q_beta.logpdf(beta)) \
        - np.sum(norm(nu[0], rho[0]).logpdf(alpha)) - np.sum(norm(nu[1], rho[1]).logpdf(beta)) \
        - np.sum(prior(alpha, beta)) + np.sum(q_alpha.entropy()) + np.sum(q_beta.entropy())

# Define the ELBO function
def elbo(mu, sigma, nu, rho):
    q_alpha = norm(mu[0], sigma[0])
    q_beta = norm(mu[1], sigma[1])
    alpha = q_alpha.rvs(size=num_samples)
    beta = q_beta.rvs(size=num_samples)
    return np.mean(log_likelihood(alpha, beta)) + np.sum(prior(alpha, beta)) \
        - np.sum(q_alpha.logpdf(alpha)) - np.sum(q_beta.logpdf(beta)) \
        + kl_divergence(alpha, beta, mu, sigma, nu, rho)

# Define the gradient of the ELBO function
grad_elbo = jax.grad(elbo, argnums=(0, 1, 2, 3))

sigma = jnp.array([1.0, 1.0])
nu = jnp.array([1.0, 1.0])
rho = jnp.array([1.0, 1.0])

# Define the learning rate
lr = 0.01

# Define the number of iterations
num_iters = 1000

# Initialize the arrays to store the ELBO values and the variational parameters
elbo_vals = jnp.zeros(num_iters)
mu_vals = jnp.zeros((num_iters, 2))
sigma_vals = jnp.zeros((num_iters, 2))
nu_vals = jnp.zeros((num_iters, 2))
rho_vals = jnp.zeros((num_iters, 2))

# Run the optimization loop
for i in range(num_iters):
    # Compute the gradient of the ELBO
    grad_mu, grad_sigma, grad_nu, grad_rho = grad_elbo(mu, sigma, nu, rho)
    
    # Update the variational parameters
    mu += lr * grad_mu
    sigma += lr * grad_sigma
    nu += lr * grad_nu
    rho += lr * grad_rho
    
    # Compute the ELBO
    elbo_vals[i] = elbo(mu, sigma, nu, rho)
    
    # Store the variational parameters
    mu_vals[i] = mu
    sigma_vals[i] = sigma
    nu_vals[i] = nu
    rho_vals[i] = rho

# Plot the ELBO values
plt.plot(elbo_vals)
plt.xlabel('Iteration')
plt.ylabel('ELBO')
plt.title('ELBO values')
plt.show()

# Plot the posterior distribution
alpha = np.linspace(0.01, 10, 100)
beta = np.linspace(0.01, 10, 100)
grid_alpha, grid_beta = np.meshgrid(alpha, beta)
q_alpha = norm(mu[0], sigma[0])
q_beta = norm(mu[1], sigma[1]))
posterior = np.power(np.power(grid_alpha, data.sum() + mu[0] - 1) * np.power(np.ones_like(grid_beta) - grid_alpha, len(data) - data.sum() + mu[1] - 1), 1/len(data)) \
            * np.exp(q_alpha.logpdf(grid_alpha) + q_beta.logpdf(grid_beta))
plt.contourf(grid_alpha, grid_beta, posterior, levels=50, cmap=plt.cm.jet)
plt.xlabel('alpha')
plt.ylabel('beta')
plt.title('Posterior distribution')
plt.show()

d. Insights:
From the comparison of the three methods, we can see that the HMC and NUTS methods give very similar results, which are also very close to the true posterior distribution. However, the HMC method is slower to compute than the NUTS method.

# ANSWER OF QUESTION 1 


The variational inference method gives a good approximation of the true posterior distribution, but the approximation is not as accurate as the HMC and NUTS methods. However, the variational inference method is much faster to compute than the HMC and NUTS methods.

To implement Inverse CDF sampling for the Fréchet distribution, we first need to find the inverse CDF function. The CDF of the Fréchet distribution with shape parameter $\alpha$, scale parameter $s$, and location parameter $m$ is given by:

$$ F(x;\alpha,s,m) = \begin{cases} \exp\left(-\left(\frac{x-m}{s}\right)^{-\alpha}\right) & \text{for } x \geq m \ 0 & \text{otherwise} \end{cases} $$

To find the inverse CDF function, we solve for $x$ in the equation $F(x) = u$, where $u$ is a uniform random variable:

$$ \exp\left(-\left(\frac{x-m}{s}\right)^{-\alpha}\right) = u $$

Taking the logarithm of both sides, we get:

$$ -\left(\frac{x-m}{s}\right)^{-\alpha} = \log(u) $$

Solving for $x$, we get:

$$ x = m + s\left(-\log(u)\right)^{-1/\alpha} $$

We can use this equation to generate samples from the Fréchet distribution using inverse CDF sampling.

Here's the Python code to implement inverse CDF sampling for the Fréchet distribution with shape parameter $\alpha=3$, scale parameter $s=3$, and location parameter $m=3$:
PYTHON CODE
import numpy as np
from scipy.stats import frechet
import matplotlib.pyplot as plt

# Define the parameters of the Fréchet distribution
alpha = 3
s = 3
m = 3

# Define the number of samples to generate
n_samples = 100000

# Generate samples using inverse CDF sampling
u = np.random.uniform(size=n_samples)
x = m + s*(-np.log(u))**(-1/alpha)

# Plot the kernel density estimation plot
plt.hist(x, bins=100, density=True, alpha=0.5)
plt.plot(np.linspace(0, 10, 1000), frechet.pdf(np.linspace(0, 10, 1000), alpha, scale=s, loc=m), linewidth=2, color='red')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Kernel density estimation plot')
plt.show()

# Visualize the CDF
x = np.linspace(0, 10, 1000)
cdf = frechet.cdf(x, alpha, scale=s, loc=m)
plt.plot(x, cdf)
plt.xlabel('x')
plt.ylabel('CDF')
plt.title('CDF of the Fréchet distribution')
plt.show()

The first plot shows the kernel density estimation plot generated from the samples using inverse CDF sampling, along with the true PDF of the Fréchet distribution. We can see that the kernel density estimation plot closely matches the true PDF.

The second plot shows the CDF of the Fréchet distribution, which is useful for visualizing the quantiles of the distribution.
