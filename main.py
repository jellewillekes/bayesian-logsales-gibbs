import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import arviz as az

'''
***************************************
Settings of the Gibbs Sampler
***************************************
'''
data_folder_path = './data/'
dataset = 'brand48'  # corresponding to student number 609948jw

nos = 1000  # number of simulations
nob = 100  # number of burn-in simulations
nod = 1  # consider every nod-th draw (thin value)
trace_draws = 100  # number of draws for traceplot

# Initialize parameters (sigma0, sigma1, and beta)
sigma0_sq = 1
sigma1_sq = 1
beta = np.zeros(4)  # Initialize beta with zeros, size based on X dimension in assignment
'''
***************************************
Load and filter Data
***************************************
'''
# Load each DataFrame and filter based on dataset name
sales = pd.read_excel(data_folder_path + 'sales.xls', usecols=[dataset]).rename(columns={dataset: 'sales'})
price = pd.read_excel(data_folder_path + 'price.xls', usecols=[dataset]).rename(columns={dataset: 'price'})
coupon = pd.read_excel(data_folder_path + 'coupon.xls', usecols=[dataset]).rename(columns={dataset: 'coupon'})
display = pd.read_excel(data_folder_path + 'displ.xls', usecols=[dataset]).rename(columns={dataset: 'display'})

# Take logarithm of sales and price
logsales = np.log(sales).rename(columns={'sales': 'logsales'})
logprice = np.log(price).rename(columns={'price': 'logprice'})

# Combine data into one DataFrame
data = pd.concat([logsales, logprice['logprice'], coupon['coupon'], display['display']], axis=1)

# Print top rows of DataFrame
print(f"Top rows of data from dataset {dataset} \n {data.head()} \n")
'''
***************************************
Implementation of the Gibbs sampler
***************************************
'''
N = len(data)
beta_samples = []
sigma0_sq_samples = []
sigma1_sq_samples = []

# Create matrices and vectors for analysis
X = data[['logprice', 'coupon', 'display']].values
X = np.column_stack([np.ones(N), X])  # Add intercept
y = data['logsales'].values

print('Start Gibbs sampler...')
for i in range((nos * nod) + nob):
    """The Gibbs sampler is part of the Markov Chain Monte Carlo (MCMC) algorithm used generate 
    samples from the posterior distributions of the params β, σ0^2 and σ1^2 given the data of 
    a regression model.
    
    Given the regression context,
    - For β (beta), the posterior conditional distribution is multivariate normal.
    - For σ^2 (sigma squared), the posterior conditional distribution is inverse gamma.
    - The mean and the inverse of the variance-covariance matrix for the conditional 
        distribution of beta are computed given current sigma values and data."""
    if i % 1000 == 0:  # Keep track of iterations
        print(i)

    # Compute inverse of the variance-covariance matrix for beta's posterior
    inv_CoVar_beta = np.linalg.inv(
        X.T @ np.diag(1 / (sigma0_sq * (1 - data['display']) + sigma1_sq * data['display'])) @ X)

    # Compute the mean for beta's posterior
    mu_beta = inv_CoVar_beta @ X.T @ np.diag(1 / (sigma0_sq * (1 - data['display']) + sigma1_sq * data['display'])) @ y

    # Sample beta from its posterior distribution
    beta = np.random.multivariate_normal(mu_beta, inv_CoVar_beta)

    # Calculate residuals
    residuals = y - X @ beta

    # Update and sample sigma0_sq and sigma1_sq based on residuals
    sigma0_sq = 1 / np.random.gamma(N / 2, 2 / np.sum((1 - data['display']) * residuals ** 2))
    sigma1_sq = 1 / np.random.gamma(N / 2, 2 / np.sum(data['display'] * residuals ** 2))

    # Store the samples only every thin operation
    if i % nod == 0:
        beta_samples.append(beta)
        sigma0_sq_samples.append(sigma0_sq)
        sigma1_sq_samples.append(sigma1_sq)

# Implementing Burn-in
beta_samples = np.array(beta_samples)
sigma0_sq_samples = np.array(sigma0_sq_samples)
sigma1_sq_samples = np.array(sigma1_sq_samples)

beta_samples_burn = np.array(beta_samples[nob:])
sigma0_sq_samples_burn = np.array(sigma0_sq_samples[nob:])
sigma1_sq_samples_burn = np.array(sigma1_sq_samples[nob:])

# Assume beta_samples_burn, sigma0_sq_samples_burn, sigma1_sq_samples_burn are your post-burn-in samples
data_dict = {
    "beta": beta_samples_burn,
    "sigma0_sq": sigma0_sq_samples_burn,
    "sigma1_sq": sigma1_sq_samples_burn
}

# Convert them to ArviZ's data format
inference_data = az.convert_to_inference_data(data_dict)

# Compute R-hat
rhat_vals = az.rhat(inference_data)

# You can access individual R-hat values like this:
print("R-hat for beta:", rhat_vals["beta"])
print("R-hat for sigma0_sq:", rhat_vals["sigma0_sq"])
print("R-hat for sigma1_sq:", rhat_vals["sigma1_sq"])
'''
***************************************
Creating and Printing Results 
***************************************
'''
# Creating a dictionary with parameter names and the respective percentiles.
percentiles_data = {
    "Parameter": [],
    "10% percentile": [],
    "median": [],
    "90% percentile": []
}

parameters = ["β0", "β1", "β2", "β3", "σ^2_0", "σ^2_1"]
samples_list = [beta_samples_burn[:, i] for i in range(4)] + [sigma0_sq_samples_burn, sigma1_sq_samples_burn]

for param, samples in zip(parameters, samples_list):
    percentiles_data["Parameter"].append(param)
    percentiles_data["10% percentile"].append(np.percentile(samples, 10))
    percentiles_data["median"].append(np.median(samples))
    percentiles_data["90% percentile"].append(np.percentile(samples, 90))

# Convert the dictionary to a pandas DataFrame.
df = pd.DataFrame(percentiles_data)

# Compute the posterior mean of the sigma ratio
posterior_mean_ratio = np.mean(sigma0_sq_samples_burn / sigma1_sq_samples_burn)

# Print details and results using the DataFrame's to_string method.
print("\nDetails MCMC sampler:")
print(f"How many simulations in total did you do (including burn-in)? : {nos}")
print(f"How many burn-in simulations did you use? : {nob}")
print(f"What is your thin value?? : {nod}")
print("\n")
print("Posterior Results:")
print(df.to_string(index=False))
print("\n")
print(f"The posterior mean of the ratio σ^2_0/σ^2_1 is: {posterior_mean_ratio}")
'''
***************************************
The Code Below Generates several Plots 
***************************************
'''
# Trace plots for the first n_draws of all parameters
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle(f'Trace plot of first {trace_draws} draws of all parameters')
axes = axes.flatten()

# Plots for Beta parameters
for i in range(4):
    axes[i].plot(beta_samples[:trace_draws, i])
    axes[i].set_title(f'Beta {i}')

# Plot for Sigma0_sq
axes[4].plot(sigma0_sq_samples[:trace_draws])
axes[4].set_title('Sigma0_sq')

# Plot for Sigma1_sq
axes[5].plot(sigma1_sq_samples[:trace_draws])
axes[5].set_title('Sigma1_sq')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Trace plots all draws of all parameters after burn_in
n_draws = 1000
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle(f'Trace plot of all parameters after burn-in ({nob})')
axes = axes.flatten()

# Plots for Beta parameters
for i in range(4):
    axes[i].plot(beta_samples_burn[:, i])
    axes[i].set_title(f'Beta {i}')

# Plot for Sigma0_sq
axes[4].plot(sigma0_sq_samples_burn)
axes[4].set_title('Sigma0_sq')

# Plot for Sigma1_sq
axes[5].plot(sigma1_sq_samples_burn)
axes[5].set_title('Sigma1_sq')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Autocorrelation plots for each beta parameter
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()  # Flattening the axes object for easier indexing

# Autocorrelation plots for each beta parameter
for i in range(4):
    plot_acf(beta_samples_burn[:, i], lags=10, title=f"Autocorrelation of β{i}", ax=axes[i])

# Autocorrelation plot for sigma0_sq
plot_acf(sigma0_sq_samples_burn, lags=10, title="Autocorrelation of σ^2_0", ax=axes[4])

# Autocorrelation plot for sigma1_sq
plot_acf(sigma1_sq_samples_burn, lags=10, title="Autocorrelation of σ^2_1", ax=axes[5])

plt.tight_layout()
plt.show()

# Plot the histograms of the posterior of the parameters
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle('Histograms of Posterior Samples')
axes = axes.flatten()

# Plots for Beta parameters
for i in range(4):
    axes[i].hist(beta_samples_burn[:, i], bins=20, density=True, alpha=0.7, color='blue')
    axes[i].set_title(f'Histogram of Beta {i}')

# Plot for Sigma0_sq
axes[4].hist(sigma0_sq_samples_burn, bins=20, density=True, alpha=0.7, color='green')
axes[4].set_title('Histogram of Sigma0_sq')

# Plot for Sigma1_sq
axes[5].hist(sigma1_sq_samples_burn, bins=20, density=True, alpha=0.7, color='red')
axes[5].set_title('Histogram of Sigma1_sq')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()



