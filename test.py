import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

np.random.seed(0)  # set random seed

# Settings
nos = 1000  # Number of samples after burn-in
nob = 500  # Number of burn-in samples
total_samples = nos + nob  # Total number of samples including burn-in
trace_draws = 500

# Initialize parameters
sigma_0_sq = 1.0
sigma_1_sq = 1.0
beta = np.zeros(4)  # [beta_0, beta_1, beta_2, beta_3]

# Load and prepare data
data_folder_path = './data/'
dataset = 'brand48'

sales = pd.read_excel(data_folder_path + 'sales.xls', usecols=[dataset]).values
price = pd.read_excel(data_folder_path + 'price.xls', usecols=[dataset]).values
coupon = pd.read_excel(data_folder_path + 'coupon.xls', usecols=[dataset]).values
display = pd.read_excel(data_folder_path + 'displ.xls', usecols=[dataset]).values

logsales = np.log(sales)
logprice = np.log(price)

# Placeholder for samples
beta_samples = []
sigma0_sq_samples = []
sigma1_sq_samples = []

# Prepare the data
y = logsales.ravel()
X = np.column_stack((np.ones_like(logsales), display, coupon, logprice))
T, k = X.shape

# Main Gibbs sampling loop
for i in range(total_samples):
    # Update conditional for beta
    sigma_t_sq = sigma_0_sq * (1 - display) + sigma_1_sq * display
    inv_sigma_t_sq = 1 / sigma_t_sq.ravel()

    # Weighted least squares for beta
    X_wls = X * np.sqrt(inv_sigma_t_sq)[:, None]
    y_wls = y * np.sqrt(inv_sigma_t_sq)
    beta_var = np.linalg.inv(X_wls.T @ X_wls)
    beta_mean = beta_var @ (X_wls.T @ y_wls)
    beta = np.random.multivariate_normal(beta_mean, beta_var)

    # Update conditional for sigma_0_sq and sigma_1_sq
    residuals = y - X @ beta
    sse_0 = np.sum((residuals[display.ravel() == 0]) ** 2)
    sse_1 = np.sum((residuals[display.ravel() == 1]) ** 2)
    n_0 = np.sum(1 - display)
    n_1 = np.sum(display)

    # Inverse Gamma posterior parameters
    shape_0 = n_0 / 2
    scale_0 = sse_0 / 2
    shape_1 = n_1 / 2
    scale_1 = sse_1 / 2

    # Draw from Inverse Gamma
    sigma_0_sq = 1 / np.random.gamma(shape_0, 1 / scale_0)
    sigma_1_sq = 1 / np.random.gamma(shape_1, 1 / scale_1)

    # Store samples
    beta_samples.append(list(beta))
    sigma0_sq_samples.append(sigma_0_sq)
    sigma1_sq_samples.append(sigma_1_sq)

# Convert to NumPy arrays for convenience
beta_samples_all = np.array(beta_samples)
sigma0_sq_samples_all = np.array(sigma0_sq_samples)
sigma1_sq_samples_all = np.array(sigma1_sq_samples)

# Remove burn-in samples and apply thinning
beta_samples = beta_samples_all[nob:, :]
sigma0_sq_samples = sigma0_sq_samples_all[nob:]
sigma1_sq_samples = sigma1_sq_samples_all[nob:]
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
samples_list = [beta_samples_all[:, i] for i in range(4)] + [sigma0_sq_samples_all, sigma1_sq_samples_all]

for param, samples in zip(parameters, samples_list):
    percentiles_data["Parameter"].append(param)
    percentiles_data["10% percentile"].append(np.percentile(samples, 10))
    percentiles_data["median"].append(np.median(samples))
    percentiles_data["90% percentile"].append(np.percentile(samples, 90))

# Convert the dictionary to a pandas DataFrame.
df = pd.DataFrame(percentiles_data)

# Compute the posterior mean of the sigma ratio
posterior_mean_ratio = np.mean(sigma0_sq_samples_all / sigma1_sq_samples_all)

# Print details and results using the DataFrame's to_string method.
print("\nDetails MCMC sampler:")
print(f"How many simulations in total did you do (including burn-in)? : {nos}")
print(f"How many burn-in simulations did you use? : {nob}")
# print(f"What is your thin value?? : {nod}")
print("\n")
print("Posterior Results:")
print(df.to_string(index=False))
print("\n")
print(f"The posterior mean of the ratio σ^2_0/σ^2_1 is: {posterior_mean_ratio}")
'''
***************************************
Geweke test diagnostic for convergence 
***************************************
'''


def geweke_diagnostic(samples, first=0.1, last=0.5):
    n = len(samples)
    first_idx = int(n * first)
    last_idx = int(n * last)

    mean_first_segment = np.mean(samples[:first_idx])
    mean_last_segment = np.mean(samples[-last_idx:])

    var_first_segment = np.var(samples[:first_idx])
    var_last_segment = np.var(samples[-last_idx:])

    numerator = mean_first_segment - mean_last_segment
    denominator = np.sqrt(var_first_segment / first_idx + var_last_segment / last_idx)
    z_score = numerator / denominator

    return mean_first_segment, mean_last_segment, z_score


# Initialize list to hold table rows
table_rows = []
table_rows.append(['Parameter', 'Mean (First Segment)', 'Mean (Last Segment)', 'Z-score'])

# Beta samples
for i in range(4):
    samples = beta_samples[:, i]
    mean_first, mean_last, z_score = geweke_diagnostic(samples)
    table_rows.append([f'beta{i}', f"{mean_first:.4f}", f"{mean_last:.4f}", f"{z_score:.4f}"])

# Sigma samples
for param_name, samples in zip(["sigma0_sq", "sigma1_sq"], [sigma0_sq_samples, sigma1_sq_samples]):
    mean_first, mean_last, z_score = geweke_diagnostic(samples)
    table_rows.append([param_name, f"{mean_first:.4f}", f"{mean_last:.4f}", f"{z_score:.4f}"])

# Print the table
print(f"Geweke Diagnostic Results:")
for row in table_rows:
    print("{:<12} {:<20} {:<20} {:<10}".format(*row))
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
    axes[i].plot(beta_samples_all[:, i])
    axes[i].set_title(f'Beta {i}')

# Plot for Sigma0_sq
axes[4].plot(sigma0_sq_samples_all)
axes[4].set_title('Sigma0_sq')

# Plot for Sigma1_sq
axes[5].plot(sigma1_sq_samples_all)
axes[5].set_title('Sigma1_sq')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Autocorrelation plots for each beta parameter
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()  # Flattening the axes object for easier indexing

# Autocorrelation plots for each beta parameter
for i in range(4):
    plot_acf(beta_samples_all[:, i], lags=10, title=f"Autocorrelation of β{i}", ax=axes[i])

# Autocorrelation plot for sigma0_sq
plot_acf(sigma0_sq_samples_all, lags=10, title="Autocorrelation of σ^2_0", ax=axes[4])

# Autocorrelation plot for sigma1_sq
plot_acf(sigma1_sq_samples_all, lags=10, title="Autocorrelation of σ^2_1", ax=axes[5])

plt.tight_layout()
plt.show()

# Plot the histograms of the posterior of the parameters
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle('Histograms of Posterior Samples')
axes = axes.flatten()

# Plots for Beta parameters
for i in range(4):
    axes[i].hist(beta_samples_all[:, i], bins=20, density=True, alpha=0.7, color='blue')
    axes[i].set_title(f'Histogram of Beta {i}')

# Plot for Sigma0_sq
axes[4].hist(sigma0_sq_samples_all, bins=20, density=True, alpha=0.7, color='green')
axes[4].set_title('Histogram of Sigma0_sq')

# Plot for Sigma1_sq
axes[5].hist(sigma1_sq_samples_all, bins=20, density=True, alpha=0.7, color='red')
axes[5].set_title('Histogram of Sigma1_sq')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
