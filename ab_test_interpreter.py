import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import matplotlib.patches as mpatches

st.title('AB Test Result Interpreter')

visitors_A_observed = st.number_input('Enter the number of visitors for version A', value=1803)
conversions_A_observed = st.number_input('Enter the number of conversions for version A', value=5)
visitors_B_observed = st.number_input('Enter the number of visitors for version B', value=2116)
conversions_B_observed = st.number_input('Enter the number of conversions for version B', value=7)

if st.button('Compute'):
    # AB test version A (moonbird)
    visitors_A_observed = 1803
    conversions_A_observed = 5
    # AB test version B (Amaury)
    visitors_B_observed = 2116
    conversions_B_observed = 7

    # Prior parameters (assuming uniform prior)
    alpha_prior = 1
    beta_prior = 1

    # Update the priors with observed data
    alpha_posterior_A = alpha_prior + conversions_A_observed
    beta_posterior_A = beta_prior + visitors_A_observed - conversions_A_observed
    alpha_posterior_B = alpha_prior + conversions_B_observed
    beta_posterior_B = beta_prior + visitors_B_observed - conversions_B_observed

    # Calculate the posterior distributions
    x = np.linspace(0, 0.05, 1000)
    posterior_A = beta.pdf(x, alpha_posterior_A, beta_posterior_A)
    posterior_B = beta.pdf(x, alpha_posterior_B, beta_posterior_B)


    # Styling
    plt.style.use('seaborn-v0_8')
    fig, _ = plt.subplots(figsize=(10, 10))
    # Find the lower and upper bounds of the x-axis to cover 99% of both posteriors
    lower_bound_A = beta.ppf(0.001, alpha_posterior_A, beta_posterior_A)
    upper_bound_A = beta.ppf(0.999, alpha_posterior_A, beta_posterior_A)
    lower_bound_B = beta.ppf(0.001, alpha_posterior_B, beta_posterior_B)
    upper_bound_B = beta.ppf(0.999, alpha_posterior_B, beta_posterior_B)


    # Subplot 1: Posterior for A
    plt.subplot(311)
    plt.plot(x, posterior_A, label='Posterior A', color='blue', linestyle='-', linewidth=2)
    plt.xlabel('Conversion rate (%)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Posterior A', fontsize=14)
    plt.xlim(min(lower_bound_A, lower_bound_B), max(upper_bound_A, upper_bound_B))
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))

    # Annotate posterior mean and 95% CI for A
    mean_A = beta.mean(alpha_posterior_A, beta_posterior_A)
    ci_A = beta.interval(0.95, alpha_posterior_A, beta_posterior_A)
    plt.axvline(mean_A, color='red', linestyle='--', linewidth=1)
    plt.annotate(f'Mean: {mean_A:.2%}', xy=(mean_A, 0), xycoords=('data', 'axes fraction'),
                xytext=(-50, 20), textcoords='offset points', fontsize=10,
                arrowprops=dict(facecolor='red', edgecolor='red', alpha=0.6, arrowstyle='->', linewidth=1.5))
    plt.fill_between(x, 0, posterior_A, where=(x >= ci_A[0]) & (x <= ci_A[1]), color='blue', alpha=0.3)


    # Subplot 2: Posterior for B
    plt.subplot(312)
    plt.plot(x, posterior_B, label='Posterior B', color='orange', linestyle='-', linewidth=2)
    plt.xlabel('Conversion rate (%)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Posterior B', fontsize=14)
    plt.xlim(min(lower_bound_A, lower_bound_B), max(upper_bound_A, upper_bound_B))
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))

    # Annotate posterior mean and 95% CI for B
    mean_B = beta.mean(alpha_posterior_B, beta_posterior_B)
    ci_B = beta.interval(0.95, alpha_posterior_B, beta_posterior_B)
    plt.axvline(mean_B, color='red', linestyle='--', linewidth=1)
    plt.annotate(f'Mean: {mean_B:.2%}', xy=(mean_B, 0), xycoords=('data', 'axes fraction'),
                xytext=(10, 20), textcoords='offset points', fontsize=10,
                arrowprops=dict(facecolor='red', edgecolor='red', alpha=0.6, arrowstyle='->', linewidth=1.5))
    plt.fill_between(x, 0, posterior_B, where=(x >= ci_B[0]) & (x <= ci_B[1]), color='orange', alpha=0.3)


    # Subplot 3: Distribution for difference between A and B
    n_samples = 100000
    samples_A = beta.rvs(alpha_posterior_A, beta_posterior_A, size=n_samples)
    samples_B = beta.rvs(alpha_posterior_B, beta_posterior_B, size=n_samples)
    plt.subplot(313)
    diff_samples = samples_B - samples_A
    plt.hist(diff_samples, bins=100, density=True, color='purple', alpha=0.6, label='Difference (B - A)')
    plt.xlabel('Difference in conversion rate (%)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Improvement Distribution', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))

    # Annotate posterior mean and 95% CI for difference
    mean_diff = np.mean(diff_samples)
    ci_diff = np.percentile(diff_samples, [2.5, 97.5])
    plt.axvline(mean_diff, color='red', linestyle='--', linewidth=1, label='Mean')
    plt.annotate(f'Mean: {mean_diff:.4%}', xy=(mean_diff, 0), xycoords=('data', 'axes fraction'),
                xytext=(10, 20), textcoords='offset points', fontsize=10,
                arrowprops=dict(facecolor='red', edgecolor='red', alpha=0.6, arrowstyle='->', linewidth=1.5))
    plt.fill_betweenx(y=[0, plt.gca().get_ylim()[1]], x1=ci_diff[0], x2=ci_diff[1], color='purple', alpha=0.3)

    # Add legends for CI's of each subplot
    ax1, ax2, ax3 = plt.gcf().axes
    ci_patch_A = mpatches.Patch(color='blue', alpha=0.3, label='95% CI')
    ax1.legend(handles=[ci_patch_A], fontsize=10)
    ci_patch_B = mpatches.Patch(color='orange', alpha=0.3, label='95% CI')
    ax2.legend(handles=[ci_patch_B], fontsize=10)
    ci_patch_diff = mpatches.Patch(color='purple', alpha=0.3, label='95% CI')
    ax3.legend(handles=[ci_patch_diff], fontsize=10)


    plt.tight_layout()
    # plt.show()
    st.pyplot(fig)

    p_B_better_than_A = np.mean(samples_B > samples_A)

    # Formatting the probabilities as strings
    p_B_better_than_A_str = f"Probability that B is better than A: {p_B_better_than_A:.2%}"
    p_B_worse_than_A_str = f"Probability that B is worse than A: {1 - p_B_better_than_A:.2%}"

    # Using st.write() to display the formatted probabilities
    st.write(p_B_better_than_A_str)
    st.write(p_B_worse_than_A_str)

