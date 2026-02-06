import numpy as np
from scipy.stats import norm

def run_diagnostic():
    # 1. Setup a redshift grid
    z = np.linspace(0, 2, 1000)
    
    # 2. Define two pixels with different widths (sigma)
    # sigma_1 is narrow (high SNR area), sigma_2 is broad (low SNR area)
    sigma_1 = 0.05
    sigma_2 = 0.15
    
    # Distributions
    n1 = norm.pdf(z, loc=0.6, scale=sigma_1)
    n2 = norm.pdf(z, loc=0.6, scale=sigma_2)
    
    # Global average distribution
    n_glob = 0.5 * (n1 + n2)
    
    # --- Calculation 1: Redshift-based Std Ratio (Physical Truth for sigma) ---
    # Global std sigma_all = sqrt( <sigma_i^2> ) for Gaussians with same mean
    sigma_all = np.sqrt(0.5 * (sigma_1**2 + sigma_2**2))
    # Ratio = sigma_all * <1/sigma_i>
    r_sigma = sigma_all * (0.5 * (1/sigma_1 + 1/sigma_2))
    
    # --- Calculation 2: Geometric Enhancement (Clustering Signal) ---
    # L2 = integral of n(z)^2
    l2_1 = np.trapz(n1**2, z)
    l2_2 = np.trapz(n2**2, z)
    l2_pix_mean = 0.5 * (l2_1 + l2_2)
    
    l2_glob = np.trapz(n_glob**2, z)
    r_geo = l2_pix_mean / l2_glob
    
    print(f"Toy Model: Two Gaussian pixels (sigma1={sigma_1}, sigma2={sigma_2})")
    print("-" * 50)
    print(f"Global sigma_all:            {sigma_all:.4f}")
    print(f"Redshift-based Std Ratio:    {r_sigma:.4f}")
    print(f"Geometric Enhancement (L2): {r_geo:.4f}")
    print("-" * 50)
    print(f"Difference (Ratio):         {r_sigma/r_geo:.3f}")
    
    print("\nObservation:")
    print("The Redshift-based Ratio is significantly higher than the L2 enhancement.")
    print("This is because the Global n(z) (mixture of narrow + broad) is 'peakier'")
    print("than a single Gaussian with width sigma_all. Its L2 remains high,")
    print("which makes the Geometric Ratio (local_L2 / global_L2) smaller.")

if __name__ == "__main__":
    run_diagnostic()
