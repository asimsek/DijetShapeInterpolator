#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numpy.core.getlimits")

import os
import argparse
import numpy as np
from scipy.stats import ks_2samp
from scipy.signal import find_peaks
from ROOT import TFile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def find_peak_region(bin_centers, sim_vals, expected_mass, side='right', n_sigma=1.5):
    """
    Estimate Gaussian peak region using the right side of the distribution,
    selecting the peak closest to expected_mass if multiple exist.
    """
    max_val = np.max(sim_vals)
    # Find local peaks with reasonable height and prominence thresholds to filter noise
    peaks, _ = find_peaks(sim_vals, height=0.01 * max_val, prominence=0.05 * max_val)
    
    if len(peaks) == 0:
        # Fallback to global max if no local peaks detected
        peak_idx = np.argmax(sim_vals)
    else:
        # Select the peak closest to the expected mass
        distances = np.abs(bin_centers[peaks] - expected_mass)
        peak_idx = peaks[np.argmin(distances)]
    
    peak_val = bin_centers[peak_idx]
    
    if side == 'right':
        mask = bin_centers > peak_val
    elif side == 'left':
        mask = bin_centers < peak_val
    else:
        raise ValueError("side must be 'right' or 'left'")
    
    weights = sim_vals[mask]
    values = bin_centers[mask]
    
    if len(values) == 0 or np.sum(weights) == 0:
        print("Warning: No data on selected side to estimate sigma. Using fixed window around peak.")
        sigma_est = (bin_centers[-1] - bin_centers[0]) / 10
    else:
        sigma_est = np.sqrt(np.average((values - peak_val)**2, weights=weights))
    
    left_edge = peak_val - n_sigma * sigma_est
    right_edge = peak_val + n_sigma * sigma_est
    peak_mask = (bin_centers >= left_edge) & (bin_centers <= right_edge)
    
    if not np.any(peak_mask):
        raise ValueError("Peak region selection returned an empty mask. Check the histogram content.")
    
    return peak_mask, peak_val, sigma_est


def main():
    parser = argparse.ArgumentParser(description='Plot histograms from ROOT files and their ratio')
    parser.add_argument('--root1', default='interpolatedSignalShapes/QstarTo2J_Run3Summer22MiniAODv4_ResonanceShapes_all.root',
                        help='Path to the first ROOT file (simulated)')
    parser.add_argument('--root2', default='interpolatedSignalShapes/QstarTo2J_Run3Summer22MiniAODv4_ResonanceShapes_sub.root',
                        help='Path to the second ROOT file (interpolated)')
    parser.add_argument('--mass', default='2000',
                        help='Mass value for histogram selection')
    parser.add_argument('--type', default='qg',
                        help='Type for histogram selection')
    parser.add_argument('--results_dir', default='.',
                        help='Directory to save the results')
    
    args = parser.parse_args()
    
    group_key = "QstarTo2J_Run3Summer22MiniAODv4"
    mass = args.mass
    histName1 = f'h_sim_{args.type}_{mass}'
    histName2 = f'h_{args.type}_{mass}'
    
    file1 = TFile.Open(args.root1)
    h_sim = file1.Get(histName1)
    file2 = TFile.Open(args.root2)
    h_interp = file2.Get(histName2)
    
    if not h_sim or not h_interp:
        print("Error: Could not retrieve histograms.")
        return
    
    nbins = h_sim.GetNbinsX()
    sim_contents = np.array([h_sim.GetBinContent(i) for i in range(1, nbins + 1)])
    interp_contents = np.array([h_interp.GetBinContent(i) for i in range(1, nbins + 1)])
    bin_centers = np.array([h_sim.GetBinCenter(i) for i in range(1, nbins + 1)])
    
    # Valid mask for KS test: remove NaNs and zero-only regions
    valid_mask = (
        ~np.isnan(sim_contents)
        & ~np.isnan(interp_contents)
        & (sim_contents > 0)
        & (interp_contents > 0)
    )
    
    sim_clean_full = sim_contents[valid_mask]
    interp_clean_full = interp_contents[valid_mask]
    bin_centers_clean = bin_centers[valid_mask]
    
    # KS test on full range
    ks_stat, ks_p = ks_2samp(sim_clean_full, interp_clean_full, method='asymp')
    
    # Determine peak region based on cleaned bin_centers and sim values
    expected_mass = float(mass)
    peak_mask, peak_val, sigma_est = find_peak_region(bin_centers_clean, sim_clean_full, expected_mass)
    
    # Apply same mask to cleaned arrays
    sim_peak = sim_clean_full[peak_mask]
    interp_peak = interp_clean_full[peak_mask]
    
    # KS test on peak region (subset of cleaned data)
    ks_peak_stat, ks_peak_p = ks_2samp(sim_peak, interp_peak, method='asymp')
    
    sample_group, year_tag = group_key.split('_')
    
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax = axs[0]
    ax.plot(bin_centers, sim_contents, drawstyle='steps-mid', color='blue', label='Simulated')
    ax.plot(bin_centers, interp_contents, '--', drawstyle='steps-mid', color='black', label='Interpolated')
    ax.set_ylabel('Normalized Entries', loc='top', fontsize=14)
    ax.tick_params(direction='in', which='both', top=True, right=True, labelsize=12)
    
    info_text = (
        f"{sample_group} -- {mass} GeV\n"
        f"{year_tag}\n"
        f"KS stat: {ks_stat:.4f}\n"
        f"p-value: {ks_p * 100:.2f}%\n"
        f"KS stat (peak): {ks_peak_stat:.4f}\n"
        f"p-value (peak): {ks_peak_p * 100:.2f}%"
    )
    ax.text(0.98, 0.76, info_text, transform=ax.transAxes,
            ha='right', va='top', fontsize=10, linespacing=2.0)
    
    # Peak region edges
    left_cut = peak_val - 1.5 * sigma_est
    right_cut = peak_val + 1.5 * sigma_est
    ax.axvline(left_cut, color='orange', linestyle='--', linewidth=1.5, label='Peak region edges')
    ax.axvline(right_cut, color='orange', linestyle='--', linewidth=1.5)
    ax.legend(loc='upper right', fontsize=10)
    
    ax_ratio = axs[1]
    ratio = np.divide(interp_contents - sim_contents, sim_contents,
                      where=sim_contents != 0, out=np.zeros_like(interp_contents))
    ax_ratio.plot(bin_centers, ratio, drawstyle='steps-mid', color='black')
    ax_ratio.set_ylabel('(Interp - Sim) / Sim', fontsize=14)
    ax_ratio.set_xlabel('m_jj [GeV]', loc='right', fontsize=14)
    ax_ratio.tick_params(direction='in', which='both', top=True, right=True, labelsize=12)
    ax_ratio.set_ylim(-2.0, 2.0)
    ax_ratio.axhline(0, color='red', linestyle='--')
    
    plt.tight_layout()
    
    os.makedirs(args.results_dir, exist_ok=True)
    pdf_path = os.path.join(args.results_dir, f"{group_key}mass{mass}_comparison.pdf")
    plt.savefig(pdf_path)
    plt.close()
    
    print(f"Saved plot to {pdf_path}")


if __name__ == '__main__':
    main()
