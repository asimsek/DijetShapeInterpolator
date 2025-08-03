#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numpy.core.getlimits")

import os
import argparse
import numpy as np
from scipy.stats import ks_2samp
from ROOT import TFile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Plot histograms from ROOT files and their ratio')
    parser.add_argument('--root1', default='interpolatedSignalShapes/QstarTo2J_Run3Summer22MiniAODv4_ResonanceShapes_all.root', help='First ROOT file')
    parser.add_argument('--root2', default='interpolatedSignalShapes/QstarTo2J_Run3Summer22MiniAODv4_ResonanceShapes_sub.root', help='Second ROOT file')
    parser.add_argument('--mass', default='2000', help='Histogram name from first file (simulated)')
    parser.add_argument('--type', default='qg', help='Histogram name from second file (interpolated)')
    parser.add_argument('--results_dir', default='.', help='Directory to save the plot')
    args = parser.parse_args()

    # Define group_key and mass based on hist names
    group_key = "QstarTo2J_Run3Summer22MiniAODv4"
    mass = args.mass
    #histName1 = f'h_sim_{group_key}_M{mass}'
    histName1 = f'h_sim_{args.type}_{mass}'
    histName2 = f'h_{args.type}_{mass}'

    print (histName1)
    print (histName2)
    # Open ROOT files and get histograms
    file1 = TFile.Open(args.root1)
    h_sim = file1.Get(histName1)
    file2 = TFile.Open(args.root2)
    h_interp = file2.Get(histName2)

    if not h_sim or not h_interp:
        print("Error: Could not retrieve histograms.")
        return

    # Extract contents and centers (assuming same binning)
    nbins = h_sim.GetNbinsX()
    sim_contents = np.array([h_sim.GetBinContent(i) for i in range(1, nbins + 1)])
    interp_contents = np.array([h_interp.GetBinContent(i) for i in range(1, nbins + 1)])
    fine_centers = np.array([h_sim.GetBinCenter(i) for i in range(1, nbins + 1)])

    # Compute KS test
    ks_stat, ks_p = ks_2samp(sim_contents, interp_contents)

    # Split group_key for sample group and year tag
    sample_group, year_tag = group_key.split('_')

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax = axs[0]
    ax.plot(fine_centers, sim_contents, drawstyle='steps-mid', color='blue', label='Simulated')
    ax.plot(fine_centers, interp_contents, '--', drawstyle='steps-mid', color='black', label='Interpolated')
    ax.set_ylabel('Normalized Entries', loc='top', fontsize=14)
    ax.legend(loc='upper right')
    ax.tick_params(direction='in', which='both', top=True, right=True, labelsize=12)

    # Add info text under the legend
    info_text = f"{sample_group} -- {mass} GeV\n{year_tag}\nKS stat: {ks_stat:.4f}\np-value: {ks_p*100.:.2f}%"
    ax.text(0.98, 0.80, info_text, transform=ax.transAxes, ha='right', va='top', fontsize=10, bbox=None, linespacing=2.0)

    # Ratio as (Interpolated - Simulated) / Simulated
    ax_ratio = axs[1]
    ratio = np.divide(interp_contents - sim_contents, sim_contents, where=sim_contents != 0, out=np.zeros_like(interp_contents))
    ax_ratio.plot(fine_centers, ratio, drawstyle='steps-mid', color='black')
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
