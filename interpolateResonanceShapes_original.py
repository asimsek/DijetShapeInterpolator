#!/usr/bin/env python3
import os
import re
import sys
import uproot
import numpy as np
import bisect
from array import array
from ROOT import (
    Math, TFile, TH1D, RooRealVar, RooArgList, RooArgSet, RooDataHist, RooHistPdf,
    RooMomentMorph, RooBinning, RooFit, TVectorD, gErrorIgnoreLevel, kWarning,
    RooMsgService, RooIntegralMorph, RooKeysPdf, RooDataSet, RooNDKeysPdf
)
from argparse import ArgumentParser
from prettytable import PrettyTable
gErrorIgnoreLevel = kWarning
RooMsgService.instance().setGlobalKillBelow(RooFit.WARNING)
REDUCED_MIN = 0.0
REDUCED_MAX = 2.0
REDUCED_NBINS = 20000 # Fine resolution (~0.0001 per bin) for accurate integration/morphing
def moment_morph_pdf(shapes_abs, target_mass, output_histo):
    # Get sorted masses and compute integrals
    sorted_masses = sorted(shapes_abs.keys())
    min_mass = min(sorted_masses)
    max_mass = max(sorted_masses)
    integrals = {m: shapes_abs[m].numEntries() for m in sorted_masses}
    lower_mass = None
    higher_mass = None
    if target_mass in shapes_abs:
        if integrals[target_mass] > 0:
            lower_mass = target_mass
            higher_mass = target_mass
        else:
            print(f"Warning: Input shape for mass {target_mass} has zero integral. Interpolating instead.")
    # Find valid lower mass with positive integral
    if lower_mass is None:
        for m in reversed([mm for mm in sorted_masses if mm < target_mass]):
            if integrals[m] > 0:
                lower_mass = m
                break
        if lower_mass is None:
            for m in [mm for mm in sorted_masses if mm > target_mass]:
                if integrals[m] > 0:
                    lower_mass = m
                    break
    # Find valid higher mass with positive integral
    if higher_mass is None:
        for m in [mm for mm in sorted_masses if mm > target_mass]:
            if integrals[m] > 0:
                higher_mass = m
                break
        if higher_mass is None:
            higher_mass = lower_mass
    if lower_mass is None or higher_mass is None:
        raise ValueError(f"No valid shapes with positive integral for interpolation at {target_mass}")
    # Issue extrapolation warnings if necessary
    if target_mass < min_mass:
        print("** WARNING: ** Attempting to extrapolate below the lowest input mass. The extrapolated shape(s) might not be reliable.")
    elif target_mass > max_mass:
        print("** WARNING: ** Attempting to extrapolate above the highest input mass. The extrapolated shape(s) might not be reliable.")
    if lower_mass == higher_mass:
        if target_mass not in shapes_abs or integrals.get(target_mass, 0) == 0:
            print(f"Warning: Using shape from mass {lower_mass} for target {target_mass} due to lack of bracketing shapes with positive integral.")
    else:
        print(f"\033[1;31mInterpolating target mass: {target_mass} GeV\033[0;0m - using {lower_mass} GeV and {higher_mass} GeV simulated samples ")
    # Create the reduced observable
    x = RooRealVar("xred", "xred", REDUCED_MIN, REDUCED_MAX)
    x.setBins(REDUCED_NBINS)
    x.setBins(40000, "cache") # Higher cache for better integration accuracy
    # Prepare smooth PDF for lower mass (nearest valid ≤ target) using RooKeysPdf; finds peak height to tune smoothing
    ds_lower = shapes_abs[lower_mass]
    h_lower_name = f"h_lower_temp_{lower_mass}"
    h_lower = TH1D(h_lower_name, h_lower_name, 2000, REDUCED_MIN, REDUCED_MAX)
    h_lower.SetDirectory(0)
    ds_lower.fillHistogram(h_lower, RooArgList(x))
    integral_lower = h_lower.Integral()
    if integral_lower > 0:
        h_lower.Scale(1.0 / integral_lower)
    peak_height_lower = h_lower.GetMaximum()
    del h_lower # Clean up
    n_entries_lower = ds_lower.numEntries()
    #Calculate adaptive rho for RooKeysPdf bandwidth: balance between peak and tail
    rho_lower = max(1.5, 0.00001 / peak_height_lower) * (1.0 / max(1, n_entries_lower)) ** 0.1
    pdf_lower = RooKeysPdf("pdf_lower", "pdf_lower", x, ds_lower, RooKeysPdf.MirrorAsymBoth, rho_lower)
    # Prepare smooth PDF for higher mass (nearest valid ≥ target), if different from lower
    pdf_higher = pdf_lower
    if lower_mass != higher_mass:
        ds_higher = shapes_abs[higher_mass]
        h_higher_name = f"h_higher_temp_{higher_mass}"
        h_higher = TH1D(h_higher_name, h_higher_name, 2000, REDUCED_MIN, REDUCED_MAX)
        h_higher.SetDirectory(0)
        ds_higher.fillHistogram(h_higher, RooArgList(x))
        integral_higher = h_higher.Integral()
        if integral_higher > 0:
            h_higher.Scale(1.0 / integral_higher)
        peak_height_higher = h_higher.GetMaximum()
        del h_higher # Clean up
        n_entries_higher = ds_higher.numEntries()
        #Calculate adaptive rho for RooKeysPdf bandwidth: balance between peak and tail
        rho_higher = max(1.5, 0.00001 / peak_height_higher) * (1.0 / max(1, n_entries_higher)) ** 0.1
        pdf_higher = RooKeysPdf("pdf_higher", "pdf_higher", x, ds_higher, RooKeysPdf.MirrorAsymBoth, rho_higher)
    # Create the morphing parameter and morph if necessary
    m_var = RooRealVar("m_var", "m_var", target_mass)
    if lower_mass == higher_mass:
        morph = pdf_lower
    else:
        pdfList = RooArgList(pdf_lower, pdf_higher)
        mrefpoints = TVectorD(2)
        mrefpoints[0] = lower_mass
        mrefpoints[1] = higher_mass
        morph = RooMomentMorph("morph", "morph", m_var, RooArgList(x), pdfList, mrefpoints, RooMomentMorph.NonLinearPosFractions)
    # Fill output histogram by integrating morphed PDF
    for i in range(1, output_histo.GetNbinsX() + 1):
        low_abs = output_histo.GetBinLowEdge(i)
        high_abs = low_abs + output_histo.GetBinWidth(i)
        width_abs = high_abs - low_abs
        low_red = low_abs / target_mass
        high_red = high_abs / target_mass
        if high_red <= REDUCED_MIN or low_red >= REDUCED_MAX:
            output_histo.SetBinContent(i, 0.)
        else:
            l = max(low_red, REDUCED_MIN)
            h = min(high_red, REDUCED_MAX)
            if l < h:
                x.setRange("intRange", l, h)
                integral = morph.createIntegral(RooArgSet(x), RooFit.Range("intRange")).getVal()
                content = integral / width_abs if width_abs > 0 else 0.
                output_histo.SetBinContent(i, content if content >= 0 else 0.)
            else:
                output_histo.SetBinContent(i, 0.)
        output_histo.SetBinError(i, 0.)
    # Normalize the histogram
    integ = output_histo.Integral("width")
    if integ > 0.0:
        output_histo.Scale(1.0 / integ)
    # Clean up objects
    del morph
    del m_var
    del pdf_lower
    if lower_mass != higher_mass:
        del pdf_higher
def rebin_to_variable(fine_hist, var_bin_array, name, title):
    var_nbins = len(var_bin_array) - 1
    h_var = TH1D(name, title, var_nbins, var_bin_array)
    # Rebin by integrating over fine bins and setting density
    for j in range(1, var_nbins + 1):
        low = h_var.GetBinLowEdge(j)
        up = h_var.GetBinLowEdge(j) + h_var.GetBinWidth(j)
        bin1 = fine_hist.FindBin(low + 1e-5)
        bin2 = fine_hist.FindBin(up - 1e-5)
        prob = fine_hist.Integral(bin1, bin2)
        width = up - low
        density = prob / width if width > 0 else 0
        h_var.SetBinContent(j, density)
        h_var.SetBinError(j, 0)
    return h_var
def main():
    # Usage examples
    usage = "Examples:\n"
    usage += "For base directory: python3 interpolateResonanceShapes_original.py -b /eos/cms/store/user/dagyel/DiJet/rootNTuples_reduced/signalSamples -o interpolatedSignalShapes/ResonanceShapes.root --interval 100 --fineBinning\n"
    usage += "For list file: python3 interpolateResonanceShapes_original.py -l inputSamples/inputSamples.txt -o interpolatedSignalShapes/ResonanceShapes.root --interval 100 --fineBinning"
    # Parse command-line arguments
    parser = ArgumentParser(description='Combined script for extracting and interpolating resonance shapes', epilog=usage)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-b", "--base_dir", dest="base_dir", help="Base directory containing signal samples", metavar="BASE_DIR")
    input_group.add_argument("-l", "--list_file", dest="list_file", help="Text file containing categorized list of root files", metavar="LIST_FILE")
    parser.add_argument("-o", "--output_file", dest="output_file", required=True, help="Output file path (directory will be used for multiple outputs)", metavar="OUTPUT_FILE")
    parser.add_argument("-f", "--final_state", dest="final_state", required=False, help="Final state (e.g. qq, qg, gg). If not provided, inferred from model.", metavar="FINAL_STATE")
    parser.add_argument("--hist-name", dest="hist_name", default="h_mjj_HLTpass_noTrig", help="Name of the histogram to extract (default: %(default)s)", metavar="HIST_NAME")
    parser.add_argument("--interval", type=int, default=100, help="Step size for mass interpolation (default: %(default)s)", metavar="STEP")
    parser.add_argument("--fineBinning", dest="fineBinning", default=False, action="store_true", help="Use fine, 1-GeV binning")
    parser.add_argument("--storePDF", dest="storePDF", default=False, action="store_true", help="Also store a 1-GeV-binned PDF")
    parser.add_argument("--storeCDF", dest="storeCDF", default=False, action="store_true", help="Also store a 1-GeV-binned CDF")
    parser.add_argument("--dry-run", dest="dry_run", default=False, action="store_true", help="Perform a dry run to test configuration without processing")
    args = parser.parse_args()
    groups = {}
    out_dir = os.path.dirname(args.output_file)
    if not out_dir:
        out_dir = '.'
    base_name = os.path.basename(args.output_file)
    os.makedirs(out_dir, exist_ok=True)
    # Process input from base directory
    if args.base_dir:
        pattern = re.compile(r"^(?P<model>\w+)(_(?P<param>[\w\-\.]+))?_M[-_](?P<mass>\d+)_.*_(?P<run>Run3\S+)")
        for root, dirs, files in os.walk(args.base_dir):
            for file in files:
                if file.endswith('_reduced_skim.root'):
                    full_path = os.path.join(root, file)
                    subfolder = os.path.basename(root)
                    match = pattern.match(subfolder)
                    if match:
                        d = match.groupdict()
                        param = d['param'] if d['param'] else ''
                        key = f"{d['model']}_{param}_{d['run']}".strip('_')
                        mass = int(d['mass'])
                        if key not in groups:
                            groups[key] = {}
                        if mass in groups[key]:
                            print(f"Warning: Duplicate mass {mass} for group {key}, overwriting with {full_path}")
                        groups[key][mass] = full_path
                    else:
                        print(f"Cannot parse subfolder: {subfolder}")
    # Process input from list file
    elif args.list_file:
        current_group = None
        with open(args.list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.endswith(':'):
                    current_group = line[:-1].strip()
                    if current_group not in groups:
                        groups[current_group] = {}
                elif ':' in line and current_group is not None:
                    mass_str, path = line.split(':', 1)
                    mass = int(mass_str.strip())
                    path = path.strip()
                    if mass in groups[current_group]:
                        print(f"Warning: Duplicate mass {mass} for group {current_group}, overwriting with {path}")
                    groups[current_group][mass] = path
                else:
                    print(f"Invalid line in list file: {line}")
    # Display found groups
    print("\033[1;31mFound groups:\033[0;0m")
    table = PrettyTable(['Sample Types', 'Simulated Mass Points'])
    table.align['Sample Types'] = 'l' # Left align first column
    table.align['Simulated Mass Points'] = 'r' # Right align second column
    for key in sorted(groups.keys()):
        table.add_row([key, ', '.join(map(str, sorted(groups[key].keys())))])
    print(table)
    print("\n")
    bin_boundaries = [1, 3, 6, 10, 16, 23, 31, 40, 50, 61, 74, 88, 103, 119, 137, 156, 176, 197, 220, 244, 270, 296, 325,
                      354, 386, 419, 453, 489, 526, 565, 606, 649, 693, 740, 788, 838, 890, 944, 1000, 1058, 1118, 1181, 1246, 1313, 1383, 1455, 1530, 1607, 1687,
                      1770, 1856, 1945, 2037, 2132, 2231, 2332, 2438, 2546, 2659, 2775, 2895, 3019, 3147, 3279, 3416, 3558, 3704, 3854, 4010, 4171, 4337, 4509,
                      4686, 4869, 5058, 5253, 5455, 5663, 5877, 6099, 6328, 6564, 6808, 7060, 7320, 7589, 7866, 8152, 8447, 8752, 9067, 9391, 9726, 10072, 10430,
                      10798, 11179, 11571, 11977, 12395, 12827, 13272, 13732, 14000]
    bin_boundaries = np.array(bin_boundaries, dtype=float)
    var_bin_array = array('d', bin_boundaries)
    var_nbins = len(bin_boundaries) - 1
    x = RooRealVar("xred", "xred", REDUCED_MIN, REDUCED_MAX)
    for group_key, mass_files in groups.items():
        if len(mass_files) < 2:
            print(f"Skipping group {group_key} with fewer than 2 masses.")
            continue
        # Infer final state from group key
        if 'GG' in group_key.upper():
            final_state = 'gg'
        elif 'QQ' in group_key.upper():
            final_state = 'qq'
        elif 'QSTAR' in group_key.upper():
            final_state = 'qg'
        else:
            print(f"Cannot infer final_state for {group_key}, skipping.")
            continue
        shapes = {}
        shapes_abs = {}
        sim_hists = {}
        output_path = os.path.join(out_dir, f"{group_key}_{base_name}")
        if args.fineBinning:
            input_min = 0
            input_max = 14000
            input_nbins = 14000
        else:
            input_min = bin_boundaries[0]
            input_max = bin_boundaries[-1]
            input_nbins = var_nbins
        # Process each mass file: apply cuts and create RooDataSet
        for mass, path in sorted(mass_files.items()):
            with uproot.open(path) as uf:
                utree = uf['rootTupleTree/tree']
                branches = ['mjj', 'deltaETAjj', 'etaWJ_j1', 'etaWJ_j2', 'pTWJ_j1', 'pTWJ_j2', 'IdTight_j1', 'IdTight_j2']
                data = utree.arrays(branches, library="np")
            mask = (np.abs(data['deltaETAjj']) < 1.1) & (np.abs(data['etaWJ_j1']) < 2.5) & (np.abs(data['etaWJ_j2']) < 2.5) & \
                   (data['IdTight_j1'] != 0) & (data['IdTight_j2'] != 0) & (data['pTWJ_j1'] > 60) & (data['pTWJ_j2'] > 30)
            mjj_filtered = data['mjj'][mask]
            xred = mjj_filtered / float(mass)
            mask_range = (xred >= REDUCED_MIN) & (xred <= REDUCED_MAX)
            mjj_abs = mjj_filtered[mask_range]
            xred_filtered = xred[mask_range]
            ds_reduced = RooDataSet(f"ds_reduced_{mass}", f"ds_reduced_{mass}", RooArgSet(x))
            for val in xred_filtered:
                x.setVal(val)
                ds_reduced.add(RooArgSet(x))
            if ds_reduced.numEntries() > 0:
                shapes_abs[mass] = ds_reduced
                # Create and fill the simulated histogram
                histname_sim = f"h_sim_{final_state}_{int(mass)}"
                h_sim_title = final_state + f" Simulated Resonance Shape M{mass}"
                if args.fineBinning:
                    h_sim_temp = TH1D(histname_sim + '_temp', h_sim_title, 14000, 0, 14000)
                    h_sim_temp.SetDirectory(0)
                    for val in mjj_abs:
                        h_sim_temp.Fill(val)
                    integ = h_sim_temp.Integral("width")
                    if integ > 0:
                        h_sim_temp.Scale(1.0 / integ)
                    h_sim = rebin_to_variable(h_sim_temp, var_bin_array, histname_sim, h_sim_title)
                    del h_sim_temp
                else:
                    h_sim = TH1D(histname_sim, h_sim_title, var_nbins, var_bin_array)
                    h_sim.SetDirectory(0)
                    for val in mjj_abs:
                        h_sim.Fill(val)
                    integ = h_sim.Integral("width")
                    if integ > 0:
                        h_sim.Scale(1.0 / integ)
                h_sim.SetXTitle("Dijet Mass [GeV]")
                h_sim.SetYTitle("Probability")
                sim_hists[mass] = h_sim
            else:
                print(f"Warning: No entries for mass {mass}")
        print(f"Producing output for group \033[1;31m{group_key}\033[0;0m in \033[1;31m{output_path}\033[0;0m")
        min_mass = min(mass_files.keys())
        max_mass = max(mass_files.keys())
        masses = list(range(min_mass, max_mass + 1, args.interval))
        if masses[-1] != max_mass:
            masses.append(max_mass)
        masses.sort()
        out_file = TFile.Open(output_path, 'recreate')
        # Write simulated histograms
        out_file.cd()
        for mass in sorted(sim_hists.keys()):
            sim_hists[mass].Write()
        # Interpolate shapes for each target mass
        for mass in masses:
            print(f"Producing {final_state} shape for m = {int(mass)} GeV")
            histname = f"h_{final_state}_{int(mass)}"
            h_shape_name = histname + '_temp'
            if args.fineBinning:
                h_shape = TH1D(h_shape_name, final_state + " Resonance Shape", 14000, 0, 14000)
            else:
                h_shape = TH1D(h_shape_name, final_state + " Resonance Shape", var_nbins, var_bin_array)
            h_shape.SetXTitle("Dijet Mass [GeV]")
            h_shape.SetYTitle("Probability")
            moment_morph_pdf(shapes_abs, mass, h_shape)
            if args.fineBinning:
                h_shape_rebinned = rebin_to_variable(h_shape, var_bin_array, histname, final_state + " Resonance Shape")
                out_file.cd()
                h_shape_rebinned.Write()
            else:
                out_file.cd()
                h_shape.Write()
            # Optionally store PDF and CDF histograms
            if args.storePDF or args.storeCDF:
                h_pdf = TH1D(histname + "_pdf", final_state + " Resonance Shape PDF", 14000, 0, 14000)
                h_cdf = TH1D(histname + "_cdf", final_state + " Resonance Shape CDF", 14000, 0, 14000)
                for i in range(1, h_shape.GetNbinsX() + 1):
                    bin_min = h_pdf.GetXaxis().FindBin(h_shape.GetXaxis().GetBinLowEdge(i) + 0.5)
                    bin_max = h_pdf.GetXaxis().FindBin(h_shape.GetXaxis().GetBinLowEdge(i) + h_shape.GetXaxis().GetBinWidth(i) - 0.5)
                    bin_content = h_shape.GetBinContent(i) / float(bin_max - bin_min + 1)
                    for b in range(bin_min, bin_max + 1):
                        h_pdf.SetBinContent(b, bin_content)
                for i in range(1, h_cdf.GetNbinsX() + 1):
                    bin_min = h_pdf.GetXaxis().FindBin(h_cdf.GetXaxis().GetBinLowEdge(i) + 0.5)
                    bin_max = h_pdf.GetXaxis().FindBin(h_cdf.GetXaxis().GetBinLowEdge(i) + h_cdf.GetXaxis().GetBinWidth(i) - 0.5)
                    curr = 0.
                    for b in range(bin_min, bin_max + 1):
                        curr = curr + h_pdf.GetBinContent(b)
                    prev = h_cdf.GetBinContent(i - 1)
                    h_cdf.SetBinContent(i, prev + curr)
                if args.storePDF:
                    h_pdf.Write()
                if args.storeCDF:
                    h_cdf.Write()
        out_file.Close()
if __name__ == '__main__':
    main()
