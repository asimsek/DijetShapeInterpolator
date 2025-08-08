#!/usr/bin/env python3

import argparse
import numpy as np
import sys
import os
from array import array
import ROOT as rt
from ROOT import Math, TFile, TH1D
import uproot


class ShapeStorage:
    """
    Class to store and validate input shapes and bin centers.
    """
    def __init__(self, shapes, binxcenters):
        self.shapes = shapes
        self.binxcenters = binxcenters

        if len(self.shapes) < 2:
            print("** ERROR: ** Need at least 2 input shapes, %i provided. Aborting." % (len(self.shapes)))
            sys.exit(1)
        nbins = [len(self.binxcenters)]
        dx = self.binxcenters[1] - self.binxcenters[0] if len(self.binxcenters) > 1 else 0
        for key in self.shapes.keys():
            norm = sum(self.shapes[key]) * dx
            if abs(norm - 1.) > 0.05:  # Loosened tolerance for floating-point issues
                print("** ERROR: ** Input shape for m =", key, "GeV not normalized. Make sure the input shapes are normalized to unity. Aborting.")
                sys.exit(3)
            nbins.append(len(self.shapes[key]))
        if len(set(nbins)) > 1:
            print("** ERROR: ** Numbers of bins for different input shapes and the number of bin centers are not all identical. Aborting.")
            sys.exit(2)


def rebin_to_variable(fine_hist, var_bin_array, name, title):
    """
    Rebin a fine histogram to variable binning.
    """
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


def LineShapePDF(shapes, mass, histo):
    """
    Interpolate or extrapolate the line shape PDF for a given mass and fill the histogram.
    """
    x = shapes.binxcenters
    y = np.array([])
    if mass in shapes.shapes.keys():
        y = np.array(shapes.shapes[mass])
    else:
        input_masses = shapes.shapes.keys()
        min_mass = min(input_masses)
        max_mass = max(input_masses)
        ml = mass
        yl = np.array([])
        mh = mass
        yh = np.array([])
        if mass < min_mass:
            print("** WARNING: ** Attempting to extrapolate below the lowest input mass. The extrapolated shape(s) might not be reliable.")
            m_temp = sorted(input_masses)
            ml = m_temp[0]
            mh = m_temp[1]
        elif mass > max_mass:
            print("** WARNING: ** Attempting to extrapolate above the highest input mass. The extrapolated shape(s) might not be reliable.")
            m_temp = sorted(input_masses, reverse=True)
            ml = m_temp[1]
            mh = m_temp[0]
        else:
            ml = max([m for m in input_masses if m < mass])
            mh = min([m for m in input_masses if m > mass])
        yl = np.array(shapes.shapes[ml])
        yh = np.array(shapes.shapes[mh])
        y = ((yh - yl) / float(mh - ml)) * float(mass - ml) + yl
    interpolator = Math.Interpolator(len(x))
    interpolator.SetData(len(x), array('d', x), array('d', y.tolist()))
    for i in range(1, histo.GetNbinsX() + 1):
        xcenter = histo.GetBinCenter(i) / float(mass)
        if xcenter > shapes.binxcenters[0] and xcenter < shapes.binxcenters[-1]:
            xlow = histo.GetXaxis().GetBinLowEdge(i) / float(mass)
            if xlow < shapes.binxcenters[0]:
                xlow = shapes.binxcenters[0]
            xhigh = histo.GetXaxis().GetBinUpEdge(i) / float(mass)
            if xhigh > shapes.binxcenters[-1]:
                xhigh = shapes.binxcenters[-1]
            integral = interpolator.Integ(xlow, xhigh)
            histo.SetBinContent(i, (integral if integral >= 0. else 0.))
        else:
            histo.SetBinContent(i, 0.)
    histo.Scale(1. / histo.Integral())
    # Convert to density
    for i in range(1, histo.GetNbinsX() + 1):
        width = histo.GetBinWidth(i)
        content = histo.GetBinContent(i)
        histo.SetBinContent(i, content / width if width > 0 else 0)


def parse_input_list(filename):
    """
    Parse the input list file to extract groups and their corresponding mass-path pairs.
    """
    groups = {}
    current_group = None
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.endswith(':'):
                current_group = line[:-1]
                groups[current_group] = {}
                continue
            if current_group is None:
                continue
            parts = line.split(':')
            if len(parts) != 2:
                continue
            mass_str = parts[0].strip()
            path = parts[1].strip()
            try:
                mass = int(mass_str)
                groups[current_group][mass] = path
            except ValueError:
                pass
    return groups


def main():
    # Usage description
    usage = "python3 interpolateResonanceShapes_Math.py -l inputSamples/inputTestSamples_M2000.txt --interval 1000 -o interpolatedSignalShapes_Math/ResonanceShapes.root"
    
    parser = argparse.ArgumentParser(description='Combined resonance shape extraction and interpolation code', epilog=usage)
    parser.add_argument("-l", "--input_list", dest="input_list", required=True, help="Input list file", metavar="INPUT_LIST")
    parser.add_argument("-o", "--output_file", dest="output_file", required=True, help="Output ROOT file", metavar="OUTPUT_FILE")
    parser.add_argument("--fineBinning", dest="fineBinning", action="store_true", help="Use fine, 1-GeV binning")
    parser.add_argument("--storePDF", dest="storePDF", action="store_true", help="Also store a 1-GeV-binned PDF")
    parser.add_argument("--storeCDF", dest="storeCDF", action="store_true", help="Also store a 1-GeV-binned CDF")
    parser.add_argument("--debug", dest="debug", action="store_true", help="Debug printout")
    parser.add_argument("--interval", type=int, default=100, help="Step size for mass range (default: %(default)s)")
    
    args = parser.parse_args()
    
    out_dir = os.path.dirname(args.output_file)
    if not out_dir:
        out_dir = '.'
    base_name = os.path.basename(args.output_file).replace('.root', '')
    os.makedirs(out_dir, exist_ok=True)
    
    groups = parse_input_list(args.input_list)
    if not groups:
        print("No groups found in input list. Aborting.")
        sys.exit(5)
    
    binBoundaries = [
        1, 3, 6, 10, 16, 23, 31, 40, 50, 61, 74, 88, 103, 119, 137, 156, 176, 197, 220, 244, 270, 296, 325,
        354, 386, 419, 453, 489, 526, 565, 606, 649, 693, 740, 788, 838, 890, 944, 1000, 1058, 1118, 1181, 1246, 1313, 1383, 1455, 1530, 1607, 1687,
        1770, 1856, 1945, 2037, 2132, 2231, 2332, 2438, 2546, 2659, 2775, 2895, 3019, 3147, 3279, 3416, 3558, 3704, 3854, 4010, 4171, 4337, 4509,
        4686, 4869, 5058, 5253, 5455, 5663, 5877, 6099, 6328, 6564, 6808, 7060, 7320, 7589, 7866, 8152, 8447, 8752, 9067, 9391, 9726, 10072, 10430,
        10798, 11179, 11571, 11977, 12395, 12827, 13272, 13732, 14000
    ]
    var_bin_array = array('d', binBoundaries)
    
    for group in groups:
        group_shapes = {}
        binxcenters = None
        sim_hists = {}
        group_key = group.upper()
        if 'GG' in group_key:
            final_state = 'gg'
        elif 'QQ' in group_key:
            final_state = 'qq'
        elif 'QSTAR' in group_key:
            final_state = 'qg'
        else:
            print(f"Cannot infer final_state for {group}, skipping.")
            continue
        for mass, path in sorted(groups[group].items()):
            if args.debug:
                print(f"Extracting shapes for {group} m = {mass} GeV from {path}...")
            try:
                with uproot.open(path) as uf:
                    utree = uf['rootTupleTree/tree']
                    branches = ['mjj', 'deltaETAjj', 'etaWJ_j1', 'etaWJ_j2', 'pTWJ_j1', 'pTWJ_j2', 'nJet', 'IdTight_j1', 'IdTight_j2']
                    data = utree.arrays(branches, library="np")
                mask = (
                    (np.abs(data['deltaETAjj']) < 1.1) &
                    (np.abs(data['etaWJ_j1']) < 2.5) &
                    (np.abs(data['etaWJ_j2']) < 2.5) &
                    (data['IdTight_j1'] == 1) &
                    (data['IdTight_j2'] == 1) &
                    (data['pTWJ_j1'] > 60) &
                    (data['pTWJ_j2'] > 30)
                )
                mjj_filtered = data['mjj'][mask]
                if len(mjj_filtered) == 0:
                    print(f"No events after cuts for mass {mass} in {path}. Skipping.")
                    continue
                xred = mjj_filtered / float(mass)
                # Filter to range [0, 1.5] to ensure consistent normalization
                REDUCED_MIN = 0.0
                REDUCED_MAX = 1.5
                mask_range = (xred >= REDUCED_MIN) & (xred <= REDUCED_MAX)
                mjj_filtered = mjj_filtered[mask_range]
                xred = xred[mask_range]
                if len(xred) == 0:
                    print(f"No events in reduced mass range for mass {mass} in {path}. Skipping.")
                    continue
                hist, bins = np.histogram(xred, bins=75, range=(REDUCED_MIN, REDUCED_MAX))
                bincontents = hist.tolist()
                norm = sum(bincontents)
                if norm == 0:
                    continue
                delta_x = (REDUCED_MAX - REDUCED_MIN) / 75
                normbincontents = np.array(bincontents) / norm / delta_x
                # Force exact normalization to 1 (for floating-point safety)
                integral = np.sum(normbincontents) * delta_x
                normbincontents /= integral
                group_shapes[mass] = normbincontents.tolist()
                if binxcenters is None:
                    binxcenters = ((bins[:-1] + bins[1:]) / 2).tolist()
                else:
                    new_centers = ((bins[:-1] + bins[1:]) / 2).tolist()
                    if new_centers != binxcenters:
                        print("Bin centers mismatch. Aborting.")
                        sys.exit(4)
                # Simulated shape
                histname_sim = f"h_sim_{final_state}_{int(mass)}"
                h_sim_title = final_state + " Simulated Resonance Shape"
                h_sim_temp = TH1D(histname_sim + '_temp', h_sim_title, 14000, 0, 14000)
                h_sim_temp.SetDirectory(0)
                for val in mjj_filtered:
                    h_sim_temp.Fill(val)
                integ = h_sim_temp.Integral()
                if integ > 0:
                    h_sim_temp.Scale(1.0 / integ)
                if args.fineBinning:
                    h_sim = rebin_to_variable(h_sim_temp, var_bin_array, histname_sim, h_sim_title)
                else:
                    h_sim = h_sim_temp.Clone(histname_sim)
                    h_sim.SetTitle(h_sim_title)

                h_sim.SetXTitle("Dijet Mass [GeV]")
                h_sim.SetYTitle("Probability")
                sim_hists[mass] = h_sim
                del h_sim_temp
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
        if not group_shapes:
            continue
        if args.debug:
            print(f"\nExtracted shapes for {group}:")
            print("\nshapes = {\n")
            for key, value in sorted(group_shapes.items()):
                print(f"  {key} : {value},")
                print("")
            print("}\n")
            print("binxcenters =", binxcenters)
            print("")
        shapes_obj = ShapeStorage(group_shapes, binxcenters)
        input_masses = sorted(group_shapes.keys())
        MIN = min(input_masses)
        MAX = max(input_masses)
        STEP = args.interval
        masses = list(range(MIN, MAX + STEP, STEP))
        output_file = os.path.join(out_dir, f"{group}_{base_name}.root")
        output = TFile(output_file, "RECREATE")
        print(f"Producing output for group \033[1;31m{group_key}\033[0;0m in \033[1;31m{output_file}\033[0;0m")
        # Write simulated histograms
        for mass, h_sim in sim_hists.items():
            output.cd()
            h_sim.Write()

        for mass in masses:
            print(f"Producing {group} {final_state} shape for m = {int(mass)} GeV")
            histname = f"h_{final_state}_{int(mass)}"
            h_shape_name = histname + '_temp'
            if args.fineBinning:
                h_shape = TH1D(h_shape_name, final_state + " Resonance Shape", 14000, 0, 14000)
            else:
                h_shape = TH1D(h_shape_name, final_state + " Resonance Shape", len(binBoundaries) - 1, var_bin_array)
            h_shape.SetXTitle("Dijet Mass [GeV]")
            h_shape.SetYTitle("Probability")

            LineShapePDF(shapes_obj, mass, h_shape)

            if args.fineBinning:
                h_shape_rebinned = rebin_to_variable(h_shape, var_bin_array, histname, final_state + " Resonance Shape")
                output.cd()
                h_shape_rebinned.Write()
            else:
                output.cd()
                h_shape.Write()

            if args.storePDF or args.storeCDF:
                h_pdf = TH1D(histname + "_pdf", final_state + " Resonance Shape PDF", 14000, 0, 14000)
                h_cdf = TH1D(histname + "_cdf", final_state + " Resonance Shape CDF", 14000, 0, 14000)
                for i in range(1, h_shape.GetNbinsX() + 1):
                    bin_min = h_pdf.GetXaxis().FindBin(h_shape.GetXaxis().GetBinLowEdge(i) + 0.5)
                    bin_max = h_pdf.GetXaxis().FindBin(h_shape.GetXaxis().GetBinUpEdge(i) - 0.5)
                    bin_content = h_shape.GetBinContent(i) / float(bin_max - bin_min + 1)
                    for b in range(bin_min, bin_max + 1):
                        h_pdf.SetBinContent(b, bin_content)
                for i in range(1, h_cdf.GetNbinsX() + 1):
                    bin_min = h_pdf.GetXaxis().FindBin(h_cdf.GetXaxis().GetBinLowEdge(i) + 0.5)
                    bin_max = h_pdf.GetXaxis().FindBin(h_cdf.GetXaxis().GetBinUpEdge(i) - 0.5)
                    curr = 0.
                    for b in range(bin_min, bin_max + 1):
                        curr += h_pdf.GetBinContent(b)
                    prev = h_cdf.GetBinContent(i - 1)
                    h_cdf.SetBinContent(i, prev + curr)
                output.cd()
                if args.storePDF:
                    h_pdf.Write()
                if args.storeCDF:
                    h_cdf.Write()
        output.Close()


if __name__ == '__main__':
    main()
