#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, re, sys
from collections import OrderedDict
from array import array
import numpy as np
import uproot

import ROOT
from ROOT import gROOT, gStyle, TCanvas, TH1D, TLegend, TLatex, TGaxis

# -------------------- Input parsing --------------------

def parse_grouped_list(path):
    """
    Text format:
      GroupA:
      500: /path/a.root
      1000: /path/b.root

      GroupB:
      750: /path/c.root

    Returns OrderedDict[str, list[tuple[int,str]]]
    """
    groups = OrderedDict()
    cur = None
    with open(path, "r") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            left, right = [s.strip() for s in line.split(":", 1)]
            if re.fullmatch(r"\d+", left) and cur is not None:
                if right:
                    groups[cur].append((int(left), right))
            else:
                cur = left
                groups.setdefault(cur, [])
    return groups

# -------------------- Data --------------------

def read_mjj_filtered(root_path, treepath="rootTupleTree/tree"):
    """Read mjj with your selection; returns np.array in GeV."""
    try:
        with uproot.open(root_path) as uf:
            utree = uf[treepath]
            branches = [
                "mjj","deltaETAjj","etaWJ_j1","etaWJ_j2",
                "pTWJ_j1","pTWJ_j2","IdTight_j1","IdTight_j2"
            ]
            data = utree.arrays(branches, library="np")
        mask = (
            (np.abs(data["deltaETAjj"]) < 1.1) &
            (np.abs(data["etaWJ_j1"]) < 2.5) &
            (np.abs(data["etaWJ_j2"]) < 2.5) &
            (data["IdTight_j1"] == 1) &
            (data["IdTight_j2"] == 1) &
            (data["pTWJ_j1"] > 60) &
            (data["pTWJ_j2"] > 30)
        )
        return data["mjj"][mask]
    except Exception as e:
        sys.stderr.write(f"[WARN] Failed to read {root_path}: {e}\n")
        return np.array([], dtype=np.float64)

# -------------------- ROOT helpers --------------------

def make_hist(name, bin_edges):
    arr = array('d', bin_edges)
    h = TH1D(name, "", len(arr)-1, arr)
    h.Sumw2(False)
    h.SetDirectory(0)
    return h

def fill_hist_from_array(h, values_TeV):
    n = len(values_TeV)
    if n == 0: return
    x = array('d', values_TeV)
    w = array('d', [1.0]*n)
    h.FillN(n, x, w)

# -------------------- Main --------------------

def main():
    p = argparse.ArgumentParser(description="Overlay mjj shapes per file (ROOT HIST).")
    p.add_argument("input", help="Grouped list file")
    p.add_argument("--tree", default="rootTupleTree/tree", help="TTree path")
    p.add_argument("--out", required=True, help="Output folder; last path item is file basename")
    p.add_argument("--bins", default="auto",
                   help="Number of bins (int) or 'auto' (global) over [0,11] TeV (default: auto)")
    p.add_argument("--logy", action="store_true", help="Log-scale y")
    p.add_argument("--lumi", default="", help="Optional lumi text to prepend on the right")
    args = p.parse_args()

    # Output paths
    out_dir = os.path.normpath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    out_base = os.path.basename(out_dir)
    out_pdf = os.path.join(out_dir, f"{out_base}.pdf")
    out_png = os.path.join(out_dir, f"{out_base}.png")

    gROOT.SetBatch(True)
    gStyle.SetOptStat(0)
    gStyle.SetLegendBorderSize(0)
    gStyle.SetLegendFillColor(0)
    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)

    # CMS-ish fonts/sizes
    for ax in ("X","Y","Z"):
        gStyle.SetTitleFont(42, ax)
        gStyle.SetLabelFont(42, ax)
        gStyle.SetTitleSize(0.055, ax)
        gStyle.SetLabelSize(0.045, ax)

    # Parse input groups
    groups = parse_grouped_list(args.input)
    if not groups:
        sys.stderr.write("[ERROR] No groups found in input.\n")
        sys.exit(1)

    # Read data (convert to TeV)
    xlow_TeV, xhigh_TeV = 0.0, 11.0
    perfile = OrderedDict()  # {group: [(mass, path, mjj_TeV), ...]}
    all_vals = []

    for gname, items in groups.items():
        lst = []
        for mass, path in items:
            mjj_GeV = read_mjj_filtered(path, treepath=args.tree)
            mjj_TeV = (mjj_GeV / 1000.0).astype(np.float64, copy=False)
            lst.append((mass, path, mjj_TeV))
            if mjj_TeV.size:
                all_vals.append(mjj_TeV)
        perfile[gname] = sorted(lst, key=lambda t: t[0])  # sort by mass for consistent overlay order

    # Common bin edges
    if isinstance(args.bins, str) and args.bins.lower() == "auto":
        if all_vals:
            concat = np.concatenate(all_vals)
            bin_edges = np.histogram_bin_edges(concat, bins="auto", range=(xlow_TeV, xhigh_TeV))
        else:
            bin_edges = np.linspace(xlow_TeV, xhigh_TeV, 111)  # fallback: 0.1 TeV bins
    else:
        try:
            nbins = int(args.bins)
            if nbins < 1: raise ValueError
            bin_edges = np.linspace(xlow_TeV, xhigh_TeV, nbins + 1)
        except Exception:
            sys.stderr.write("[WARN] Invalid --bins; using 'auto'.\n")
            if all_vals:
                concat = np.concatenate(all_vals)
                bin_edges = np.histogram_bin_edges(concat, bins="auto", range=(xlow_TeV, xhigh_TeV))
            else:
                bin_edges = np.linspace(xlow_TeV, xhigh_TeV, 111)

    # Histogram style
    # Different color for each group; first group solid line, others dashed lines
    group_colors = [
        ROOT.kBlack, ROOT.kRed+1, ROOT.kBlue+1, ROOT.kGreen+2, ROOT.kMagenta+2,
        ROOT.kOrange+7, ROOT.kCyan+1, ROOT.kViolet+1, ROOT.kTeal+4, ROOT.kPink+7,
        ROOT.kAzure+1, ROOT.kSpring+9, ROOT.kGray+2
    ]

    hists_by_group = OrderedDict()
    ymax = 0.0

    for gi, (gname, items) in enumerate(perfile.items()):
        col = group_colors[gi % len(group_colors)]
        lstyle = 1 if gi == 0 else 2  # first group solid, others dashed
        glist = []
        for fi, (mass, path, mjj_TeV) in enumerate(items):
            h = make_hist(f"h_{gi}_{fi}", bin_edges)
            if mjj_TeV.size:
                fill_hist_from_array(h, mjj_TeV)

            # normalize EACH distribution to its integral
            integ = h.Integral()
            if integ > 0:
                h.Scale(1.0 / integ)
            h.SetLineColor(col)
            h.SetLineWidth(2)
            h.SetLineStyle(lstyle)
            h.GetXaxis().SetTitle("m_{jj} [TeV]")
            h.GetYaxis().SetTitle("Normalized yield / TeV")
            h.GetYaxis().SetTitleOffset(0.85)
            h.GetXaxis().SetTitleOffset(1.0)
            #h.GetYaxis().SetNdivisions(505, False)
            h.GetXaxis().SetNdivisions(511, False)
            h.GetXaxis().SetRangeUser(0.0, 11.0) 
            h.GetXaxis().SetTickLength(0.02)
            h.GetYaxis().SetTickLength(0.02)
            h.SetMarkerSize(0)
            glist.append(h)
            ymax = max(ymax, h.GetMaximum())
        hists_by_group[gname] = glist

    ymax = ymax * 1.30 if ymax > 0 else 1.0
    # Canvas
    can = TCanvas("c", "c", 1600, 1000)
    can.SetMargin(0.10, 0.02, 0.14, 0.06)  # left, right, bottom, top (more top for CMS text)
    if args.logy: can.SetLogy(True)

    first_drawn = None
    first = True
    for glist in hists_by_group.values():
        for h in glist:
            if first:
                first_drawn = h
                if args.logy:
                    # choose a positive min; otherwise ROOT complains
                    ymin_pos = 1e-6
                    for g2 in hists_by_group.values():
                        for hh in g2:
                            nb = hh.GetNbinsX()
                            for ib in range(1, nb+1):
                                v = hh.GetBinContent(ib)
                                if v > 0:
                                    ymin_pos = min(ymin_pos, v)
                    first_drawn.SetMinimum(ymin_pos * 0.5)
                    first_drawn.SetMaximum(ymax if ymax > 0 else 10.0)
                else:
                    first_drawn.SetMinimum(0.0)
                    first_drawn.SetMaximum(ymax)
                h.Draw("HIST")
                first = False
            else:
                h.Draw("HIST SAME")

    # Legend: **only group names** (one entry per group using the first hist in that group)
    # --- Right & top-aligned legend (auto height) ---
    labels = [(gname.replace("_"," "), glist[0])
              for gname, glist in hists_by_group.items() if glist]
    n = len(labels)

    text_size = 0.035              
    row_h    = text_size * 1.35    # approx row height in NDC
    pad_top  = 0.90                # a bit below the pad top to avoid clipping
    x2, y2   = 0.90, pad_top       # TOP-RIGHT corner
    leg_w    = 0.35                # tweak width to taste
    leg_h    = row_h * n + 0.02    # rows + small padding

    x1 = x2 - leg_w
    y1 = max(0.50, y2 - leg_h)     # grow downward, don't go below mid-pad

    leg = ROOT.TLegend(x1, y1, x2, y2)  # NDC coords
    leg.SetTextSize(text_size)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    for lab, h0 in labels:
        leg.AddEntry(h0, lab, "l")
    leg.Draw()


    # CMS Text
    cms = TLatex(); cms.SetNDC(True); cms.SetTextFont(62); cms.SetTextSize(0.060)
    cms.DrawLatex(0.12, 0.87, "CMS")
    pre = TLatex(); pre.SetNDC(True); pre.SetTextFont(52); pre.SetTextSize(0.050)
    pre.DrawLatex(0.20, 0.87, "Preliminary")

    # sqrt(s) text on the far-right of the top margin
    right = TLatex(); right.SetNDC(True); right.SetTextFont(42); right.SetTextSize(0.040); right.SetTextAlign(31); right.SetTextFont(42)
    etext = f"{args.lumi},  #sqrt{{s}} = 13.6 TeV" if args.lumi else "#sqrt{s} = 13.6 TeV"
    right.DrawLatex(0.975, 0.950, etext)

    can.Update()
    can.SaveAs(out_pdf)
    can.SaveAs(out_png)

if __name__ == "__main__":
    main()

