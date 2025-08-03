# Resonance Shape Interpolation

This repository provides Python scripts for extracting and interpolating resonance shapes from signal samples in CMS dijet analyses. The main script performs moment morphing to generate interpolated shapes at arbitrary masses based on simulated samples. A verification plotting script is also included to compare interpolated shapes with simulated ones.

The code is designed to run within the CMS software environment (CMSSW) on lxplus.


## Installation

1. Log in to lxplus and set up the CMSSW environment:

   ```
   cmsrel CMSSW_15_0_10
   cd CMSSW_15_0_10/src
   cmsenv
   ```

2. Clone this repository:

   ```
   git clone --recursive https://github.com/asimsek/DijetShapeInterpolator DijetShapeInterpolator
   cd DijetShapeInterpolator
   ```

## Usage

### Interpolation Script: `interpolateResonanceShapes_original.py`

This script extracts histograms from signal sample ROOT files, applies selection cuts, and performs moment morphing interpolation using RooFit to generate shapes at specified mass points.


#### Examples

For base directory:

```
python3 interpolateResonanceShapes_original.py -b /eos/cms/store/user/dagyel/DiJet/rootNTuples_reduced/signalSamples -o interpolatedSignalShapes/ResonanceShapes.root --interval 100 --fineBinning
```

For list file:

```
python3 interpolateResonanceShapes_original.py -l inputSamples/inputSamples.txt -o interpolatedSignalShapes/ResonanceShapes.root --interval 100 --fineBinning
```

The script will process groups of samples and produce output ROOT files with interpolated histograms.

### Verification Plotting Script: `verifyInterpolation_plotter.py`

This script plots comparisons between simulated and interpolated shapes, including ratios and Kolmogorov-Smirnov (KS) test results.

#### Examples

```
python3 verifyInterpolation_plotter.py --root1 interpolatedSignalShapes/QstarTo2J_Run3Summer22MiniAODv4_ResonanceShapes_all.root --root2 interpolatedSignalShapes/QstarTo2J_Run3Summer22MiniAODv4_ResonanceShapes_sub.root --mass 7000 --type qg
```

```
python3 verifyInterpolation_plotter.py --root1 interpolatedSignalShapes/QstarTo2J_Run3Summer22MiniAODv4_ResonanceShapes_all.root --root2 interpolatedSignalShapes/QstarTo2J_Run3Summer22MiniAODv4_ResonanceShapes_sub.root --mass 2000 --type qg
```



