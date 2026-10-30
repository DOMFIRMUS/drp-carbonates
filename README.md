# drp-carbonates — resolution-aware digital characterisation of two carbonate micro-CT stacks

Reproducible analysis pipeline behind the article *"Support, Resolution and Segmentation Effects on Micro-CT Estimates of Porosity, Specific Surface and Box-Counting Dimension in Carbonate Rocks"*
(Cardoso et al., submitted to *Mathematical Geosciences*). One command regenerates every number, table and figure of the
article from the public micro-CT images.

Requires Python ≥ 3.11.

```bash
python -m pip install -r requirements.txt
python run_all.py            # about 1-1.5 h on a laptop, < 3 GB RAM
python -m pytest -q tests    # 39 validation tests on synthetic volumes and series
```

## Data

The images are **not** redistributed here. Download them from the Digital
Porous Media Portal (formerly Digital Rocks Portal), project
*"Carbonates: porosity and permeability voxel to voxel"*
(Ferreira, Surmas, Silva & Peçanha; doi:[10.17612/V09Y-AW80](https://doi.org/10.17612/V09Y-AW80)):

| Sample | Rock | Stack used | Voxel size (from the slice metadata) | Lab porosity |
|---|---|---|---|---|
| `Calcite-1` | Indiana Limestone (99 % calcite) | 556 × 494 × 1516 voxels, cropped | **45.26 µm** | 0.149 |
| `Dolomite-3` | Silurian Dolomite (97 % dolomite) | 926 × 778 × 1453 voxels, cropped | **28.84 µm** | 0.118 |

The analysed stacks are rectangular crops (556 × 494 and 926 × 778 voxels, all slices) of the
reconstructions distributed by the portal; the crop offsets were not recorded, so the cropped
stacks are archived with the derived data on Zenodo (`*_analysed_stack_*.zip`, written by
`python make_release.py --with-stacks`). Unzip them as `Calcite-1_OriginalImage_C/` and
`Dolomite-3_OriginalImage_C/` next to this directory, or edit `image_dir` in `config.toml`. The voxel size is read from the ImageJ label written in every
slice (e.g. `Calcite-1_OriginalImage_16bits-U_895x898x1516_45.26um0000.tif`).
An earlier analysis of these images by the authors assumed 28.0 µm for both samples;
`run_all.py --steps legacy` reproduces its values exactly and shows
what changes with the metadata voxel sizes.

## Pipeline

| Step | Module | What it computes |
|---|---|---|
| `ingest` | `drp/io.py` | stacks the TIFFs into a memory-mapped `uint16` cache; parses voxel size; per-slice histograms |
| `qc` | `drp/steps.py` | solid-peak drift along z and its association with slice porosity (see *Drift*) |
| `segmentation` | `drp/histogram.py` | end-members, thresholds of every method, porosity vs. criterion, grey-level mass balance |
| `legacy` | `drp/fractal.py`, `drp/surface.py` | exact replication of the previous workflow (slices 300–599, mean − 1.1 sd, 28.0 µm, porespy box counting) |
| `profiles` | `drp/steps.py` | porosity profiles along x, y, z (x, y also over a z window of common height); effective sample size; calibrated trend tests; sensitivity to the drift correction |
| `rev` | `drp/rev.py` | variance–volume scaling of window porosity (binary and mass balance), descriptive z-slab resampling of α, Monte Carlo test against stationary media with a 1-mm correlation scale |
| `cube` | `drp/steps_morph.py` | central cube of equal **physical** edge (20 mm) in both samples: S_V by 5 estimators, box counting, two-point correlation, connectivity, sweep over segmentation criteria |
| `psd` | `drp/poresize.py` | continuous (opening-based) pore-size distribution |
| `esf` | `drp/sharpness.py` | effective spatial resolution from edge-spread profiles |
| `replicates` | `drp/steps_morph.py` | non-overlapping 5-mm cubes tiling both stacks (n = 260 and 160) |
| `resolution` | `drp/volume.py` | area-weighted block averaging to coarser voxels, same physical criterion |
| `sensitivity` | `drp/sensitivity.py` | variance–volume exponent without the z trend, without the drift correction and against null fields of 1–5 mm correlation scale; lateral non-uniformity of the solid peak; sensitivity of the porosity budget to the solid end-member (tables T15, T16) |
| `compare` | `drp/compare.py` | cross-sample synthesis with uncertainties → `outputs/data/results.json` |
| `tables`, `figures`, `report` | `drp/tables.py`, `drp/figures.py`, `drp/report.py` | manuscript tables (CSV/XLSX/LaTeX/MD), figures (PDF + 600-dpi PNG), summary |
| `captions`, `supplement` | `drp/captions.py`, `drp/supplement.py` | figure captions with the numbers from `results.json`; supplementary material (DOCX) incl. estimator validation |
| `export` | `drp/export.py` | derived-data package for archiving (CSV/JSON/compressed TIFF + data dictionary) |

`python make_release.py --dest ../SUBMISSAO` then copies figures, tables and supplementary
material into a submission folder and builds the two Zenodo archives (code, derived data).

Run a subset with `python run_all.py --steps cube figures --samples dolomite3`;
`--quick` runs a reduced version into `outputs_quick/` for testing. Every run
writes `outputs/provenance.json` (package versions, config and code hashes, steps
run) and updates `outputs/provenance_steps.json` (when each step and sample last
ran, with which configuration and code); the export step refuses to package
outputs produced with a different `config.toml`.

Nothing loads a whole stack into memory: every estimator works slab by slab and
the slab decompositions are exact (tested against whole-volume results).

## Methods (summary for the Methods section)

**Segmentation on a physical scale.** Grey levels are mapped onto a linear
partial-volume scale `f(I) = (I_s − I)/(I_s − I_p)`, with `I_s` the solid-peak
mode and `I_p` the pore end-member (0.1 % grey-level quantile; sensitivity range
0.02–0.5 %). A voxel is pore when `f(I) > f`; `f = 0.5` (majority-pore voxels)
is the primary criterion. The same `f` is the same physical criterion in both
scans, which is not true of `mean − k·sd`. Otsu (1979), Li & Lee (1993),
Kittler & Illingworth (1986), triangle (Zack et al. 1977) and laboratory-matched
thresholds are reported for comparison.

**Threshold-free porosity.** Under linear mixing of attenuation,
`φ_mix = (I_s − Ī)/(I_s − I_p)`; it needs no threshold and is unbiased under
zero-mean noise. The solid-phase (sub-resolution) porosity needed to close the
budget with the laboratory value is `φ_bg = (φ_lab − φ_mix)/(1 − φ_mix)`.

**Drift.** A slow change of the X-ray gain makes darker slices look more porous,
i.e. a *negative* association between the per-slice solid-peak level and the
apparent porosity. The `qc` step measures it (Spearman ρ with a circular-block
bootstrap interval and a p-value against phase-randomised surrogates, because
both series are almost perfectly autocorrelated). Here the association is
inconclusive in both samples, so the correction (per-slice gain fixing the
solid-peak mode) is an a-priori choice set per sample in `config.toml` (the
dolomite, whose solid peak drops 5.7 % at the top of the stack). The
`profiles` step reports the z statistics with and without the correction.

**Specific surface S_V.** (i) Cauchy–Crofton intercept counting,
`S_V = 2⟨P_L⟩`, over the 13 lattice directions weighted by their Voronoi solid
angles (Ohser & Mücklich 2000); (ii) marching cubes (Lorensen & Cline 1987;
Lewiner et al. 2003) on the grey levels at the threshold (sub-voxel interface)
and on the binary image; (iii) voxel faces (upper bound). Validation on digital
spheres: Crofton within 3 %, binary marching cubes biased +5–10 %, faces ≈ 1.5×.

**Box counting.** Boxes are counted when they contain boundary voxels (pore
voxels with a solid 6-neighbour); only complete boxes are inspected and the
count is extrapolated from the occupied fraction; four grid offsets are
averaged; box sizes are set in millimetres, and the sizes inside the fitting
range are whole-voxel sizes whose end points lie inside the range for every
voxel size, so both scans and all coarsened grids are fitted over the same
window. The local slope `−d ln N/d ln s` and the occupancy `N s³/V` are
reported: above occupancy ≈ 0.5 nearly every box contains interface and the
slope tends to 3 whatever the geometry (saturation). A scaling regime is
claimed only for a run of consecutive unsaturated sizes whose local slopes stay
within ±0.05 of their mean over at least 0.8 decade (and at least three
sizes). Controls: a level-5 Menger sponge (D = ln 20/ln 3; aligned boxes of
3–81 voxels) is recovered within 4 %, but every box of the sponge is saturated,
so the occupancy rule can only resolve D below about 2.6; a Boolean model of
smooth spheres (no fractal structure) shows a short quasi-plateau with slope
≈ 1.8, so a slope below 2 over a fraction of a decade is not evidence of
fractal geometry. Between-sample and between-resolution contrasts of D are
also refitted on one interpolated grid of sizes common to all curves, because
integer box sizes of different voxels never coincide.
The porespy definition ("box contains both phases") is biased low at small
boxes (a plane gives slope 1 at s = 2 voxels) and is kept only to reproduce the
previous values.

**REV.** Porosity in 2000 random cubic windows per size (0.25–20 mm) from a
block summed-area table; `Var(φ_V) ∝ V^−α` is fitted for windows ≥ 2 mm
(requested edge; the realised range is reported). α = 1 corresponds to a finite
integral range (Matheron 1971; Kanit et al. 2003); α < 1 means long-range
heterogeneity and no REV at the scale of the sample. The OLS standard error of
the slope is not a sampling error (nested, overlapping windows), so α gets (i) a
standard error from a moving-block bootstrap of 8-mm z-slabs (lower bound) and
(ii) a Monte Carlo test against stationary Gaussian fields with a finite
integral range (1-mm correlation length, longer than measured), analysed with
the same windows and fit. Windows lie inside one finite stack, so the
finite-domain form `Var = K(V^−α − V_stack^−α)` (Krige's relation; Lantuéjoul
1991) is fitted as a sensitivity analysis. The spatial standard error of a
whole-stack porosity is the larger of the two extrapolations to the stack
volume. The grey-level mass balance gets its own variance–volume curve.

**Profiles.** Slice porosity along x, y and z; effective sample size from a
bias-corrected AR(1) model (the initial-positive-sequence estimator of Thiébaux
& Zwiers 1984 is kept for reference; it is biased high when n_eff is small);
mean intervals with the variance corrected for serial dependence; linear trends
with Newey–West standard errors whose |t| is calibrated by an AR(1) parametric
bootstrap (size 2.5–7 % at 5 % in simulations with ρ = 0.98–0.995), with
phase-randomised surrogates as a conservative bound; heterogeneity compared at
equal slab thickness (1 mm) and equal extent (x and y profiles averaged over the
same z height in both samples).

**Connectivity, correlation, pore size.** 6- and 26-connected labelling with
exact slab merging; percolation in x, y, z; two-point correlation along the
three axes by FFT (correlation length at 1/e); opening-based continuous pore-size
distribution (Münch & Holzer 2008).

**Effective resolution.** Grey-level profiles across iso-surface points are
fitted with an error-function edge; the median σ over full-contrast, well-fitted
edges approximates the point-spread function (validated on a blurred synthetic
sphere and on slabs): curvature and roughness widen edges, whereas a second
interface inside the fitting window narrows them (−7 % for a 16-voxel pore at
σ = 2.5 voxels, negligible from 20 voxels).

**Replicates and inference.** One specimen per lithology: tests compare the two
imaged volumes, not rock types. Sub-volumes of a z-layer are correlated, and
layer means are strongly correlated along z (only a few effectively
independent layers per specimen), so specimen means and contrasts (first minus
second sample) are computed from the series of z-layer means with an AR(1)
effective size (Kendall-corrected lag-1 correlation capped at 0.95) and a
Welch–Satterthwaite t; in simulations of the design (13 and 8 layers) the test
size was 4–13 % at the 5 % level, so p-values are labelled approximate, and the
tables also say whether the ranges of the layer means of the two specimens are
disjoint (a model-free description). Porosity-adjusted contrasts (ANCOVA,
evaluated across the common porosity range) are descriptive. Hedges g and
Cliff's δ are descriptive; i.i.d. tests are not reported.
Replicate cubes with fewer than three unsaturated box sizes (the most porous)
have no D_unsat; their number and porosity are reported with two sensitivity
variants. A Kozeny–Carman proxy `k = φ³/(5 S_V²)` of the resolved pore space is
reported per criterion and per sub-volume.

## Outputs

```
outputs/
  data/results.json          every reported number, with uncertainties
  data/*.json|npz|csv        per-step results (profiles, box counts, S2, clusters, ...)
  tables/T01..T13.*          CSV (full precision), XLSX (one sheet each), LaTeX, Markdown
  figures/F01..F13.{pdf,png} journal figures (vector PDF + 600-dpi PNG)
  RESULTS_SUMMARY.md         auto-generated summary (Portuguese)
  FIGURE_CAPTIONS.md         captions (English)
  Supplementary_Material.docx
  zenodo_data/               derived-data package
  provenance.json, logs/
```

## Assumptions to keep in mind

* The uncertainty of the laboratory porosity is not given by the data source;
  ±0.005 (1 sd) is assumed in `config.toml` and used only in the budget.
* The pore end-member is the darkest 0.1 % of voxels; blur keeps small pores
  from ever reaching it, which is why every result is also given as a function
  of `f` and of the end-member range.
* Block averaging idealises a coarser acquisition (no extra noise or PSF).
* The edge-spread σ approximates the PSF (see *Effective resolution*).
* The block-bootstrap SE of α resamples only along z (lower bound); the
  Monte Carlo null uses a correlation length longer than measured (conservative).
* The drift-correction setting is an a-priori choice (the association test is
  inconclusive in both samples); Table T14 reports the alternative.
* LaTeX tables need `\usepackage{graphicx}` (they are scaled to the text width).

## Citation and licence

Archived release (this version): https://doi.org/10.5281/zenodo.23818175

Derived data and analysed image stacks: https://doi.org/10.5281/zenodo.23818173

Code: MIT licence (`LICENSE`). Derived data: CC BY 4.0. Please cite the article
and the archived release (see `CITATION.cff`), and the source images
(doi:10.17612/V09Y-AW80).

## References

Kanit T, Forest S, Galliet I, Mounoury V, Jeulin D (2003) Int J Solids Struct 40:3647–3679 ·
Künsch HR (1989) Ann Stat 17:1217–1241 ·
Lantuéjoul C (1991) J Microsc 161:387–403 ·
Kittler J, Illingworth J (1986) Pattern Recognit 19:41–47 ·
Lewiner T, Lopes H, Vieira AW, Tavares G (2003) J Graph Tools 8:1–15 ·
Li CH, Lee CK (1993) Pattern Recognit 26:617–625 ·
Lorensen WE, Cline HE (1987) Comput Graph 21:163–169 ·
Matheron G (1971) The theory of regionalized variables and its applications ·
Münch B, Holzer L (2008) J Am Ceram Soc 91:4059–4067 ·
Newey WK, West KD (1987) Econometrica 55:703–708 ·
Ohser J, Mücklich F (2000) Statistical analysis of microstructures in materials science. Wiley ·
Otsu N (1979) IEEE Trans Syst Man Cybern 9:62–66 ·
Politis DN, Romano JP (1992) In: Exploring the limits of bootstrap. Wiley, pp 263–270 ·
Theiler J, Eubank S, Longtin A, Galdrikian B, Farmer JD (1992) Physica D 58:77–94 ·
Thiébaux HJ, Zwiers FW (1984) J Clim Appl Meteorol 23:800–811 ·
Zack GW, Rogers WE, Latt SA (1977) J Histochem Cytochem 25:741–753.
