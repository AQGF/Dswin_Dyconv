# DSwin_Dyconv Experiment Analysis

This document summarizes the implementation status, current quantitative results, and the main debugging conclusions from the recent IXI/TransMorph experiments.

## 1. Scope

The repository currently contains the following experiment families under `IXI/TransMorph`:

- `TransMorph-bspl` baseline
- `DSwin3D + DynConv`
- `DSwin3D + AGDynConv`
- lightweight B-spline variants
- expanded lightweight AGDynConv variant:
  - `embed_dim = 64`
  - `resize_channels = (24, 24)`

The main recent comparison target is `TransMorph-bspl` on IXI.

## 2. Current Evaluation Protocol

Validation and analysis are based on the same 30 VOI structures used by the IXI TransMorph codebase.

The 30-VOI Dice is computed in:

- `IXI/TransMorph/utils.py` -> `dice_val_VOI`

The VOI list is:

`[1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36]`

These indices correspond to the standard FreeSurfer-mapped structures such as:

- cerebral white matter / cortex
- lateral ventricles
- cerebellum white matter / cortex
- thalamus, caudate, putamen, pallidum
- 3rd and 4th ventricle
- brain stem
- hippocampus, amygdala
- CSF
- ventral DC
- choroid plexus

## 3. CSV-Based Quantitative Comparison

The following two CSV files were compared under the same 30-VOI protocol:

- baseline:
  - `TransMorphBSplineJan_ncc_1_diffusion_1.csv`
- ours:
  - `TransMorphBSpline_DSwin3D_AGDynConv_Lite_ncc_1_diffusion_1_offmag_1.0_offsmooth_1.0.csv`

Both contain 115 test subjects.

### 3.1 Overall Case-Wise 30-VOI Mean

Case-wise mean means:
- first compute the mean Dice across the 30 VOIs for each subject
- then compute the mean/std across all subjects

Results:

- `TransMorph-bspl`
  - mean = `0.758479`
  - std = `0.026386`
  - var = `0.000696`
- `DSwin3D_AGDynConv_Lite`
  - mean = `0.754283`
  - std = `0.024753`
  - var = `0.000613`

Gap:

- `delta = -0.004196`

This indicates that the current lightweight AGDynConv variant is close to the baseline, but still below it.

### 3.2 Structure-Level Deltas

The comparison showed that the method is not uniformly worse.

Structures where the current method is stronger:

- Left-Putamen: `+0.018145`
- Right-Putamen: `+0.014263`
- Left-Pallidum: `+0.012740`
- Left-Cerebellum-Cortex: `+0.004079`
- Right-Pallidum: `+0.003684`
- Right-Cerebellum-Cortex: `+0.002040`

Structures where the current method is clearly weaker:

- Left-choroid-plexus: `-0.023873`
- Right-choroid-plexus: `-0.019702`
- CSF: `-0.017069`
- 3rd-Ventricle: `-0.012965`
- Right-Cerebral-Cortex: `-0.011467`
- Left-Cerebral-Cortex: `-0.010575`
- Right-Amygdala: `-0.009744`
- Left-Hippocampus: `-0.009056`
- Left-Cerebral-White-Matter: `-0.008592`
- Right-Cerebral-White-Matter: `-0.008260`
- Right-Hippocampus: `-0.008194`

Interpretation:

- the method already helps some basal-ganglia-related structures
- the remaining weakness is concentrated in:
  - choroid plexus and other difficult small structures
  - CSF / ventricle-related regions
  - cortex / white-matter level large-structure consistency

This is consistent with a model whose local dynamic modules are partially useful, but whose final geometric refinement capacity is still slightly insufficient.

## 4. Training Dynamics Observed So Far

### 4.1 Original Lite AGDynConv

Experiment directory:

- `logs/TransMorphBSpline_DSwin3D_AGDynConv_Lite_ncc_1_diffusion_1_offmag_1.0_offsmooth_1.0`

Observed validation trend from `logfile.log`:

- epoch 0: `0.4466`
- epoch 1: `0.5778`
- epoch 2: `0.6157`
- epoch 3: `0.6530`
- epoch 4: `0.6697`
- epoch 10: `0.7013`
- epoch 19: `0.7171`
- epoch 499: `0.7445`

Interpretation:

- the model does not fail to start training
- the early-stage optimization is not obviously broken
- the main issue is the final ceiling, not the initial ramp-up

### 4.2 Offset Collapse Observation

During training, the following behavior was observed:

- `offset_mag` quickly drops to approximately `1e-6`
- `offset_smooth` quickly drops to approximately `1e-10`

Interpretation:

- the offset branch is likely collapsing toward a near-zero solution
- the model can solve most of the registration objective without actually using large deformable offsets
- therefore, simply adding deformable offset does not guarantee the dynamic correspondence mechanism is effectively used

This is an important methodological observation for future improvement.

### 4.3 Detach Guidance Test

`detach_guidance = False` was tested.

Observed behavior:

- the overall loss trend remained almost identical to the detached version
- the offset-related terms still quickly collapsed to very small values

Interpretation:

- the main problem is not just the guidance gradient being detached
- the current offset branch likely lacks a strong enough incentive to remain active
- the issue is closer to structural under-utilization than simple gradient blocking

## 5. Main Debugging Conclusions

### 5.1 What the results suggest

The current evidence suggests:

- this is not a complete failure of DSwin3D / AGDynConv
- the method already changes behavior on some structures
- however, the Lite model still has a slightly lower final ceiling than `TransMorph-bspl`

### 5.2 Most likely bottleneck

The most likely bottleneck is:

- insufficient tail / decoder / B-spline head refinement capacity

rather than:

- catastrophic optimization failure in the first few epochs

This is why the following direction was prioritized:

- increase `resize_channels`
- slightly increase `embed_dim`

## 6. Current Tuning Decisions

Based on the above, the next practical tuning direction was chosen as:

- `AdamW`
- `detach_guidance = True`
- `offset_mag_weight = 0`
- `offset_smooth_weight = 0.1`

and then a light capacity expansion:

- `embed_dim: 48 -> 64`
- `resize_channels: (16,16) -> (24,24)`

This was implemented as the expanded AGDynConv Lite variant.

## 7. Files Added for Expanded Variant

Added files:

- `IXI/TransMorph/models/configs_TransMorph_bspl_DSwin3D_AGDynConv_Lite_Expanded.py`
- `IXI/TransMorph/models/TransMorph_bspl_DSwin3D_AGDynConv_Lite_Expanded.py`
- `IXI/TransMorph/train_TransMorph_bspl_DSwin3D_AGDynConv_Lite_Expanded_AdamW_OffMag0_OffSmooth0p1.py`
- `IXI/TransMorph/infer_TransMorph_bspl_DSwin3D_AGDynConv_Lite_Expanded_AdamW_OffMag0_OffSmooth0p1.py`

The expanded configuration currently uses:

- `embed_dim = 64`
- `depths = (1, 1, 2, 1)`
- `resize_channels = (24, 24)`
- `AdamW`
- `detach_guidance = True`
- `offset_mag_weight = 0`
- `offset_smooth_weight = 0.1`

## 8. Recommended Next Debug Steps

Recommended next steps are:

1. Compare the expanded variant against the current Lite variant using the same 30-VOI metric.
2. If improved, add one more controlled variant:
   - `embed_dim = 64`
   - `resize_channels = (32, 32)`
3. Add explicit logging for:
   - `offset_magnitude` per stage
   - `branch_energy` per stage
   - dynamic-conv gate branch weights
4. Run inference-time ablations:
   - normal guidance
   - zero guidance
   - zero offsets

These steps will help answer whether:

- DSwin3D is truly contributing useful correspondence modeling
- AGDynConv is truly using attention-derived guidance
- the remaining gap is mostly due to capacity or to ineffective dynamic behavior
