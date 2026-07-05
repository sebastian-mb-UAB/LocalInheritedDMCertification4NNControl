# LocalInheritedDMCertification4NNControl

This repository contains the datasets, trained neural-network controllers, and MATLAB scripts used for PID-to-neural-network imitation learning and post-training inherited disk-margin certification in a nonlinear Van de Vusse CSTR benchmark.

The associated manuscript is:

**Post-Training Inherited Disk-Margin Certification for Local-Affine Neural Network Controllers Trained from PID Teachers**

The main goal is to train neural-network controllers that imitate a validated PID teacher, evaluate their autonomous closed-loop behavior, and support local robustness certification through inherited disk-margin analysis.

---

## Repository contents

The repository includes:

- PID-generated datasets for imitation learning.
- Training scripts for four neural-network controller variants:
  - `BL`: Batch Learning.
  - `BL_S`: Smoothed Batch Learning.
  - `ESS`: Enhanced Scheduled Sampling.
  - `ESS_FT`: Enhanced Scheduled Sampling with Fine-Tuning.
- Representative trained neural-network controllers used as the 6th training seed in the paper.
- Nonlinear CSTR validation scripts comparing the PID teacher and the trained neural-network controllers.

---

## Folder structure

```text
LocalInheritedDMCertification4NNControl/
│
├── D_tr_T.mat
├── D_val_T.mat
├── D_loc_T.mat
├── D_loc_val_T.mat
│
├── 02_train_cstr_BL.m
├── 03_train_cstr_ESS_FT.m
├── 04_validate_cstr_cases_PID_vs_NN.m
│
├── models/
│   ├── Models.txt
│   ├── trainedNN_CSTR_BL.mat
│   ├── trainedNN_CSTR_BL_smooth.mat
│   ├── trainedNN_CSTR_ESS.mat
│   └── trainedNN_CSTR_ESS_FT.mat
│
└── README.md
```

---

## Datasets

The repository includes the following PID-teacher datasets:

| File | Description |
|---|---|
| `D_tr_T.mat` | Main training dataset generated from the nonlinear CSTR under PID control. |
| `D_val_T.mat` | Independent validation dataset generated from the PID teacher. |
| `D_loc_T.mat` | Local dataset around the nominal operating point, including equilibrium anchor samples. |
| `D_loc_val_T.mat` | Local validation dataset used to check behavior near the certified operating region. |

The neural-network input regressor is

```math
\phi_k =
[e_k, e_{k-1}, \ldots, e_{k-w_s+1}, i_{e,k}, u_{k-1}],
```

where:

- `e_k` is the tracking error.
- `w_s = 10` is the error-window size.
- `i_{e,k}` is the accumulated error.
- `u_{k-1}` is the previous control deviation.

The neural-network output is the manipulated-variable deviation:

```math
u_{N,k} = Q_{N,k} - Q_0.
```

---

## Pretrained models

The folder `models/` contains the representative trained neural-network controllers used as the 6th seed in the paper.

| File | Controller |
|---|---|
| `trainedNN_CSTR_BL.mat` | Batch Learning controller. |
| `trainedNN_CSTR_BL_smooth.mat` | Smoothed Batch Learning controller. |
| `trainedNN_CSTR_ESS.mat` | Enhanced Scheduled Sampling controller. |
| `trainedNN_CSTR_ESS_FT.mat` | Enhanced Scheduled Sampling with Fine-Tuning controller. |
| `Models.txt` | List of available trained models. |

These models can be loaded directly for validation or certification without retraining.

---

## Requirements

The scripts are written in MATLAB and require:

- MATLAB.
- Deep Learning Toolbox.
- Control System Toolbox.
- The helper classes and functions included in the repository, such as:
  - `CSTR_NNTrainingUtils`
  - `CSTR_LocalStabilityUtils`

Some certification routines may also require additional MATLAB functionality depending on the specific robustness-analysis scripts used.

---

## Main scripts

### `02_train_cstr_BL.m`

This script trains the Batch Learning neural-network controller for PID imitation.

It can generate two variants:

- `BL`: standard batch learning.
- `BL_S`: batch learning with smoothness regularization.

The controller is trained using:

- Global PID imitation loss.
- Local equilibrium-region imitation loss.
- Equilibrium control penalty enforcing `u_N(phi*) ≈ 0`.
- Local closed-loop stability checkpointing based on the spectral radius condition `rho(A_cl) < 1`.

#### Main hyperparameters

```matlab
cfg.activationType = "leakyrelu";

cfg.maxEpochs      = 120;
cfg.miniBatchSize  = 512;
cfg.lr             = 1e-3;
cfg.gradClip       = 5.0;

cfg.smoothLambda   = 0.10;   % use 0 for BL, 0.10 for BL_S

cfg.local.use        = true;
cfg.local.lambdaLoc  = 0.50;
cfg.local.lambdaEq   = 0.10;
cfg.local.batchSize  = 512;

cfg.cert.useFilter = true;
cfg.cert.checkFreq = 5;
cfg.cert.rhoTarget = 0.98;
cfg.cert.rhoAccept = 1.00;
```

#### Outputs

Depending on the value of `cfg.smoothLambda`, the script saves:

```text
trainedNN_CSTR_BL.mat
trainedNN_CSTR_BL_smooth.mat
```

---

### `03_train_cstr_ESS_FT.m`

This script trains the Enhanced Scheduled Sampling controller and then applies a fine-tuning stage.

During ESS training, the applied control action is a mixture of the PID teacher action and the neural-network action:

```math
u_{\mathrm{app}} =
p_{\mathrm{ANN}} u_N +
(1-p_{\mathrm{ANN}})u_T.
```

The value of `pANN` is increased progressively using an inverse-sigmoid schedule. The fine-tuning stage then keeps a high neural-network contribution and injects small control noise to improve robustness.

#### Main features

- Scheduled sampling between PID and neural-network control actions.
- Error-weighted imitation loss.
- Local equilibrium-region imitation loss.
- Equilibrium control penalty enforcing `u_N(phi*) ≈ 0`.
- Stability-oriented checkpoint selection.
- Fine-tuning with injected control-deviation noise.

#### Main hyperparameters

```matlab
cfg.activationType = "leakyrelu";

cfg.N_ess       = 120;
cfg.N_ft        = 60;
cfg.sigmoid_k   = 10;

cfg.pANN_ft     = 0.90;
cfg.sigma_ft    = 0.05;      % L/min deviation noise during fine-tuning

cfg.lr_early    = 1e-3;
cfg.lr_mid      = 5e-4;
cfg.lr_late     = 2e-4;
cfg.lr_ft       = 5e-5;

cfg.lambdaErr   = 0.15;
cfg.chunkSize   = 2000;
cfg.gradClip    = 5.0;

cfg.local.use        = true;
cfg.local.lambdaLoc  = 0.50;
cfg.local.lambdaEq   = 0.10;
cfg.local.batchSize  = 512;

cfg.cert.useFilter  = true;
cfg.cert.checkFreq  = 5;
cfg.cert.rhoTarget  = 0.98;
cfg.cert.rhoAccept  = 1.00;

cfg.cert.minPannESS = 0.85;
cfg.cert.minPannFT  = 0.85;
```

#### Learning-rate schedule

The ESS phase uses the following staged learning-rate schedule:

```matlab
if pANN < 0.40
    lr = 1e-3;
elseif pANN < 0.75
    lr = 5e-4;
else
    lr = 2e-4;
end
```

The fine-tuning stage uses:

```matlab
lr_ft = 5e-5;
```

#### Outputs

```text
trainedNN_CSTR_ESS.mat
trainedNN_CSTR_ESS_FT.mat
```

---

### `04_validate_cstr_cases_PID_vs_NN.m`

This script validates the PID teacher and the trained neural-network controllers in nonlinear CSTR simulations.

The evaluated controllers are:

- PID teacher.
- BL.
- BL_S.
- ESS.
- ESS_FT.

The simulation horizon is:

```matlab
simopt.t_end = 25.0;       % min
simopt.dt    = 0.12/60;    % 0.12 s in min
```

---

## Nonlinear validation cases

### Case 1

```matlab
CBr  = 1.2        at t = 3.0 min
dQ   = -0.10 Q0   at t = 12.5 min
dCAi = +0.05 CAi0 at t = 20.0 min
```

### Case 2

```matlab
CBr  = 1.0        at t = 3.0 min
dQ   = +0.10 Q0   at t = 12.5 min
dCAi = -0.05 CAi0 at t = 20.0 min
```

---

## Validation outputs

The nonlinear validation script generates:

```text
results_cstr_cases/
├── cstr_case_validation_results.mat
├── cstr_case_validation_metrics.csv
├── cstr_case_validation_table.tex
└── figures/
    └── *.png
```

The reported metrics include:

- Tracking error.
- Disturbance-rejection error.
- Total variation of the control action.
- Peak tracking error.
- Maximum manipulated-variable deviation.
- Saturation-related indicators.
- Operating-region displacement.

---

## Typical workflow

### Option 1: Use pretrained models

The simplest way to reproduce the nonlinear validation is to use the pretrained 6th-seed models in `models/`.

Run:

```matlab
run('04_validate_cstr_cases_PID_vs_NN.m')
```

This compares the PID teacher against the pretrained neural-network controllers in the two nonlinear validation cases.

---

### Option 2: Retrain the controllers

#### Train BL or BL_S

To train the BL or BL_S controller, run:

```matlab
run('02_train_cstr_BL.m')
```

To switch between BL and BL_S, modify:

```matlab
cfg.smoothLambda = 0;      % BL
cfg.smoothLambda = 0.10;   % BL_S
```

#### Train ESS and ESS_FT

To train the ESS and ESS_FT controllers, run:

```matlab
run('03_train_cstr_ESS_FT.m')
```

This script first trains the ESS controller and then performs the fine-tuning stage to obtain ESS_FT.

#### Validate all controllers

After training, run:

```matlab
run('04_validate_cstr_cases_PID_vs_NN.m')
```

---

## Controller variants

### BL

Standard batch-learning imitation of the PID teacher.

The loss mainly penalizes one-step differences between the neural-network output and the PID teacher action.

### BL_S

Smoothed batch-learning imitation.

This variant adds a smoothness penalty to reduce excessive control-action variations.

### ESS

Enhanced Scheduled Sampling.

The neural network is trained in a closed-loop-oriented way by progressively increasing the contribution of the neural-network action during training.

### ESS_FT

Fine-tuned ESS controller.

The ESS controller is further trained with a high neural-network contribution, fixed at `pANN = 0.90`, and small injected control-deviation noise with standard deviation `0.05 L/min`.

---

## Checkpoint selection

During training, candidate checkpoints are evaluated using:

- Validation MSE.
- Autonomous closed-loop imitation error.
- Equilibrium control output `u_N(phi*)`.
- Local closed-loop spectral radius `rho(A_cl)`.

A checkpoint is considered locally stable when:

```math
\rho(A_{cl}) < 1.
```

The training scripts monitor the spectral radius every 5 epochs:

```matlab
cfg.cert.checkFreq = 5;
```

The main acceptance threshold is:

```matlab
cfg.cert.rhoAccept = 1.00;
```

and the target value used for ranking is:

```matlab
cfg.cert.rhoTarget = 0.98;
```

For ESS and ESS_FT, candidate checkpoints are considered eligible only when the neural-network contribution is sufficiently high:

```matlab
cfg.cert.minPannESS = 0.85;
cfg.cert.minPannFT  = 0.85;
```

---

## Notes on standardization

The input scaler is fitted only on the global training dataset:

```matlab
scaler.mu    = mean(X_raw,1);
scaler.sigma = std(X_raw,0,1);
```

The same scaler is then used for:

- Training data.
- Validation data.
- Local certification-oriented data.
- Autonomous closed-loop simulations.

This avoids data leakage from the validation and local datasets into the global input normalization.

---

## Notes on local data

The local dataset is used to improve the neural-network behavior around the nominal operating point used for local certification.

The local regularization weights are:

```matlab
cfg.local.lambdaLoc = 0.50;
cfg.local.lambdaEq  = 0.10;
```

The equilibrium regressor is defined in raw variables as:

```matlab
phiEq_raw = zeros(prob.nn.nphi,1);
```

This corresponds to zero tracking error, zero delayed errors, zero integral-error channel, and zero previous control deviation in the nominal deviation formulation.

---

## Reproducibility note

The paper reports multi-seed results over six independent training seeds. This repository includes the representative trained models corresponding to the 6th seed used for the nonlinear case-study figures and validation routines.

---

## Citation

If you use this repository, please cite the associated paper:

```bibtex
@article{madrigal2026localinheriteddm,
  title   = {Post-Training Inherited Disk-Margin Certification for Local-Affine Neural Network Controllers Trained from PID Teachers},
  author  = {Madrigal, Sebasti\'an and coauthors},
  journal = {Under review},
  year    = {2026}
}
```

---

## Contact

For questions about the repository, please contact:

**Sebastián Madrigal**  
sebastian.madrigal@uab.cat
