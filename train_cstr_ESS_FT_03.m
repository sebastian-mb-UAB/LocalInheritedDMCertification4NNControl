clear; clc; close all;

%% ============================================================
%  03_train_cstr_ESS_FT.m
%
%  Enhanced Scheduled Sampling training for nonlinear CSTR PID imitation.
%
%  Conditions:
%    - ESS / FT imitation loss
%    - Error-weighted imitation loss
%    - Local equilibrium-region imitation loss
%    - Equilibrium control penalty u_N(phi*) ~= 0
%    - Local closed-loop stability checkpoint filter rho(A_cl)<1
%
%  During ESS training:
%      u_app = p_ANN*u_N + (1-p_ANN)*u_T
%
%  Outputs:
%      trainedNN_CSTR_ESS.mat
%      trainedNN_CSTR_ESS_FT.mat
% ============================================================

load('D_tr_T.mat',  'dataTr',   'prob');
load('D_val_T.mat', 'dataValT');

if ~isfile('D_loc_T.mat')
    error('D_loc_T.mat not found. Run the local teacher dataset generator first.');
end
load('D_loc_T.mat', 'dataLocT');

cstr   = prob.cstr;
ctrl   = prob.ctrl;
simopt = prob.simopt;
nncfg  = prob.nn;

%% ------------------------------------------------------------
%  Configuration
%% ------------------------------------------------------------
cfg.activationType = "leakyrelu";

cfg.N_ess       = 120;
cfg.N_ft        = 60;
cfg.sigmoid_k   = 10;

cfg.pANN_ft     = 0.90;
cfg.sigma_ft    = 0.05;      % L/min deviation noise during FT

cfg.lr_early    = 1e-3;
cfg.lr_mid      = 5e-4;
cfg.lr_late     = 2e-4;
cfg.lr_ft       = 5e-5;

cfg.lambdaErr   = 0.15;
cfg.chunkSize   = 2000;
cfg.gradClip    = 5.0;

cfg.valFreq     = 5;

% Local certification-oriented regularization
cfg.local.use        = true;
cfg.local.lambdaLoc  = 0.50;
cfg.local.lambdaEq   = 0.10;
cfg.local.batchSize  = 512;

% Local closed-loop stability checkpointing
cfg.cert.useFilter     = true;
cfg.cert.checkFreq     = 5;
cfg.cert.rhoTarget     = 0.98;
cfg.cert.rhoAccept     = 1.00;

% ESS checkpoints are only considered stable candidates when the NN
% contribution is sufficiently high.
cfg.cert.minPannESS    = 0.85;
cfg.cert.minPannFT     = 0.85;

fprintf('\n============================================================\n');
fprintf('        CERTIFICATION-ORIENTED ESS / ESS_FT TRAINING\n');
fprintf('============================================================\n');
fprintf('Local loss enabled       : %d\n', cfg.local.use);
fprintf('lambdaLoc                : %.4f\n', cfg.local.lambdaLoc);
fprintf('lambdaEq                 : %.4f\n', cfg.local.lambdaEq);
fprintf('rhoTarget                : %.4f\n', cfg.cert.rhoTarget);
fprintf('rhoAccept                : %.4f\n', cfg.cert.rhoAccept);
fprintf('minPannESS               : %.4f\n', cfg.cert.minPannESS);
fprintf('minPannFT                : %.4f\n', cfg.cert.minPannFT);

%% ------------------------------------------------------------
%  Local dataset extraction
%% ------------------------------------------------------------
if isfield(dataLocT, 'phi_aug')
    X_loc_raw = dataLocT.phi_aug;
else
    X_loc_raw = dataLocT.phi;
end

if isfield(dataLocT, 'uT_aug')
    Y_loc_raw = dataLocT.uT_aug;
else
    Y_loc_raw = dataLocT.uT;
end

fprintf('Training samples        : %d\n', size(dataTr.phi,1));
fprintf('Validation samples      : %d\n', size(dataValT.phi,1));
fprintf('Local augmented samples : %d\n', size(X_loc_raw,1));

%% ------------------------------------------------------------
%  Scaler from global teacher training data
%% ------------------------------------------------------------
scaler.mu    = mean(dataTr.phi,1);
scaler.sigma = std(dataTr.phi,0,1);
scaler.sigma(scaler.sigma < 1e-8) = 1;

mu_vec    = scaler.mu(:);
sigma_vec = scaler.sigma(:);

X_val_sc = (dataValT.phi - scaler.mu)./scaler.sigma;
X_loc_sc = (X_loc_raw    - scaler.mu)./scaler.sigma;

% Equilibrium regressor phi*.
% For the nominal deviation formulation:
%   e = 0, past errors = 0, integral-error channel = 0, u_{k-1} = 0.
phiEq_raw = zeros(nncfg.nphi,1);
phiEq_sc  = (phiEq_raw.' - scaler.mu)./scaler.sigma;
phiEq_dl  = dlarray(phiEq_sc.','CB');

localPack = struct();
localPack.use       = cfg.local.use;
localPack.X_loc_sc  = X_loc_sc;
localPack.Y_loc_raw = Y_loc_raw(:);
localPack.Nloc      = size(X_loc_sc,1);
localPack.phiEq_dl  = phiEq_dl;

%% ------------------------------------------------------------
%  Initialize network
%% ------------------------------------------------------------
fprintf('\nInitializing ESS from random weights\n');
net = CSTR_NNTrainingUtils.buildMLP(nncfg.nphi, cfg.activationType);

%% ------------------------------------------------------------
%  Adam setup
%% ------------------------------------------------------------
averageGrad   = [];
averageSqGrad = [];
beta1 = 0.9;
beta2 = 0.999;
epsAdam = 1e-8;
globalIter = 0;

N_total = cfg.N_ess + cfg.N_ft;

loss_history       = nan(N_total,1);
val_mse_history    = nan(N_total,1);
pANN_history       = nan(N_total,1);
lr_history         = nan(N_total,1);
delta_history      = nan(N_total,1);
rhoAcl_history     = nan(N_total,1);
isSchur_history    = false(N_total,1);
eqOutput_history   = nan(N_total,1);
eligible_history   = false(N_total,1);

%% ------------------------------------------------------------
%  Checkpoint structures
%% ------------------------------------------------------------
bestESS = initBestStruct("ESS");
bestFT  = initBestStruct("FT");

stabOpts = CSTR_LocalStabilityUtils.defaultOptions(prob);

%% ============================================================
%  PHASE 1: ESS
%% ============================================================
fprintf('\n====== PHASE 1: ESS ======\n');

for epoch = 1:cfg.N_ess

    pANN = inverseSigmoidSchedule(epoch, cfg.N_ess, cfg.sigmoid_k);
    pANN_history(epoch) = pANN;

    if pANN < 0.40
        lr = cfg.lr_early;
    elseif pANN < 0.75
        lr = cfg.lr_mid;
    else
        lr = cfg.lr_late;
    end
    lr_history(epoch) = lr;

    [net, avgLoss, globalIter, averageGrad, averageSqGrad] = ...
        runOneMixedEpisode(net, prob, dataTr.refCB, scaler, localPack, ...
        pANN, 0.0, lr, cfg, globalIter, averageGrad, averageSqGrad, ...
        beta1, beta2, epsAdam);

    loss_history(epoch) = avgLoss;

    uN_val = extractdata(forward(net, dlarray(X_val_sc.','CB'))).';
    val_mse_history(epoch) = mean((uN_val(:)-dataValT.uT).^2);

    uEq = double(extractdata(forward(net, phiEq_dl)));
    eqOutput_history(epoch) = uEq;

    doCheck = cfg.cert.useFilter && ...
        (epoch == 1 || mod(epoch,cfg.cert.checkFreq)==0 || epoch == cfg.N_ess);

    if doCheck
        metrics = evaluateCheckpoint(net, scaler, prob, dataValT, stabOpts, phiEq_dl);

        delta_history(epoch)    = metrics.deltaInf;
        rhoAcl_history(epoch)   = metrics.rhoAcl;
        isSchur_history(epoch)  = metrics.isSchur;
        eqOutput_history(epoch) = metrics.uEq;

        eligible = pANN >= cfg.cert.minPannESS;
        eligible_history(epoch) = eligible;

        metrics.phase           = "ESS";
        metrics.epoch           = epoch;
        metrics.pANN            = pANN;
        metrics.valMSE          = val_mse_history(epoch);
        metrics.eligible        = eligible;
        metrics.stableCandidate = eligible && metrics.isSchur && metrics.rhoAcl < cfg.cert.rhoAccept;

        bestESS = maybeUpdateBestCheckpoint(bestESS, metrics);

        if bestESS.epoch == epoch
            fprintf('  *** New best ESS checkpoint | Ep=%d | eligible=%d | ValMSE=%.4e | rho=%.6f | Schur=%d | DeltaInf=%.4e ***\n', ...
                epoch, eligible, metrics.valMSE, metrics.rhoAcl, metrics.isSchur, metrics.deltaInf);
        end

        fprintf('ESS %3d/%d | pANN=%.3f | loss=%.4e | valMSE=%.4e | rho=%.6f | Schur=%d | eligible=%d | DeltaInf=%.4e\n', ...
            epoch, cfg.N_ess, pANN, loss_history(epoch), ...
            val_mse_history(epoch), metrics.rhoAcl, metrics.isSchur, ...
            eligible, metrics.deltaInf);
    else
        fprintf('ESS %3d/%d | pANN=%.3f | loss=%.4e | valMSE=%.4e | uEq=%.3e\n', ...
            epoch, cfg.N_ess, pANN, loss_history(epoch), ...
            val_mse_history(epoch), uEq);
    end
end

%% ------------------------------------------------------------
%  Select and save best ESS model
%% ------------------------------------------------------------
if isempty(bestESS.net)
    warning('No ESS checkpoint was selected. Using last ESS network.');
    bestESS.net = net;
    bestESS.epoch = cfg.N_ess;
    bestESS.valMSE = val_mse_history(cfg.N_ess);
    bestESS.phase = "ESS_LAST";
end

netESS = bestESS.net;
net    = netESS;

nnFcnESS = @(phi_k,uTvirt_k,k) ...
    CSTR_NNTrainingUtils.evalDlnet(netESS, scaler, phi_k);

dataValN_ESS = CSTR_NNTrainingUtils.simulateNNClosedLoop(prob, dataValT.refCB, nnFcnESS);

Y_tr_ESS  = CSTR_NNTrainingUtils.batchPredictDlnet(netESS, scaler, dataTr.phi);
Y_val_ESS = CSTR_NNTrainingUtils.batchPredictDlnet(netESS, scaler, dataValT.phi);
Y_loc_ESS = CSTR_NNTrainingUtils.batchPredictDlnet(netESS, scaler, X_loc_raw);

mse_tr_ESS  = mean((Y_tr_ESS  - dataTr.uT).^2);
mae_tr_ESS  = mean(abs(Y_tr_ESS  - dataTr.uT));
mse_val_ESS = mean((Y_val_ESS - dataValT.uT).^2);
mae_val_ESS = mean(abs(Y_val_ESS - dataValT.uT));
mse_loc_ESS = mean((Y_loc_ESS - Y_loc_raw).^2);
mae_loc_ESS = mean(abs(Y_loc_ESS - Y_loc_raw));

DeltaInf_val_ESS = CSTR_NNTrainingUtils.seqSupNorm(dataValN_ESS.Delta);
maxErr_val_ESS   = max(abs(dataValN_ESS.e));
maxUdev_val_ESS  = max(abs(dataValN_ESS.Q - prob.cstr.Q0));

try
    [rhoAcl_ESS, rhoInfo_ESS] = CSTR_LocalStabilityUtils.computeRhoAcl(netESS, scaler, prob, stabOpts);
catch ME
    warning(ME.identifier,'Final ESS rho(Acl) failed: %s', ME.message);
    rhoAcl_ESS = inf;
    rhoInfo_ESS = struct();
end

netSaved = netESS;
scalerSaved = scaler;

nnFcn = @(phi_k,uTvirt_k,k) ...
    CSTR_NNTrainingUtils.evalDlnet(netSaved, scalerSaved, phi_k);

save('trainedNN_CSTR_ESS.mat', 'netESS', 'net', 'scaler', 'nnFcn', ...
    'cfg', 'bestESS', ...
    'mse_tr_ESS', 'mae_tr_ESS', ...
    'mse_val_ESS', 'mae_val_ESS', ...
    'mse_loc_ESS', 'mae_loc_ESS', ...
    'rhoAcl_ESS', 'rhoInfo_ESS', ...
    'DeltaInf_val_ESS', 'maxErr_val_ESS', 'maxUdev_val_ESS', ...
    'loss_history', 'val_mse_history', 'pANN_history', ...
    'lr_history', 'delta_history', 'rhoAcl_history', ...
    'isSchur_history', 'eqOutput_history', 'eligible_history', ...
    'dataValN_ESS');

fprintf('\nSaved: trainedNN_CSTR_ESS.mat\n');
fprintf('Selected ESS epoch : %d\n', bestESS.epoch);
fprintf('ESS Val MSE        : %.6e\n', mse_val_ESS);
fprintf('ESS rho(Acl)       : %.10f\n', rhoAcl_ESS);
fprintf('ESS Schur stable   : %d\n', rhoAcl_ESS < 1);

%% ============================================================
%  PHASE 2: FINE-TUNING
%% ============================================================
fprintf('\n====== PHASE 2: ESS Fine-Tuning ======\n');

net = netESS;
averageGrad   = [];
averageSqGrad = [];

for ft = 1:cfg.N_ft

    epoch = cfg.N_ess + ft;

    pANN = cfg.pANN_ft;
    lr   = cfg.lr_ft;

    pANN_history(epoch) = pANN;
    lr_history(epoch)   = lr;

    [net, avgLoss, globalIter, averageGrad, averageSqGrad] = ...
        runOneMixedEpisode(net, prob, dataTr.refCB, scaler, localPack, ...
        pANN, cfg.sigma_ft, lr, cfg, globalIter, averageGrad, averageSqGrad, ...
        beta1, beta2, epsAdam);

    loss_history(epoch) = avgLoss;

    uN_val = extractdata(forward(net, dlarray(X_val_sc.','CB'))).';
    val_mse_history(epoch) = mean((uN_val(:)-dataValT.uT).^2);

    uEq = double(extractdata(forward(net, phiEq_dl)));
    eqOutput_history(epoch) = uEq;

    doCheck = cfg.cert.useFilter && ...
        (ft == 1 || mod(ft,cfg.cert.checkFreq)==0 || ft == cfg.N_ft);

    if doCheck
        metrics = evaluateCheckpoint(net, scaler, prob, dataValT, stabOpts, phiEq_dl);

        delta_history(epoch)    = metrics.deltaInf;
        rhoAcl_history(epoch)   = metrics.rhoAcl;
        isSchur_history(epoch)  = metrics.isSchur;
        eqOutput_history(epoch) = metrics.uEq;

        eligible = pANN >= cfg.cert.minPannFT;
        eligible_history(epoch) = eligible;

        metrics.phase           = "FT";
        metrics.epoch           = epoch;
        metrics.pANN            = pANN;
        metrics.valMSE          = val_mse_history(epoch);
        metrics.eligible        = eligible;
        metrics.stableCandidate = eligible && metrics.isSchur && metrics.rhoAcl < cfg.cert.rhoAccept;

        bestFT = maybeUpdateBestCheckpoint(bestFT, metrics);

        if bestFT.epoch == epoch
            fprintf('  *** New best FT checkpoint | Ep=%d | eligible=%d | ValMSE=%.4e | rho=%.6f | Schur=%d | DeltaInf=%.4e ***\n', ...
                epoch, eligible, metrics.valMSE, metrics.rhoAcl, metrics.isSchur, metrics.deltaInf);
        end

        fprintf('FT  %3d/%d | pANN=%.3f | loss=%.4e | valMSE=%.4e | rho=%.6f | Schur=%d | eligible=%d | DeltaInf=%.4e\n', ...
            ft, cfg.N_ft, pANN, loss_history(epoch), ...
            val_mse_history(epoch), metrics.rhoAcl, metrics.isSchur, ...
            eligible, metrics.deltaInf);
    else
        fprintf('FT  %3d/%d | pANN=%.3f | loss=%.4e | valMSE=%.4e | uEq=%.3e\n', ...
            ft, cfg.N_ft, pANN, loss_history(epoch), ...
            val_mse_history(epoch), uEq);
    end
end

%% ------------------------------------------------------------
%  Select and save best FT model
%% ------------------------------------------------------------
if isempty(bestFT.net)
    warning('No FT checkpoint was selected. Using final FT network.');
    bestFT.net = net;
    bestFT.epoch = N_total;
    bestFT.valMSE = val_mse_history(end);
    bestFT.phase = "FT_LAST";
end

netFT = bestFT.net;
net   = netFT;

nnFcnFT = @(phi_k,uTvirt_k,k) ...
    CSTR_NNTrainingUtils.evalDlnet(netFT, scaler, phi_k);

dataValN_FT = CSTR_NNTrainingUtils.simulateNNClosedLoop(prob, dataValT.refCB, nnFcnFT);

Y_tr_FT  = CSTR_NNTrainingUtils.batchPredictDlnet(netFT, scaler, dataTr.phi);
Y_val_FT = CSTR_NNTrainingUtils.batchPredictDlnet(netFT, scaler, dataValT.phi);
Y_loc_FT = CSTR_NNTrainingUtils.batchPredictDlnet(netFT, scaler, X_loc_raw);

mse_tr_FT  = mean((Y_tr_FT  - dataTr.uT).^2);
mae_tr_FT  = mean(abs(Y_tr_FT  - dataTr.uT));
mse_val_FT = mean((Y_val_FT - dataValT.uT).^2);
mae_val_FT = mean(abs(Y_val_FT - dataValT.uT));
mse_loc_FT = mean((Y_loc_FT - Y_loc_raw).^2);
mae_loc_FT = mean(abs(Y_loc_FT - Y_loc_raw));

DeltaInf_val_FT = CSTR_NNTrainingUtils.seqSupNorm(dataValN_FT.Delta);
maxErr_val_FT   = max(abs(dataValN_FT.e));
maxUdev_val_FT  = max(abs(dataValN_FT.Q - prob.cstr.Q0));

try
    [rhoAcl_FT, rhoInfo_FT] = CSTR_LocalStabilityUtils.computeRhoAcl(netFT, scaler, prob, stabOpts);
catch ME
    warning(ME.identifier,'Final FT rho(Acl) failed: %s', ME.message);
    rhoAcl_FT = inf;
    rhoInfo_FT = struct();
end

netSaved = netFT;
scalerSaved = scaler;

nnFcn = @(phi_k,uTvirt_k,k) ...
    CSTR_NNTrainingUtils.evalDlnet(netSaved, scalerSaved, phi_k);

save('trainedNN_CSTR_ESS_FT.mat', 'netFT', 'net', 'scaler', 'nnFcn', ...
    'cfg', 'bestESS', 'bestFT', ...
    'mse_tr_ESS', 'mae_tr_ESS', ...
    'mse_val_ESS', 'mae_val_ESS', ...
    'mse_loc_ESS', 'mae_loc_ESS', ...
    'rhoAcl_ESS', 'rhoInfo_ESS', ...
    'DeltaInf_val_ESS', 'maxErr_val_ESS', 'maxUdev_val_ESS', ...
    'mse_tr_FT', 'mae_tr_FT', ...
    'mse_val_FT', 'mae_val_FT', ...
    'mse_loc_FT', 'mae_loc_FT', ...
    'rhoAcl_FT', 'rhoInfo_FT', ...
    'DeltaInf_val_FT', 'maxErr_val_FT', 'maxUdev_val_FT', ...
    'loss_history', 'val_mse_history', 'pANN_history', ...
    'lr_history', 'delta_history', 'rhoAcl_history', ...
    'isSchur_history', 'eqOutput_history', 'eligible_history', ...
    'dataValN_ESS', 'dataValN_FT');

fprintf('\nSaved: trainedNN_CSTR_ESS_FT.mat\n');
fprintf('Selected FT epoch  : %d\n', bestFT.epoch);
fprintf('FT Val MSE         : %.6e\n', mse_val_FT);
fprintf('FT rho(Acl)        : %.10f\n', rhoAcl_FT);
fprintf('FT Schur stable    : %d\n', rhoAcl_FT < 1);

%% ------------------------------------------------------------
%  Complete plots after ESS / ESS_FT training
%% ------------------------------------------------------------

figDir = fullfile('results_training_figures', 'ESS_FT');
if ~exist(figDir, 'dir')
    mkdir(figDir);
end

%% ============================================================
%  Prepare additional autonomous simulations for plots
%% ============================================================

nnFcnESS_plot = @(phi_k,uTvirt_k,k) ...
    CSTR_NNTrainingUtils.evalDlnet(netESS, scaler, phi_k);

nnFcnFT_plot = @(phi_k,uTvirt_k,k) ...
    CSTR_NNTrainingUtils.evalDlnet(netFT, scaler, phi_k);

dataTrN_ESS = CSTR_NNTrainingUtils.simulateNNClosedLoop(prob, dataTr.refCB, nnFcnESS_plot);
dataTrN_FT  = CSTR_NNTrainingUtils.simulateNNClosedLoop(prob, dataTr.refCB, nnFcnFT_plot);

%% Base local dataset predictions for plotting
if isfield(dataLocT, 'phi')
    Y_loc_base_ESS = CSTR_NNTrainingUtils.batchPredictDlnet(netESS, scaler, dataLocT.phi);
    Y_loc_base_FT  = CSTR_NNTrainingUtils.batchPredictDlnet(netFT,  scaler, dataLocT.phi);
else
    Y_loc_base_ESS = [];
    Y_loc_base_FT  = [];
end

%% ============================================================
%  1) ESS / FT training curves
%% ============================================================

figure('Name','ESS / ESS_FT - Training curves', ...
       'Position',[50 50 1500 850]);

tiledlayout(4,1,'TileSpacing','compact','Padding','compact');

nexttile;
plot(loss_history, 'LineWidth',1.3); hold on;
plot(val_mse_history, '--', 'LineWidth',1.3);
xline(cfg.N_ess, 'k--', 'FT start', 'LineWidth',1.1);
if isfinite(bestESS.epoch)
    xline(bestESS.epoch, 'b:', 'best ESS', 'LineWidth',1.1);
end
if isfinite(bestFT.epoch)
    xline(bestFT.epoch, 'r:', 'best FT', 'LineWidth',1.1);
end
grid on;
xlabel('Epoch');
ylabel('Loss / MSE');
title('ESS / ESS_{FT}: training and validation loss');
legend('Training loss','Validation MSE','FT start','Best ESS','Best FT','Location','best');

nexttile;
semilogy(loss_history, 'LineWidth',1.3); hold on;
semilogy(val_mse_history, '--', 'LineWidth',1.3);
xline(cfg.N_ess, 'k--', 'FT start', 'LineWidth',1.1);
if isfinite(bestESS.epoch)
    xline(bestESS.epoch, 'b:', 'best ESS', 'LineWidth',1.1);
end
if isfinite(bestFT.epoch)
    xline(bestFT.epoch, 'r:', 'best FT', 'LineWidth',1.1);
end
grid on;
xlabel('Epoch');
ylabel('Loss / MSE');
title('ESS / ESS_{FT}: loss curves in logarithmic scale');
legend('Training loss','Validation MSE','FT start','Best ESS','Best FT','Location','best');

nexttile;
plot(pANN_history, 'LineWidth',1.3);
xline(cfg.N_ess, 'k--', 'FT start', 'LineWidth',1.1);
if isfinite(bestESS.epoch)
    xline(bestESS.epoch, 'b:', 'best ESS', 'LineWidth',1.1);
end
if isfinite(bestFT.epoch)
    xline(bestFT.epoch, 'r:', 'best FT', 'LineWidth',1.1);
end
grid on;
xlabel('Epoch');
ylabel('p_{ANN}');
title('Scheduled-sampling probability');

nexttile;
plot(lr_history, 'LineWidth',1.3);
xline(cfg.N_ess, 'k--', 'FT start', 'LineWidth',1.1);
if isfinite(bestESS.epoch)
    xline(bestESS.epoch, 'b:', 'best ESS', 'LineWidth',1.1);
end
if isfinite(bestFT.epoch)
    xline(bestFT.epoch, 'r:', 'best FT', 'LineWidth',1.1);
end
grid on;
xlabel('Epoch');
ylabel('Learning rate');
title('Learning-rate schedule');

saveas(gcf, fullfile(figDir, 'ESS_FT_training_curves_complete.png'));

%% ============================================================
%  2) Local stability and checkpoint diagnostics
%% ============================================================

figure('Name','ESS / ESS_FT - Local stability diagnostics', ...
       'Position',[80 80 1500 900]);

tiledlayout(4,1,'TileSpacing','compact','Padding','compact');

nexttile;
plot(rhoAcl_history, 'o-', 'LineWidth',1.2, 'MarkerSize',4); hold on;
yline(1.0, 'r--', 'Schur boundary', 'LineWidth',1.2);
yline(cfg.cert.rhoTarget, 'k:', '\rho target', 'LineWidth',1.2);
xline(cfg.N_ess, 'k--', 'FT start', 'LineWidth',1.1);
if isfinite(bestESS.epoch)
    xline(bestESS.epoch, 'b:', 'best ESS', 'LineWidth',1.1);
end
if isfinite(bestFT.epoch)
    xline(bestFT.epoch, 'r:', 'best FT', 'LineWidth',1.1);
end
grid on;
xlabel('Epoch');
ylabel('\rho(A_{cl})');
title('Local closed-loop spectral radius');
legend('\rho(A_{cl})','Schur boundary','Target','FT start','Best ESS','Best FT','Location','best');

nexttile;
stairs(double(isSchur_history), 'LineWidth',1.3); hold on;
stairs(double(eligible_history), '--', 'LineWidth',1.3);
xline(cfg.N_ess, 'k--', 'FT start', 'LineWidth',1.1);
ylim([-0.1 1.1]);
grid on;
xlabel('Epoch');
ylabel('Flag');
title('Schur stability and checkpoint eligibility flags');
legend('Schur stable','Eligible checkpoint','Location','best');

nexttile;
plot(delta_history, 'o-', 'LineWidth',1.2, 'MarkerSize',4);
xline(cfg.N_ess, 'k--', 'FT start', 'LineWidth',1.1);
if isfinite(bestESS.epoch)
    xline(bestESS.epoch, 'b:', 'best ESS', 'LineWidth',1.1);
end
if isfinite(bestFT.epoch)
    xline(bestFT.epoch, 'r:', 'best FT', 'LineWidth',1.1);
end
grid on;
xlabel('Epoch');
ylabel('\Delta_\infty');
title('Autonomous dynamic imitation error at monitored epochs');

nexttile;
plot(eqOutput_history, 'LineWidth',1.3); hold on;
yline(0, 'k:');
xline(cfg.N_ess, 'k--', 'FT start', 'LineWidth',1.1);
if isfinite(bestESS.epoch)
    xline(bestESS.epoch, 'b:', 'best ESS', 'LineWidth',1.1);
end
if isfinite(bestFT.epoch)
    xline(bestFT.epoch, 'r:', 'best FT', 'LineWidth',1.1);
end
grid on;
xlabel('Epoch');
ylabel('u_N(\phi^\star)');
title('Equilibrium control output');

saveas(gcf, fullfile(figDir, 'ESS_FT_local_stability_diagnostics.png'));

%% ============================================================
%  3) PID Teacher vs ESS / ESS_FT on TRAIN dataset: one-step imitation
%% ============================================================

figure('Name','ESS / ESS_FT - Train dataset one-step imitation', ...
       'Position',[100 100 1500 900]);

tiledlayout(4,1,'TileSpacing','compact','Padding','compact');

nexttile;
plot(dataTr.t, dataTr.uT, 'LineWidth',1.25); hold on;
plot(dataTr.t, Y_tr_ESS, '--', 'LineWidth',1.15);
plot(dataTr.t, Y_tr_FT, '-.', 'LineWidth',1.15);
grid on;
xlabel('Time [min]');
ylabel('u [L/min dev.]');
title('Train dataset: one-step imitation of PID teacher');
legend('PID teacher u^T','ESS u^N','ESS_{FT} u^N','Location','best');

nexttile;
plot(dataTr.t, Y_tr_ESS - dataTr.uT, 'LineWidth',1.15); hold on;
plot(dataTr.t, Y_tr_FT  - dataTr.uT, '--', 'LineWidth',1.15);
grid on;
xlabel('Time [min]');
ylabel('u^N-u^T');
title('Train dataset: one-step imitation error');
legend('ESS error','ESS_{FT} error','Location','best');

nexttile;
plot(dataTr.t, dataTr.refCB, 'k--', 'LineWidth',1.0); hold on;
plot(dataTr.t, dataTr.CB, 'LineWidth',1.2);
grid on;
xlabel('Time [min]');
ylabel('C_B [gmol/L]');
title('Train dataset generated by PID teacher');
legend('C_{B,r}','C_B','Location','best');

nexttile;
plot(dataTr.t, dataTr.Q, 'LineWidth',1.2); hold on;
yline(prob.cstr.Q0, 'k:', 'Q_0');
yline(prob.cstr.Q0*0.90, ':', 'Q_0 - 10%');
yline(prob.cstr.Q0*1.10, ':', 'Q_0 + 10%');
grid on;
xlabel('Time [min]');
ylabel('Q [L/min]');
title('Train dataset: manipulated variable under PID teacher');
legend('Q','Q_0','Q_0 - 10%','Q_0 + 10%','Location','best');

saveas(gcf, fullfile(figDir, 'ESS_FT_train_dataset_onestep.png'));

%% ============================================================
%  4) PID Teacher vs ESS / ESS_FT on VALIDATION dataset: one-step imitation
%% ============================================================

figure('Name','ESS / ESS_FT - Validation dataset one-step imitation', ...
       'Position',[120 120 1500 900]);

tiledlayout(4,1,'TileSpacing','compact','Padding','compact');

nexttile;
plot(dataValT.t, dataValT.uT, 'LineWidth',1.25); hold on;
plot(dataValT.t, Y_val_ESS, '--', 'LineWidth',1.15);
plot(dataValT.t, Y_val_FT, '-.', 'LineWidth',1.15);
grid on;
xlabel('Time [min]');
ylabel('u [L/min dev.]');
title('Validation dataset: one-step imitation of PID teacher');
legend('PID teacher u^T','ESS u^N','ESS_{FT} u^N','Location','best');

nexttile;
plot(dataValT.t, Y_val_ESS - dataValT.uT, 'LineWidth',1.15); hold on;
plot(dataValT.t, Y_val_FT  - dataValT.uT, '--', 'LineWidth',1.15);
grid on;
xlabel('Time [min]');
ylabel('u^N-u^T');
title('Validation dataset: one-step imitation error');
legend('ESS error','ESS_{FT} error','Location','best');

nexttile;
plot(dataValT.t, dataValT.refCB, 'k--', 'LineWidth',1.0); hold on;
plot(dataValT.t, dataValT.CB, 'LineWidth',1.2);
grid on;
xlabel('Time [min]');
ylabel('C_B [gmol/L]');
title('Validation dataset generated by PID teacher');
legend('C_{B,r}','C_B','Location','best');

nexttile;
plot(dataValT.t, dataValT.Q, 'LineWidth',1.2); hold on;
yline(prob.cstr.Q0, 'k:', 'Q_0');
yline(prob.cstr.Q0*0.90, ':', 'Q_0 - 10%');
yline(prob.cstr.Q0*1.10, ':', 'Q_0 + 10%');
grid on;
xlabel('Time [min]');
ylabel('Q [L/min]');
title('Validation dataset: manipulated variable under PID teacher');
legend('Q','Q_0','Q_0 - 10%','Q_0 + 10%','Location','best');

saveas(gcf, fullfile(figDir, 'ESS_FT_validation_dataset_onestep.png'));

%% ============================================================
%  5) Local dataset imitation
%% ============================================================

if isfield(dataLocT, 'phi') && isfield(dataLocT, 'uT')
    figure('Name','ESS / ESS_FT - Local dataset imitation', ...
           'Position',[140 140 1500 900]);

    tiledlayout(4,1,'TileSpacing','compact','Padding','compact');

    nexttile;
    plot(dataLocT.t, dataLocT.uT, 'LineWidth',1.25); hold on;
    plot(dataLocT.t, Y_loc_base_ESS, '--', 'LineWidth',1.15);
    plot(dataLocT.t, Y_loc_base_FT, '-.', 'LineWidth',1.15);
    grid on;
    xlabel('Time [min]');
    ylabel('u [L/min dev.]');
    title('Local dataset: one-step imitation near equilibrium');
    legend('PID teacher u^T_{loc}','ESS u^N','ESS_{FT} u^N','Location','best');

    nexttile;
    plot(dataLocT.t, Y_loc_base_ESS - dataLocT.uT, 'LineWidth',1.15); hold on;
    plot(dataLocT.t, Y_loc_base_FT  - dataLocT.uT, '--', 'LineWidth',1.15);
    grid on;
    xlabel('Time [min]');
    ylabel('u^N-u^T');
    title('Local dataset: one-step imitation error');
    legend('ESS error','ESS_{FT} error','Location','best');

    nexttile;
    plot(dataLocT.t, dataLocT.refCB, 'k--', 'LineWidth',1.0); hold on;
    plot(dataLocT.t, dataLocT.CB, 'LineWidth',1.2);
    grid on;
    xlabel('Time [min]');
    ylabel('C_B [gmol/L]');
    title('Local trajectory generated by PID teacher');
    legend('C_{B,r}','C_B','Location','best');

    nexttile;
    plot(dataLocT.t, dataLocT.Q, 'LineWidth',1.2); hold on;
    yline(prob.cstr.Q0, 'k:', 'Q_0');
    grid on;
    xlabel('Time [min]');
    ylabel('Q [L/min]');
    title('Local manipulated variable under PID teacher');
    legend('Q','Q_0','Location','best');

    saveas(gcf, fullfile(figDir, 'ESS_FT_local_dataset_imitation.png'));
end

%% ============================================================
%  6) Autonomous closed-loop validation on TRAIN reference
%% ============================================================

figure('Name','ESS / ESS_FT - Autonomous train reference', ...
       'Position',[160 160 1500 950]);

tiledlayout(4,1,'TileSpacing','compact','Padding','compact');

nexttile;
plot(dataTr.t, dataTr.refCB, 'k--', 'LineWidth',1.0); hold on;
plot(dataTr.t, dataTr.CB, 'LineWidth',1.25);
plot(dataTrN_ESS.t, dataTrN_ESS.CB, '--', 'LineWidth',1.2);
plot(dataTrN_FT.t, dataTrN_FT.CB, '-.', 'LineWidth',1.2);
grid on;
xlabel('Time [min]');
ylabel('C_B [gmol/L]');
title('Autonomous closed-loop response on training reference');
legend('C_{B,r}','PID teacher','ESS','ESS_{FT}','Location','best');

nexttile;
plot(dataTr.t, dataTr.e, 'LineWidth',1.2); hold on;
plot(dataTrN_ESS.t, dataTrN_ESS.e, '--', 'LineWidth',1.2);
plot(dataTrN_FT.t, dataTrN_FT.e, '-.', 'LineWidth',1.2);
grid on;
xlabel('Time [min]');
ylabel('e [gmol/L]');
title('Tracking error on training reference');
legend('PID teacher','ESS','ESS_{FT}','Location','best');

nexttile;
plot(dataTr.t, dataTr.Q, 'LineWidth',1.2); hold on;
plot(dataTrN_ESS.t, dataTrN_ESS.Q, '--', 'LineWidth',1.2);
plot(dataTrN_FT.t, dataTrN_FT.Q, '-.', 'LineWidth',1.2);
yline(prob.cstr.Q0, 'k:', 'Q_0');
grid on;
xlabel('Time [min]');
ylabel('Q [L/min]');
title('Manipulated variable on training reference');
legend('PID teacher','ESS','ESS_{FT}','Q_0','Location','best');

nexttile;
plot(dataTrN_ESS.t, dataTrN_ESS.Delta, 'LineWidth',1.2); hold on;
plot(dataTrN_FT.t, dataTrN_FT.Delta, '--', 'LineWidth',1.2);
grid on;
xlabel('Time [min]');
ylabel('\Delta = u^N-u^{T,v}');
title('Dynamic imitation error on training reference');
legend('ESS','ESS_{FT}','Location','best');

saveas(gcf, fullfile(figDir, 'ESS_FT_autonomous_train_reference.png'));

%% ============================================================
%  7) Autonomous closed-loop validation on VALIDATION reference
%% ============================================================

figure('Name','ESS / ESS_FT - Autonomous validation reference', ...
       'Position',[180 180 1500 950]);

tiledlayout(4,1,'TileSpacing','compact','Padding','compact');

nexttile;
plot(dataValT.t, dataValT.refCB, 'k--', 'LineWidth',1.0); hold on;
plot(dataValT.t, dataValT.CB, 'LineWidth',1.25);
plot(dataValN_ESS.t, dataValN_ESS.CB, '--', 'LineWidth',1.2);
plot(dataValN_FT.t, dataValN_FT.CB, '-.', 'LineWidth',1.2);
grid on;
xlabel('Time [min]');
ylabel('C_B [gmol/L]');
title('Autonomous closed-loop response on validation reference');
legend('C_{B,r}','PID teacher','ESS','ESS_{FT}','Location','best');

nexttile;
plot(dataValT.t, dataValT.e, 'LineWidth',1.2); hold on;
plot(dataValN_ESS.t, dataValN_ESS.e, '--', 'LineWidth',1.2);
plot(dataValN_FT.t, dataValN_FT.e, '-.', 'LineWidth',1.2);
grid on;
xlabel('Time [min]');
ylabel('e [gmol/L]');
title('Tracking error on validation reference');
legend('PID teacher','ESS','ESS_{FT}','Location','best');

nexttile;
plot(dataValT.t, dataValT.Q, 'LineWidth',1.2); hold on;
plot(dataValN_ESS.t, dataValN_ESS.Q, '--', 'LineWidth',1.2);
plot(dataValN_FT.t, dataValN_FT.Q, '-.', 'LineWidth',1.2);
yline(prob.cstr.Q0, 'k:', 'Q_0');
grid on;
xlabel('Time [min]');
ylabel('Q [L/min]');
title('Manipulated variable on validation reference');
legend('PID teacher','ESS','ESS_{FT}','Q_0','Location','best');

nexttile;
plot(dataValN_ESS.t, dataValN_ESS.Delta, 'LineWidth',1.2); hold on;
plot(dataValN_FT.t, dataValN_FT.Delta, '--', 'LineWidth',1.2);
grid on;
xlabel('Time [min]');
ylabel('\Delta = u^N-u^{T,v}');
title('Dynamic imitation error on validation reference');
legend('ESS','ESS_{FT}','Location','best');

saveas(gcf, fullfile(figDir, 'ESS_FT_autonomous_validation_reference.png'));

%% ============================================================
%  8) Autonomous validation control signals
%% ============================================================

figure('Name','ESS / ESS_FT - Autonomous validation control signals', ...
       'Position',[200 200 1500 850]);

tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

nexttile;
plot(dataValN_ESS.t, dataValN_ESS.uTv, 'LineWidth',1.2); hold on;
plot(dataValN_ESS.t, dataValN_ESS.uN, '--', 'LineWidth',1.2);
grid on;
xlabel('Time [min]');
ylabel('u [L/min dev.]');
title('ESS: virtual teacher and NN control during autonomous validation');
legend('u^{T,v}','u^N','Location','best');

nexttile;
plot(dataValN_FT.t, dataValN_FT.uTv, 'LineWidth',1.2); hold on;
plot(dataValN_FT.t, dataValN_FT.uN, '--', 'LineWidth',1.2);
grid on;
xlabel('Time [min]');
ylabel('u [L/min dev.]');
title('ESS_{FT}: virtual teacher and NN control during autonomous validation');
legend('u^{T,v}','u^N','Location','best');

nexttile;
plot(dataValT.t, dataValT.uT, 'LineWidth',1.2); hold on;
plot(dataValN_ESS.t, dataValN_ESS.uN, '--', 'LineWidth',1.2);
plot(dataValN_FT.t, dataValN_FT.uN, '-.', 'LineWidth',1.2);
grid on;
xlabel('Time [min]');
ylabel('u [L/min dev.]');
title('Teacher control on validation trajectory vs autonomous NN controls');
legend('u^T on D_{val}^T','ESS u^N','ESS_{FT} u^N','Location','best');

saveas(gcf, fullfile(figDir, 'ESS_FT_autonomous_validation_control_signals.png'));

%% ============================================================
%  9) Final compact metric comparison
%% ============================================================

figure('Name','ESS / ESS_FT - Final scalar metrics', ...
       'Position',[220 220 1400 750]);

tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

nexttile;
bar([mse_tr_ESS, mse_val_ESS, mse_loc_ESS; ...
     mse_tr_FT,  mse_val_FT,  mse_loc_FT]);
grid on;
set(gca, 'XTickLabel', {'ESS','ESS_{FT}'});
ylabel('MSE');
title('One-step imitation MSE');
legend('Train','Validation','Local','Location','best');

nexttile;
bar([mae_tr_ESS, mae_val_ESS, mae_loc_ESS; ...
     mae_tr_FT,  mae_val_FT,  mae_loc_FT]);
grid on;
set(gca, 'XTickLabel', {'ESS','ESS_{FT}'});
ylabel('MAE');
title('One-step imitation MAE');
legend('Train','Validation','Local','Location','best');

nexttile;
bar([rhoAcl_ESS; rhoAcl_FT]); hold on;
yline(1.0, 'r--', 'Schur boundary', 'LineWidth',1.2);
grid on;
set(gca, 'XTickLabel', {'ESS','ESS_{FT}'});
ylabel('\rho(A_{cl})');
title('Final local closed-loop spectral radius');

nexttile;
bar([DeltaInf_val_ESS, maxErr_val_ESS, maxUdev_val_ESS; ...
     DeltaInf_val_FT,  maxErr_val_FT,  maxUdev_val_FT]);
grid on;
set(gca, 'XTickLabel', {'ESS','ESS_{FT}'});
ylabel('Value');
title('Autonomous validation metrics');
legend('\Delta_\infty','max|e|','max|Q-Q_0|','Location','best');

saveas(gcf, fullfile(figDir, 'ESS_FT_final_scalar_metrics.png'));

fprintf('Figures saved in: %s\n', figDir);

%% ============================================================
%  Print complete summary
%% ============================================================

fprintf('\n============================================================\n');
fprintf('              ESS / ESS_FT TRAINING SUMMARY\n');
fprintf('============================================================\n');

fprintf('\n--- One-step imitation metrics ---\n');
fprintf('ESS     Train MSE = %.6e | Train MAE = %.6e\n', mse_tr_ESS, mae_tr_ESS);
fprintf('ESS     Val   MSE = %.6e | Val   MAE = %.6e\n', mse_val_ESS, mae_val_ESS);
fprintf('ESS     Local MSE = %.6e | Local MAE = %.6e\n', mse_loc_ESS, mae_loc_ESS);
fprintf('ESS_FT  Train MSE = %.6e | Train MAE = %.6e\n', mse_tr_FT, mae_tr_FT);
fprintf('ESS_FT  Val   MSE = %.6e | Val   MAE = %.6e\n', mse_val_FT, mae_val_FT);
fprintf('ESS_FT  Local MSE = %.6e | Local MAE = %.6e\n', mse_loc_FT, mae_loc_FT);

fprintf('\n--- Local linear stability diagnostics ---\n');
fprintf('ESS     rho(Acl) = %.10f | Schur = %d | Selected epoch = %d\n', ...
    rhoAcl_ESS, rhoAcl_ESS < 1, bestESS.epoch);
fprintf('ESS_FT  rho(Acl) = %.10f | Schur = %d | Selected epoch = %d\n', ...
    rhoAcl_FT, rhoAcl_FT < 1, bestFT.epoch);

fprintf('\n--- Autonomous dynamic imitation errors ---\n');
fprintf('ESS     Val DeltaInf = %.6e\n', DeltaInf_val_ESS);
fprintf('ESS_FT  Val DeltaInf = %.6e\n', DeltaInf_val_FT);

fprintf('\n--- Autonomous max tracking errors ---\n');
fprintf('ESS     Val max|e| = %.6e\n', maxErr_val_ESS);
fprintf('ESS_FT  Val max|e| = %.6e\n', maxErr_val_FT);

fprintf('\n--- Autonomous max control deviations ---\n');
fprintf('ESS     Val max|Q-Q0| = %.6e\n', maxUdev_val_ESS);
fprintf('ESS_FT  Val max|Q-Q0| = %.6e\n', maxUdev_val_FT);

fprintf('============================================================\n');

%% ============================================================
%  Local functions
%% ============================================================

function [net, avgLoss, globalIter, averageGrad, averageSqGrad] = ...
    runOneMixedEpisode(net, prob, CBref, scaler, localPack, pANN, ...
    sigmaNoise, lr, cfg, globalIter, averageGrad, averageSqGrad, ...
    beta1, beta2, epsAdam)

    cstr   = prob.cstr;
    ctrl   = prob.ctrl;
    simopt = prob.simopt;
    nncfg  = prob.nn;

    dt = simopt.dt;
    N  = numel(CBref);

    CA = cstr.CA0;
    CB = cstr.CB0;

    xiT = 0;
    vdT = 0;

    eHist = zeros(nncfg.ws,1);
    ieRaw = 0;
    uPrev = 0;

    phi_chunk = zeros(cfg.chunkSize, nncfg.nphi);
    uT_chunk  = zeros(cfg.chunkSize, 1);
    e_chunk   = zeros(cfg.chunkSize, 1);

    chunkIdx = 0;
    epochLoss = 0;
    nUpdates = 0;

    mu_vec    = scaler.mu(:);
    sigma_vec = scaler.sigma(:);

    for k = 1:N-1

        yDev = CB - cstr.CB0;
        rDev = CBref(k) - cstr.CB0;
        e    = rDev - yDev;

        ieRaw = ieRaw + dt*e;
        phik = [e; eHist(1:nncfg.ws-1); ieRaw; uPrev];

        uT = ctrl.Kp*(ctrl.beta*rDev - yDev + xiT - vdT);

        phi_sc = (phik - mu_vec)./sigma_vec;
        uANN   = double(extractdata(forward(net, dlarray(phi_sc,'CB'))));

        if sigmaNoise > 0
            uANN = uANN + sigmaNoise*randn;
        end

        uApp = pANN*uANN + (1-pANN)*uT;
        uApp = min(max(uApp, simopt.Q_min-cstr.Q0), simopt.Q_max-cstr.Q0);

        Q = cstr.Q0 + uApp;

        dCA = -cstr.k1*CA - cstr.k3*CA^2 ...
              + (Q/cstr.V)*(cstr.CAi0 - CA);

        dCB =  cstr.k1*CA - cstr.k2*CB ...
              - (Q/cstr.V)*CB;

        dxi = e/ctrl.Ti;
        dvd = (ctrl.Td/ctrl.Tf)*dCB - (1/ctrl.Tf)*vdT;

        chunkIdx = chunkIdx + 1;
        phi_chunk(chunkIdx,:) = phik.';
        uT_chunk(chunkIdx)    = uT;
        e_chunk(chunkIdx)     = e;

        if chunkIdx == cfg.chunkSize || k == N-1

            nActual = chunkIdx;

            X_raw = phi_chunk(1:nActual,:);
            Y_raw = uT_chunk(1:nActual);
            E_raw = e_chunk(1:nActual);

            X_sc = (X_raw - scaler.mu)./scaler.sigma;

            X_dl = dlarray(X_sc.','CB');
            Y_dl = dlarray(Y_raw.','CB');
            E_dl = dlarray(E_raw.','CB');

            if cfg.local.use && localPack.use
                idxLoc = randi(localPack.Nloc, cfg.local.batchSize, 1);
                X_loc_dl = dlarray(localPack.X_loc_sc(idxLoc,:).','CB');
                Y_loc_dl = dlarray(localPack.Y_loc_raw(idxLoc).','CB');
            else
                X_loc_dl = [];
                Y_loc_dl = [];
            end

            [loss_c, grads] = dlfeval(@essCertLoss, net, X_dl, Y_dl, E_dl, ...
                X_loc_dl, Y_loc_dl, localPack.phiEq_dl, cfg);

            grads = clipGradients(grads, cfg.gradClip);

            globalIter = globalIter + 1;

            [net, averageGrad, averageSqGrad] = adamupdate(net, grads, ...
                averageGrad, averageSqGrad, globalIter, lr, beta1, beta2, epsAdam);

            epochLoss = epochLoss + double(extractdata(loss_c));
            nUpdates = nUpdates + 1;
            chunkIdx = 0;
        end

        CA = CA + dt*dCA;
        CB = CB + dt*dCB;

        xiT = xiT + dt*dxi;
        vdT = vdT + dt*dvd;

        eHist = [e; eHist(1:end-1)];
        uPrev = uApp;
    end

    avgLoss = epochLoss/max(nUpdates,1);
end

function [loss, grads] = essCertLoss(net, X_dl, Y_dl, E_dl, ...
    X_loc_dl, Y_loc_dl, phiEq_dl, cfg)

    uANN = forward(net, X_dl);

    mseLoss = mean((uANN - Y_dl).^2, 'all');

    w = 1 + abs(E_dl);
    weightedLoss = mean(w.*(uANN - Y_dl).^2, 'all');

    if cfg.local.use && ~isempty(X_loc_dl)
        uLoc = forward(net, X_loc_dl);
        locLoss = mean((uLoc - Y_loc_dl).^2, 'all');
    else
        locLoss = dlarray(0);
    end

    uEq = forward(net, phiEq_dl);
    eqLoss = mean(uEq.^2, 'all');

    loss = mseLoss ...
         + cfg.lambdaErr*weightedLoss ...
         + cfg.local.lambdaLoc*locLoss ...
         + cfg.local.lambdaEq*eqLoss;

    grads = dlgradient(loss, net.Learnables);
end

function metrics = evaluateCheckpoint(net, scaler, prob, dataValT, stabOpts, phiEq_dl)

    try
        [rhoAcl, rhoInfo] = CSTR_LocalStabilityUtils.computeRhoAcl(net, scaler, prob, stabOpts);
    catch
        rhoAcl = inf;
        rhoInfo = struct();
    end

    isSchur = isfinite(rhoAcl) && rhoAcl < 1.0;

    nnFcnTmp = @(phi_k,uTvirt_k,k) ...
        CSTR_NNTrainingUtils.evalDlnet(net, scaler, phi_k);

    dataValN = CSTR_NNTrainingUtils.simulateNNClosedLoop(prob, dataValT.refCB, nnFcnTmp);

    deltaInf = CSTR_NNTrainingUtils.seqSupNorm(dataValN.Delta);

    uEq = double(extractdata(forward(net, phiEq_dl)));

    metrics = struct();
    metrics.net       = net;
    metrics.rhoAcl    = rhoAcl;
    metrics.rhoInfo   = rhoInfo;
    metrics.isSchur   = isSchur;
    metrics.deltaInf  = deltaInf;
    metrics.uEq       = uEq;
    metrics.absUEq    = abs(uEq);
end

function best = initBestStruct(phaseName)

    best = struct();
    best.phase           = phaseName;
    best.net             = [];
    best.epoch           = NaN;
    best.pANN            = NaN;
    best.valMSE          = inf;
    best.rhoAcl          = inf;
    best.deltaInf        = inf;
    best.uEq             = inf;
    best.absUEq          = inf;
    best.isSchur         = false;
    best.eligible        = false;
    best.stableCandidate = false;
end

function best = maybeUpdateBestCheckpoint(best, cand)

    if isempty(best.net)
        best = copyCandidateToBest(best, cand);
        return;
    end

    newStable = cand.stableCandidate;
    oldStable = best.stableCandidate;

    if newStable && ~oldStable
        best = copyCandidateToBest(best, cand);
        return;
    elseif ~newStable && oldStable
        return;
    end

    if newStable && oldStable
        if cand.valMSE < best.valMSE*0.995
            best = copyCandidateToBest(best, cand);
            return;
        elseif cand.valMSE <= best.valMSE*1.005
            if cand.rhoAcl < best.rhoAcl - 1e-4
                best = copyCandidateToBest(best, cand);
                return;
            elseif abs(cand.rhoAcl - best.rhoAcl) <= 1e-4
                if cand.deltaInf < best.deltaInf
                    best = copyCandidateToBest(best, cand);
                    return;
                elseif abs(cand.deltaInf - best.deltaInf) <= 1e-6
                    if cand.absUEq < best.absUEq
                        best = copyCandidateToBest(best, cand);
                        return;
                    end
                end
            end
        end
        return;
    end

    % If neither is a stable candidate, prefer lower rho(Acl).
    if cand.rhoAcl < best.rhoAcl - 1e-4
        best = copyCandidateToBest(best, cand);
        return;
    elseif abs(cand.rhoAcl - best.rhoAcl) <= 1e-4
        if cand.valMSE < best.valMSE
            best = copyCandidateToBest(best, cand);
            return;
        end
    end
end

function best = copyCandidateToBest(best, cand)

    best.phase           = cand.phase;
    best.net             = cand.net;
    best.epoch           = cand.epoch;
    best.pANN            = cand.pANN;
    best.valMSE          = cand.valMSE;
    best.rhoAcl          = cand.rhoAcl;
    best.deltaInf        = cand.deltaInf;
    best.uEq             = cand.uEq;
    best.absUEq          = cand.absUEq;
    best.isSchur         = cand.isSchur;
    best.eligible        = cand.eligible;
    best.stableCandidate = cand.stableCandidate;
end

function grads = clipGradients(grads, maxNorm)

    gradSqSum = 0;

    for i = 1:height(grads)
        g = extractdata(grads.Value{i});
        gradSqSum = gradSqSum + sum(g(:).^2);
    end

    globalNorm = sqrt(gradSqSum);

    if globalNorm > maxNorm
        scale = maxNorm/globalNorm;
        grads = dlupdate(@(g) g*scale, grads);
    end
end

function p = inverseSigmoidSchedule(epoch, maxEpochs, k)

    tau = (epoch-1)/(maxEpochs-1);
    s = 2*k*tau - k;
    p = 1/(1+exp(-s));
    p = min(p,0.98);
end