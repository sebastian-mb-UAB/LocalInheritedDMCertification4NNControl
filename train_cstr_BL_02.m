clear; clc; close all;

%% ============================================================
%  02_train_cstr_BL.m
%
%  Batch Learning NN controller for nonlinear CSTR PID imitation.
%
%  Conditions:
%    - Global PID imitation loss
%    - Local equilibrium-region imitation loss
%    - Equilibrium control penalty u_N(phi*) ~= 0
%    - Local closed-loop stability checkpoint filter rho(A_cl)<1
%
%  NN input:
%      phi_k = [e_k, e_{k-1}, ..., e_{k-ws+1}, int(e), u_{k-1}]
%
%  NN output:
%      u_k^N = Q_k^N - Q0   [L/min deviation]
%
%  Output:
%      trainedNN_CSTR_BL.mat
%      or trainedNN_CSTR_BL_smooth.mat
% ============================================================

load('D_tr_T.mat',  'dataTr',   'prob');
load('D_val_T.mat', 'dataValT');

if ~isfile('D_loc_T.mat')
    error('D_loc_T.mat not found.');
end
load('D_loc_T.mat', 'dataLocT');

X_raw     = dataTr.phi;
Y_raw     = dataTr.uT;
X_val_raw = dataValT.phi;
Y_val_raw = dataValT.uT;

X_loc_raw = dataLocT.phi_aug;
Y_loc_raw = dataLocT.uT_aug;

Ntr   = size(X_raw,1);
Nval  = size(X_val_raw,1);
Nloc  = size(X_loc_raw,1);

fprintf('Training samples        : %d\n', Ntr);
fprintf('Validation samples      : %d\n', Nval);
fprintf('Local augmented samples : %d\n', Nloc);

%% ------------------------------------------------------------
%  Configuration
%% ------------------------------------------------------------
modelName = 'BL';

cfg.activationType = "leakyrelu";  % IQC sector [0.01,1]

% Main BL training
cfg.maxEpochs      = 120;
cfg.miniBatchSize  = 512;
cfg.lr             = 1e-3;
cfg.gradClip       = 5.0;

% Set >0 to train BL_smooth
cfg.smoothLambda   = 0.10;

% Local certification-oriented regularization
cfg.local.use        = true;
cfg.local.lambdaLoc  = 0.50;
cfg.local.lambdaEq   = 0.10;
cfg.local.batchSize  = 512;

% Local stability checkpointing
cfg.cert.useFilter   = true;
cfg.cert.checkFreq   = 5;
cfg.cert.rhoTarget   = 0.98;
cfg.cert.rhoAccept   = 1.00;

if cfg.smoothLambda > 0
    modelName = 'BL_smooth';
end

fprintf('\n--- Training CSTR %s controller ---\n', modelName);
fprintf('Local loss enabled       : %d\n', cfg.local.use);
fprintf('lambdaLoc                : %.4f\n', cfg.local.lambdaLoc);
fprintf('lambdaEq                 : %.4f\n', cfg.local.lambdaEq);
fprintf('rhoTarget                : %.4f\n', cfg.cert.rhoTarget);
fprintf('rhoAccept                : %.4f\n\n', cfg.cert.rhoAccept);

%% ------------------------------------------------------------
%  Standardization
%% ------------------------------------------------------------
% IMPORTANT:
% Scaler is fitted only on the global training dataset.
% Local and validation datasets use the same scaler.
scaler.mu    = mean(X_raw,1);
scaler.sigma = std(X_raw,0,1);
scaler.sigma(scaler.sigma < 1e-8) = 1;

X_sc      = (X_raw     - scaler.mu)./scaler.sigma;
X_val_sc  = (X_val_raw - scaler.mu)./scaler.sigma;
X_loc_sc  = (X_loc_raw - scaler.mu)./scaler.sigma;

% Equilibrium anchor phi* = 0 in raw variables
phiEq_raw = zeros(prob.nn.nphi,1);
phiEq_sc  = (phiEq_raw.' - scaler.mu)./scaler.sigma;
phiEq_dl  = dlarray(phiEq_sc.','CB');

%% ------------------------------------------------------------
%  Network
%% ------------------------------------------------------------
net = CSTR_NNTrainingUtils.buildMLP(prob.nn.nphi, cfg.activationType);

%% ------------------------------------------------------------
%  Adam setup
%% ------------------------------------------------------------
averageGrad   = [];
averageSqGrad = [];

beta1   = 0.9;
beta2   = 0.999;
epsAdam = 1e-8;
iter    = 0;

numBatches = ceil(Ntr/cfg.miniBatchSize);

%% ------------------------------------------------------------
%  Histories
%% ------------------------------------------------------------
loss_history        = nan(cfg.maxEpochs,1);
val_loss_history    = nan(cfg.maxEpochs,1);
rhoAcl_history      = nan(cfg.maxEpochs,1);
isSchur_history     = false(cfg.maxEpochs,1);
deltaInf_history    = nan(cfg.maxEpochs,1);
eqOutput_history    = nan(cfg.maxEpochs,1);

%% ------------------------------------------------------------
%  Best checkpoint storage
%% ------------------------------------------------------------
best.net             = [];
best.epoch           = NaN;
best.valMSE          = inf;
best.rhoAcl          = inf;
best.deltaInf        = inf;
best.eqOutput        = inf;
best.isSchur         = false;
best.stableCandidate = false;
best.reason          = "none";

stabOpts = CSTR_LocalStabilityUtils.defaultOptions(prob);

%% ------------------------------------------------------------
%  Training loop
%% ------------------------------------------------------------
for epoch = 1:cfg.maxEpochs

    if cfg.smoothLambda == 0
        idxOrder = randperm(Ntr);
    else
        % Keep temporal order if smoothness penalty is active
        idxOrder = 1:Ntr;
    end

    epochLoss = 0;

    for b = 1:numBatches
        idx1 = (b-1)*cfg.miniBatchSize + 1;
        idx2 = min(b*cfg.miniBatchSize, Ntr);
        idx  = idxOrder(idx1:idx2);

        X_b = dlarray(X_sc(idx,:).','CB');
        Y_b = dlarray(Y_raw(idx).','CB');

        if cfg.local.use
            idxLoc = randi(Nloc, cfg.local.batchSize, 1);
            X_loc_b = dlarray(X_loc_sc(idxLoc,:).','CB');
            Y_loc_b = dlarray(Y_loc_raw(idxLoc).','CB');
        else
            X_loc_b = [];
            Y_loc_b = [];
        end

        [loss_b, grads] = dlfeval(@blCertLoss, net, X_b, Y_b, ...
            X_loc_b, Y_loc_b, phiEq_dl, cfg);

        grads = clipGradients(grads, cfg.gradClip);

        iter = iter + 1;
        [net, averageGrad, averageSqGrad] = adamupdate(net, grads, ...
            averageGrad, averageSqGrad, iter, cfg.lr, beta1, beta2, epsAdam);

        epochLoss = epochLoss + double(extractdata(loss_b));
    end

    loss_history(epoch) = epochLoss/numBatches;

    %% One-step validation MSE
    Y_val_pred = extractdata(forward(net, dlarray(X_val_sc.','CB'))).';
    val_loss_history(epoch) = mean((Y_val_pred(:)-Y_val_raw).^2);

    %% Equilibrium output
    uEq = double(extractdata(forward(net, phiEq_dl)));
    eqOutput_history(epoch) = uEq;

    %% Certification-oriented checkpoint diagnostics
    doCheck = cfg.cert.useFilter && ...
        (epoch == 1 || mod(epoch,cfg.cert.checkFreq)==0 || epoch == cfg.maxEpochs);

    if doCheck
        try
            [rhoAcl, rhoInfo] = CSTR_LocalStabilityUtils.computeRhoAcl(net, scaler, prob, stabOpts);
            isSchur = rhoAcl < 1.0;
        catch ME
            warning('rho(Acl) computation failed at epoch %d: %s', epoch, ME.message);
            rhoAcl = inf;
            rhoInfo = struct();
            isSchur = false;
        end

        rhoAcl_history(epoch)  = rhoAcl;
        isSchur_history(epoch) = isSchur;

        nnFcnTmp = @(phi_k,uTvirt_k,k) ...
            CSTR_NNTrainingUtils.evalDlnet(net, scaler, phi_k);

        dataValN_tmp = CSTR_NNTrainingUtils.simulateNNClosedLoop(prob, dataValT.refCB, nnFcnTmp);
        deltaInf = CSTR_NNTrainingUtils.seqSupNorm(dataValN_tmp.Delta);
        deltaInf_history(epoch) = deltaInf;

        stableCandidate = isfinite(rhoAcl) && isfinite(deltaInf) && rhoAcl < cfg.cert.rhoAccept;

        if rhoAcl < cfg.cert.rhoTarget
            reason = "rho_below_target";
        elseif rhoAcl < cfg.cert.rhoAccept
            reason = "rho_below_accept";
        else
            reason = "rho_not_schur";
        end

        newMetrics = struct();
        newMetrics.valMSE          = val_loss_history(epoch);
        newMetrics.rhoAcl          = rhoAcl;
        newMetrics.deltaInf        = deltaInf;
        newMetrics.eqOutput        = abs(uEq);
        newMetrics.isSchur         = isSchur;
        newMetrics.stableCandidate = stableCandidate;
        newMetrics.reason          = reason;

        if isBetterBLCertCheckpoint(newMetrics, best)
            best.net             = net;
            best.epoch           = epoch;
            best.valMSE          = newMetrics.valMSE;
            best.rhoAcl          = newMetrics.rhoAcl;
            best.deltaInf        = newMetrics.deltaInf;
            best.eqOutput        = newMetrics.eqOutput;
            best.isSchur         = newMetrics.isSchur;
            best.stableCandidate = newMetrics.stableCandidate;
            best.reason          = newMetrics.reason;

            fprintf('  *** New best checkpoint at epoch %d | ValMSE=%.4e | rho=%.6f | Schur=%d | DeltaInf=%.4e | reason=%s ***\n', ...
                epoch, best.valMSE, best.rhoAcl, best.isSchur, best.deltaInf, best.reason);
        end

        fprintf('Epoch %3d/%d | Train %.4e | Val %.4e | rho(Acl)=%.6f | Schur=%d | uEq=%.3e | DeltaInf=%.3e\n', ...
            epoch, cfg.maxEpochs, loss_history(epoch), val_loss_history(epoch), ...
            rhoAcl, isSchur, uEq, deltaInf);

    else
        if mod(epoch,10)==0 || epoch==1
            fprintf('Epoch %3d/%d | Train %.4e | Val %.4e | uEq=%.3e\n', ...
                epoch, cfg.maxEpochs, loss_history(epoch), val_loss_history(epoch), uEq);
        end
    end
end

%% ------------------------------------------------------------
%  Select final network
%% ------------------------------------------------------------
if isempty(best.net)
    warning('No checkpoint was selected. Using final network.');
    best.net = net;
    best.epoch = cfg.maxEpochs;
    best.valMSE = val_loss_history(end);

    try
        [best.rhoAcl, ~] = CSTR_LocalStabilityUtils.computeRhoAcl(net, scaler, prob, stabOpts);
    catch
        best.rhoAcl = inf;
    end

    best.isSchur = best.rhoAcl < 1;
    best.stableCandidate = best.isSchur;
    best.reason = "fallback_final_network";
else
    net = best.net;
end

fprintf('\n--- Selected %s checkpoint ---\n', modelName);
fprintf('Epoch              : %d\n', best.epoch);
fprintf('Val MSE            : %.6e\n', best.valMSE);
fprintf('rho(Acl)           : %.10f\n', best.rhoAcl);
fprintf('Schur stable       : %d\n', best.isSchur);
fprintf('Stable candidate   : %d\n', best.stableCandidate);
fprintf('Selection reason   : %s\n', best.reason);

%% ------------------------------------------------------------
%  Final evaluation
%% ------------------------------------------------------------
Y_pred_tr  = CSTR_NNTrainingUtils.batchPredictDlnet(net, scaler, X_raw);
Y_pred_val = CSTR_NNTrainingUtils.batchPredictDlnet(net, scaler, X_val_raw);

Y_pred_loc = CSTR_NNTrainingUtils.batchPredictDlnet(net, scaler, X_loc_raw);

mse_tr  = mean((Y_pred_tr  - Y_raw).^2);
mae_tr  = mean(abs(Y_pred_tr  - Y_raw));

mse_val = mean((Y_pred_val - Y_val_raw).^2);
mae_val = mean(abs(Y_pred_val - Y_val_raw));

mse_loc = mean((Y_pred_loc - Y_loc_raw).^2);
mae_loc = mean(abs(Y_pred_loc - Y_loc_raw));

fprintf('\n[%s] Train MSE = %.6e | MAE = %.6e\n', modelName, mse_tr, mae_tr);
fprintf('[%s] Val   MSE = %.6e | MAE = %.6e\n', modelName, mse_val, mae_val);
fprintf('[%s] Local MSE = %.6e | MAE = %.6e\n', modelName, mse_loc, mae_loc);

%% ------------------------------------------------------------
%  Build nnFcn
%% ------------------------------------------------------------
netSaved = net;
scalerSaved = scaler;

nnFcn = @(phi_k,uTvirt_k,k) ...
    CSTR_NNTrainingUtils.evalDlnet(netSaved, scalerSaved, phi_k);

%% Sanity check
phi_test = dataTr.phi(100,:).';
uT_test  = dataTr.uT(100);
uN_test  = nnFcn(phi_test,uT_test,100);

fprintf('[Sanity] k=100: uT = %.5f | uN = %.5f | delta = %.5f\n', ...
    uT_test, uN_test, uN_test-uT_test);

%% ------------------------------------------------------------
%  Autonomous validation on independent reference
%% ------------------------------------------------------------
dataValN = CSTR_NNTrainingUtils.simulateNNClosedLoop(prob, dataValT.refCB, nnFcn);

deltaInf_final = CSTR_NNTrainingUtils.seqSupNorm(dataValN.Delta);
maxErr_final   = max(abs(dataValN.e));
maxUdev_final  = max(abs(dataValN.Q - prob.cstr.Q0));

fprintf('[Autonomous validation] max|Delta| = %.6e\n', deltaInf_final);
fprintf('[Autonomous validation] max|e_NN|  = %.6e\n', maxErr_final);
fprintf('[Autonomous validation] max|Q-Q0|  = %.6e\n', maxUdev_final);

%% ------------------------------------------------------------
%  Final rho(Acl) check
%% ------------------------------------------------------------
try
    [rhoAcl_final, rhoInfo_final] = CSTR_LocalStabilityUtils.computeRhoAcl(net, scaler, prob, stabOpts);
catch ME
    warning(ME.identifier,'Final rho(Acl) computation failed: %s', ME.message);
    rhoAcl_final = inf;
    rhoInfo_final = struct();
end

fprintf('[Final local stability] rho(Acl) = %.10f | Schur = %d\n', ...
    rhoAcl_final, rhoAcl_final < 1);

%% ------------------------------------------------------------
%  Save
%% ------------------------------------------------------------
outFile = sprintf('trainedNN_CSTR_%s.mat', modelName);

save(outFile, 'net', 'scaler', 'nnFcn', 'cfg', ...
    'mse_tr', 'mae_tr', 'mse_val', 'mae_val', ...
    'mse_loc', 'mae_loc', ...
    'loss_history', 'val_loss_history', ...
    'rhoAcl_history', 'isSchur_history', 'deltaInf_history', ...
    'eqOutput_history', ...
    'best', 'rhoAcl_final', 'rhoInfo_final', ...
    'dataValN', 'deltaInf_final', 'maxErr_final', 'maxUdev_final');

fprintf('\nSaved: %s\n', outFile);

%% ------------------------------------------------------------
%  Complete plots after BL / BL_smooth training
%% ------------------------------------------------------------

figDir = fullfile('results_training_figures', modelName);
if ~exist(figDir, 'dir')
    mkdir(figDir);
end

%% ============================================================
%  1) Teacher imitation on full TRAINING dataset
%% ============================================================

figure('Name',[modelName ' - Full teacher imitation - Training'], ...
       'Position',[50 50 1400 750]);

tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

nexttile;
plot(dataTr.t, Y_raw, 'LineWidth',1.2); hold on;
plot(dataTr.t, Y_pred_tr, '--', 'LineWidth',1.1);
grid on;
xlabel('Time [min]');
ylabel('u [L/min dev.]');
title([modelName ' - Teacher imitation on training data']);
legend('u^T','u^N','Location','best');

nexttile;
plot(dataTr.t, Y_pred_tr - Y_raw, 'LineWidth',1.1);
grid on;
xlabel('Time [min]');
ylabel('u^N-u^T');
title('Training one-step imitation error');

nexttile;
plot(dataTr.t, dataTr.refCB, 'k--', 'LineWidth',1.0); hold on;
plot(dataTr.t, dataTr.CB, 'LineWidth',1.2);
grid on;
xlabel('Time [min]');
ylabel('C_B [gmol/L]');
title('Training trajectory generated by PID teacher');
legend('C_{B,r}','C_B','Location','best');

saveas(gcf, fullfile(figDir, [modelName '_full_training_imitation.png']));

%% ============================================================
%  2) Teacher imitation on full VALIDATION dataset
%% ============================================================

figure('Name',[modelName ' - Full teacher imitation - Validation'], ...
       'Position',[80 80 1400 750]);

tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

nexttile;
plot(dataValT.t, Y_val_raw, 'LineWidth',1.2); hold on;
plot(dataValT.t, Y_pred_val, '--', 'LineWidth',1.1);
grid on;
xlabel('Time [min]');
ylabel('u [L/min dev.]');
title([modelName ' - Teacher imitation on validation data']);
legend('u^T','u^N','Location','best');

nexttile;
plot(dataValT.t, Y_pred_val - Y_val_raw, 'LineWidth',1.1);
grid on;
xlabel('Time [min]');
ylabel('u^N-u^T');
title('Validation one-step imitation error');

nexttile;
plot(dataValT.t, dataValT.refCB, 'k--', 'LineWidth',1.0); hold on;
plot(dataValT.t, dataValT.CB, 'LineWidth',1.2);
grid on;
xlabel('Time [min]');
ylabel('C_B [gmol/L]');
title('Validation trajectory generated by PID teacher');
legend('C_{B,r}','C_B','Location','best');

saveas(gcf, fullfile(figDir, [modelName '_full_validation_imitation.png']));

%% ============================================================
%  3) Local dataset imitation
%% ============================================================

Y_loc_nonanchor = CSTR_NNTrainingUtils.batchPredictDlnet(net, scaler, dataLocT.phi);

figure('Name',[modelName ' - Local dataset imitation'], ...
       'Position',[100 100 1400 850]);

tiledlayout(4,1,'TileSpacing','compact','Padding','compact');

nexttile;
plot(dataLocT.t, dataLocT.uT, 'LineWidth',1.2); hold on;
plot(dataLocT.t, Y_loc_nonanchor, '--', 'LineWidth',1.1);
grid on;
xlabel('Time [min]');
ylabel('u [L/min dev.]');
title([modelName ' - Local teacher imitation near equilibrium']);
legend('u^T_{loc}','u^N_{loc}','Location','best');

nexttile;
plot(dataLocT.t, Y_loc_nonanchor - dataLocT.uT, 'LineWidth',1.1);
grid on;
xlabel('Time [min]');
ylabel('u^N-u^T');
title('Local one-step imitation error');

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

saveas(gcf, fullfile(figDir, [modelName '_local_dataset_imitation.png']));

%% ============================================================
%  4) Autonomous closed-loop validation: PID teacher vs NN
%% ============================================================

figure('Name',[modelName ' - Autonomous validation - Closed-loop'], ...
       'Position',[120 120 1400 900]);

tiledlayout(4,1,'TileSpacing','compact','Padding','compact');

nexttile;
plot(dataValT.t, dataValT.refCB, 'k--', 'LineWidth',1.0); hold on;
plot(dataValT.t, dataValT.CB, 'LineWidth',1.25);
plot(dataValN.t, dataValN.CB, '--', 'LineWidth',1.25);
grid on;
xlabel('Time [min]');
ylabel('C_B [gmol/L]');
title([modelName ' - Autonomous validation: output response']);
legend('C_{B,r}','PID teacher','NN autonomous','Location','best');

nexttile;
plot(dataValT.t, dataValT.e, 'LineWidth',1.2); hold on;
plot(dataValN.t, dataValN.e, '--', 'LineWidth',1.2);
grid on;
xlabel('Time [min]');
ylabel('e [gmol/L]');
title('Tracking error');
legend('PID teacher','NN autonomous','Location','best');

nexttile;
plot(dataValT.t, dataValT.Q, 'LineWidth',1.2); hold on;
plot(dataValN.t, dataValN.Q, '--', 'LineWidth',1.2);
yline(prob.cstr.Q0, 'k:', 'Q_0');
grid on;
xlabel('Time [min]');
ylabel('Q [L/min]');
title('Manipulated variable');
legend('PID teacher','NN autonomous','Q_0','Location','best');

nexttile;
plot(dataValN.t, dataValN.Delta, 'LineWidth',1.2);
grid on;
xlabel('Time [min]');
ylabel('\Delta = u^N-u^{T,v}');
title('Dynamic imitation error during autonomous validation');

saveas(gcf, fullfile(figDir, [modelName '_autonomous_validation_full.png']));

%% ============================================================
%  5) Autonomous control comparison
%% ============================================================

figure('Name',[modelName ' - Autonomous validation - Control signals'], ...
       'Position',[140 140 1400 750]);

tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

nexttile;
plot(dataValN.t, dataValN.uTv, 'LineWidth',1.2); hold on;
plot(dataValN.t, dataValN.uN, '--', 'LineWidth',1.2);
grid on;
xlabel('Time [min]');
ylabel('u [L/min dev.]');
title('Virtual teacher and NN control during autonomous validation');
legend('u^{T,v}','u^N','Location','best');

nexttile;
plot(dataValN.t, dataValN.uN - dataValN.uTv, 'LineWidth',1.2);
grid on;
xlabel('Time [min]');
ylabel('u^N-u^{T,v}');
title('Control residual');

nexttile;
plot(dataValT.t, dataValT.uT, 'LineWidth',1.2); hold on;
plot(dataValN.t, dataValN.uN, '--', 'LineWidth',1.2);
grid on;
xlabel('Time [min]');
ylabel('u [L/min dev.]');
title('Teacher control on validation teacher trajectory vs NN autonomous control');
legend('u^T on D_{val}^T','u^N autonomous','Location','best');

saveas(gcf, fullfile(figDir, [modelName '_autonomous_control_signals.png']));

%% ============================================================
%  6) Training curves
%% ============================================================

figure('Name',[modelName ' - Training curves'], ...
       'Position',[160 160 1400 650]);

tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

nexttile;
plot(loss_history, 'LineWidth',1.3); hold on;
plot(val_loss_history, '--', 'LineWidth',1.3);
if isfinite(best.epoch)
    xline(best.epoch, 'k--', 'best', 'LineWidth',1.0);
end
grid on;
xlabel('Epoch');
ylabel('MSE loss');
title([modelName ' - Training and validation loss']);
legend('Training','Validation','Best checkpoint','Location','best');

nexttile;
semilogy(loss_history, 'LineWidth',1.3); hold on;
semilogy(val_loss_history, '--', 'LineWidth',1.3);
if isfinite(best.epoch)
    xline(best.epoch, 'k--', 'best', 'LineWidth',1.0);
end
grid on;
xlabel('Epoch');
ylabel('MSE loss');
title('Loss curves in logarithmic scale');
legend('Training','Validation','Best checkpoint','Location','best');

nexttile;
plot(eqOutput_history, 'LineWidth',1.3);
if isfinite(best.epoch)
    xline(best.epoch, 'k--', 'best', 'LineWidth',1.0);
end
yline(0, 'k:');
grid on;
xlabel('Epoch');
ylabel('u_N(\phi^\star)');
title('Equilibrium control output');

nexttile;
plot(abs(eqOutput_history), 'LineWidth',1.3);
if isfinite(best.epoch)
    xline(best.epoch, 'k--', 'best', 'LineWidth',1.0);
end
grid on;
xlabel('Epoch');
ylabel('|u_N(\phi^\star)|');
title('Absolute equilibrium control error');

saveas(gcf, fullfile(figDir, [modelName '_training_curves_and_equilibrium.png']));

%% ============================================================
%  7) Local stability diagnostics
%% ============================================================

figure('Name',[modelName ' - Local stability diagnostics'], ...
       'Position',[180 180 1400 750]);

tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

nexttile;
plot(rhoAcl_history, 'o-', 'LineWidth',1.2, 'MarkerSize',4); hold on;
yline(1.0, 'r--', 'Schur boundary', 'LineWidth',1.2);
yline(cfg.cert.rhoTarget, 'k:', '\rho target', 'LineWidth',1.2);
if isfinite(best.epoch)
    xline(best.epoch, 'k--', 'best', 'LineWidth',1.0);
end
grid on;
xlabel('Epoch');
ylabel('\rho(A_{cl})');
title('Local closed-loop spectral radius during training');
legend('\rho(A_{cl})','Schur boundary','Target','Best checkpoint','Location','best');

nexttile;
stairs(double(isSchur_history), 'LineWidth',1.3);
ylim([-0.1 1.1]);
if isfinite(best.epoch)
    xline(best.epoch, 'k--', 'best', 'LineWidth',1.0);
end
grid on;
xlabel('Epoch');
ylabel('Schur flag');
title('Local Schur stability flag');

nexttile;
plot(deltaInf_history, 'o-', 'LineWidth',1.2, 'MarkerSize',4);
if isfinite(best.epoch)
    xline(best.epoch, 'k--', 'best', 'LineWidth',1.0);
end
grid on;
xlabel('Epoch');
ylabel('\Delta_\infty');
title('Autonomous dynamic imitation error at monitored epochs');

saveas(gcf, fullfile(figDir, [modelName '_local_stability_diagnostics.png']));

%% ============================================================
%  8) Final compact comparison
%% ============================================================

figure('Name',[modelName ' - Final compact validation summary'], ...
       'Position',[200 200 1500 850]);

tiledlayout(4,1,'TileSpacing','compact','Padding','compact');

nexttile;
plot(dataValT.t, dataValT.refCB, 'k--', 'LineWidth',1.0); hold on;
plot(dataValT.t, dataValT.CB, 'LineWidth',1.25);
plot(dataValN.t, dataValN.CB, '--', 'LineWidth',1.25);
grid on;
xlabel('Time [min]');
ylabel('C_B [gmol/L]');
title('Validation output response');
legend('C_{B,r}','PID teacher','NN autonomous','Location','best');

nexttile;
plot(dataValT.t, dataValT.Q, 'LineWidth',1.2); hold on;
plot(dataValN.t, dataValN.Q, '--', 'LineWidth',1.2);
yline(prob.cstr.Q0, 'k:', 'Q_0');
grid on;
xlabel('Time [min]');
ylabel('Q [L/min]');
title('Validation manipulated variable');
legend('PID teacher','NN autonomous','Q_0','Location','best');

nexttile;
plot(dataValN.t, dataValN.Delta, 'LineWidth',1.2);
grid on;
xlabel('Time [min]');
ylabel('\Delta');
title('Dynamic imitation error');

nexttile;
bar([mse_tr, mse_val, mse_loc, rhoAcl_final, deltaInf_final]);
grid on;
set(gca, 'XTickLabel', {'MSE_{tr}','MSE_{val}','MSE_{loc}','rho(Acl)','DeltaInf'});
ylabel('Value');
title('Final scalar metrics');

saveas(gcf, fullfile(figDir, [modelName '_final_compact_summary.png']));

fprintf('Figures saved in: %s\n', figDir);

%% ============================================================
%  Print final summary
%% ============================================================
fprintf('\n============================================================\n');
fprintf('              %s TRAINING SUMMARY\n', modelName);
fprintf('============================================================\n');
fprintf('Selected epoch              : %d\n', best.epoch);
fprintf('Train MSE                   : %.6e\n', mse_tr);
fprintf('Train MAE                   : %.6e\n', mae_tr);
fprintf('Validation MSE              : %.6e\n', mse_val);
fprintf('Validation MAE              : %.6e\n', mae_val);
fprintf('Local MSE                   : %.6e\n', mse_loc);
fprintf('Local MAE                   : %.6e\n', mae_loc);
fprintf('Final rho(Acl)              : %.10f\n', rhoAcl_final);
fprintf('Final Schur stable          : %d\n', rhoAcl_final < 1);
fprintf('Autonomous max |Delta|      : %.6e\n', deltaInf_final);
fprintf('Autonomous max |e_NN|       : %.6e\n', maxErr_final);
fprintf('Autonomous max |Q_NN-Q0|    : %.6e\n', maxUdev_final);
fprintf('============================================================\n');

%% ============================================================
%  Local functions
%% ============================================================

function [loss, grads] = blCertLoss(net, X_b, Y_b, X_loc_b, Y_loc_b, phiEq_dl, cfg)

    Y_pred = forward(net, X_b);

    mseLoss = mean((Y_pred - Y_b).^2, 'all');

    if cfg.smoothLambda > 0 && size(Y_pred,2) > 1
        dY = Y_pred(:,2:end)-Y_pred(:,1:end-1);
        smoothLoss = mean(dY.^2,'all');
    else
        smoothLoss = dlarray(0);
    end

    if cfg.local.use && ~isempty(X_loc_b)
        Y_loc_pred = forward(net, X_loc_b);
        locLoss = mean((Y_loc_pred - Y_loc_b).^2, 'all');
    else
        locLoss = dlarray(0);
    end

    uEq = forward(net, phiEq_dl);
    eqLoss = mean(uEq.^2, 'all');

    loss = mseLoss ...
         + cfg.smoothLambda*smoothLoss ...
         + cfg.local.lambdaLoc*locLoss ...
         + cfg.local.lambdaEq*eqLoss;

    grads = dlgradient(loss, net.Learnables);
end

function better = isBetterBLCertCheckpoint(newMetrics, oldMetrics)

    better = false;

    if isempty(oldMetrics.net)
        better = true;
        return;
    end

    newStable = newMetrics.stableCandidate;
    oldStable = oldMetrics.stableCandidate;

    % Stable/certification-oriented checkpoints dominate unstable ones.
    if newStable && ~oldStable
        better = true;
        return;
    elseif ~newStable && oldStable
        better = false;
        return;
    end

    % If both are stable, prefer lower validation MSE, with rho as tie-break.
    if newStable && oldStable
        if newMetrics.valMSE < oldMetrics.valMSE*0.995
            better = true;
            return;
        elseif newMetrics.valMSE <= oldMetrics.valMSE*1.005
            if newMetrics.rhoAcl < oldMetrics.rhoAcl
                better = true;
                return;
            elseif abs(newMetrics.rhoAcl - oldMetrics.rhoAcl) <= 1e-4
                if newMetrics.deltaInf < oldMetrics.deltaInf
                    better = true;
                    return;
                end
            end
        end
        return;
    end

    % If neither is stable, prefer the one closest to Schur stability.
    if newMetrics.rhoAcl < oldMetrics.rhoAcl - 1e-4
        better = true;
        return;
    elseif abs(newMetrics.rhoAcl - oldMetrics.rhoAcl) <= 1e-4
        if newMetrics.valMSE < oldMetrics.valMSE
            better = true;
            return;
        end
    end
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