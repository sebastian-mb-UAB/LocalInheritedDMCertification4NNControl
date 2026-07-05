clear; clc; close all;

%% ============================================================
%  04_validate_cstr_cases_PID_vs_NN.m
%
%  Nonlinear CSTR validation:
%    PID Teacher vs BL vs BL_smooth vs ESS vs ESS_FT
%
%  Cases:
%    Case 1:
%      CBr = 1.2 at t = 3.0 min
%      dQ  = -0.10 Q0 at t = 12.5 min
%      dCAi= +0.05 CAi0 at t = 20.0 min
%
%    Case 2:
%      CBr = 1.0 at t = 3.0 min
%      dQ  = +0.10 Q0 at t = 12.5 min
%      dCAi= -0.05 CAi0 at t = 20.0 min
%
%  Outputs:
%    results_cstr_cases/cstr_case_validation_results.mat
%    results_cstr_cases/cstr_case_validation_metrics.csv
%    results_cstr_cases/cstr_case_validation_table.tex
%    results_cstr_cases/figures/*.png
% ============================================================

%% ------------------------------------------------------------
%  Setup
%% ------------------------------------------------------------
prob = CSTR_NNTrainingUtils.defaultProblem();
cstr = prob.cstr;
ctrl = prob.ctrl;

simopt = prob.simopt;
simopt.t_end = 25.0;       % min
simopt.dt    = 0.12/60;    % 0.12 s in min

outDir = 'results_cstr_cases';
figDir = fullfile(outDir, 'figures');

if ~exist(outDir, 'dir'), mkdir(outDir); end
if ~exist(figDir, 'dir'), mkdir(figDir); end

%% ------------------------------------------------------------
%  Case definitions
%% ------------------------------------------------------------
caseDefs = struct([]);

caseDefs(1).name         = 'Case 1';
caseDefs(1).shortName    = 'case1';
caseDefs(1).t_sp         = 3.0;
caseDefs(1).CB_ref_after = 1.2;
caseDefs(1).t_dQ         = 12.5;
caseDefs(1).dQ           = -0.10*cstr.Q0;
caseDefs(1).t_dCAi       = 20.0;
caseDefs(1).dCAi         = +0.05*cstr.CAi0;

caseDefs(2).name         = 'Case 2';
caseDefs(2).shortName    = 'case2';
caseDefs(2).t_sp         = 3.0;
caseDefs(2).CB_ref_after = 1.0;
caseDefs(2).t_dQ         = 12.5;
caseDefs(2).dQ           = +0.10*cstr.Q0;
caseDefs(2).t_dCAi       = 20.0;
caseDefs(2).dCAi         = -0.05*cstr.CAi0;

%% ------------------------------------------------------------
%  Load NN controllers
%% ------------------------------------------------------------
models = {};

models{end+1} = struct( ...
    'name', 'PID', ...
    'latex', 'PID', ...
    'type', 'pid', ...
    'file', '');

models{end+1} = makeNNModel('BL', ...
    '$K_N^{\mathrm{BL}}$', ...
    'trainedNN_CSTR_BL.mat');

models{end+1} = makeNNModel('BL_S', ...
    '$K_N^{\mathrm{BL_S}}$', ...
    'trainedNN_CSTR_BL_smooth.mat');

models{end+1} = makeNNModel('ESS', ...
    '$K_N^{\mathrm{ESS}}$', ...
    'trainedNN_CSTR_ESS.mat');

models{end+1} = makeNNModel('ESS_FT', ...
    '$K_N^{\mathrm{ESS_{FT}}}$', ...
    'trainedNN_CSTR_ESS_FT.mat');

nModels = numel(models);
nCases  = numel(caseDefs);

%% ------------------------------------------------------------
%  Simulate all controllers in all cases
%% ------------------------------------------------------------
Results = struct();
metricsRows = {};

fprintf('\n============================================================\n');
fprintf('        NONLINEAR CSTR CASE VALIDATION: PID vs NN\n');
fprintf('============================================================\n');

for c = 1:nCases
    caseDef = caseDefs(c);

    fprintf('\n--- %s ---\n', caseDef.name);

    for m = 1:nModels
        model = models{m};

        fprintf('Simulating %-8s ... ', model.name);

        switch lower(model.type)
            case 'pid'
                out = simulateCSTRCasePID(prob, caseDef, simopt);

            case 'nn'
                nnFcn = buildNNFcnFromModel(model);
                out = simulateCSTRCaseNN(prob, caseDef, simopt, nnFcn);

            otherwise
                error('Unknown model type.');
        end

        Results(c,m).caseName  = caseDef.name;
        Results(c,m).modelName = model.name;
        Results(c,m).out       = out;

        met = out.metrics;

        fprintf('IAE_total=%.5f | TVu_total=%.5f | max|e|=%.5f\n', ...
            met.IAE_total, met.TVu_total, met.maxAbsE);

        metricsRows(end+1,:) = { ...
            caseDef.name, model.name, ...
            met.Jer, met.TVur, ...
            met.JeQ, met.TVuQ, ...
            met.JeCAi, met.TVuCAi, ...
            met.IAE_total, met.TVu_total, ...
            met.maxAbsE, met.maxAbsQDev};
    end

    plotCaseComparison(caseDef, models, Results(c,:), figDir);
end

%% ------------------------------------------------------------
%  Build metrics table and save CSV
%% ------------------------------------------------------------
MetricsTable = cell2table(metricsRows, ...
    'VariableNames', {'Case','Controller', ...
    'Jer','TVur','JeQ','TVuQ','JeCAi','TVuCAi', ...
    'IAE_total','TVu_total','maxAbsE','maxAbsQDev'});

csvFile = fullfile(outDir, 'cstr_case_validation_metrics.csv');
writetable(MetricsTable, csvFile);

fprintf('\nSaved CSV metrics: %s\n', csvFile);

%% ------------------------------------------------------------
%  Generate LaTeX table similar to MoReRT paper table
%% ------------------------------------------------------------
texFile = fullfile(outDir, 'cstr_case_validation_table.tex');
writeLatexMetricsTable(texFile, MetricsTable, models, caseDefs);

fprintf('Saved LaTeX table: %s\n', texFile);

%% ------------------------------------------------------------
%  Save full results
%% ------------------------------------------------------------
matFile = fullfile(outDir, 'cstr_case_validation_results.mat');
save(matFile, 'Results', 'MetricsTable', 'models', 'caseDefs', 'prob', 'simopt');

fprintf('Saved full results: %s\n', matFile);

fprintf('\nDone.\n');

%% ============================================================
%  Local functions
%% ============================================================

function model = makeNNModel(name, latexName, fileName)
    if ~isfile(fileName)
        warning('Model file not found: %s', fileName);
    end

    model = struct();
    model.name  = name;
    model.latex = latexName;
    model.type  = 'nn';
    model.file  = fileName;
end

function nnFcn = buildNNFcnFromModel(model)
    S = load(model.file);

    if isfield(S, 'netFT')
        net = S.netFT;
    elseif isfield(S, 'netESS')
        net = S.netESS;
    elseif isfield(S, 'net')
        net = S.net;
    else
        error('No recognized NN variable found in %s.', model.file);
    end

    if ~isfield(S, 'scaler')
        error('Scaler not found in %s.', model.file);
    end

    scaler = S.scaler;

    nnFcn = @(phi_k,uTvirt_k,k) CSTR_NNTrainingUtils.evalDlnet(net, scaler, phi_k);
end

function out = simulateCSTRCasePID(prob, caseDef, simopt)
    cstr = prob.cstr;
    ctrl = prob.ctrl;

    dt = simopt.dt;
    t  = (0:dt:simopt.t_end).';
    N  = numel(t);

    CA    = zeros(N,1);
    CB    = zeros(N,1);
    xi    = zeros(N,1);
    vd    = zeros(N,1);
    Qctrl = zeros(N,1);
    Qact  = zeros(N,1);
    CAi   = zeros(N,1);
    refCB = zeros(N,1);
    e     = zeros(N,1);
    uDev  = zeros(N,1);

    CA(1) = cstr.CA0;
    CB(1) = cstr.CB0;
    xi(1) = 0;
    vd(1) = 0;

    for k = 1:N-1
        tk = t(k);

        if tk < caseDef.t_sp
            CB_ref = cstr.CB0;
        else
            CB_ref = caseDef.CB_ref_after;
        end

        dQ = 0;
        if tk >= caseDef.t_dQ
            dQ = caseDef.dQ;
        end

        dCAi = 0;
        if tk >= caseDef.t_dCAi
            dCAi = caseDef.dCAi;
        end

        CAi_tmp = cstr.CAi0 + dCAi;

        y = CB(k) - cstr.CB0;
        r = CB_ref - cstr.CB0;
        ek = r - y;

        u = ctrl.Kp*(ctrl.beta*r - y + xi(k) - vd(k));

        Q = cstr.Q0 + u + dQ;
        Q = min(max(Q, simopt.Q_min), simopt.Q_max);

        dCA_dt = -cstr.k1*CA(k) - cstr.k3*CA(k)^2 ...
                 + (Q/cstr.V)*(CAi_tmp - CA(k));

        dCB_dt =  cstr.k1*CA(k) - cstr.k2*CB(k) ...
                 - (Q/cstr.V)*CB(k);

        dxi_dt = ek/ctrl.Ti;
        dvd_dt = (ctrl.Td/ctrl.Tf)*dCB_dt - (1/ctrl.Tf)*vd(k);

        CA(k+1) = CA(k) + dt*dCA_dt;
        CB(k+1) = CB(k) + dt*dCB_dt;
        xi(k+1) = xi(k) + dt*dxi_dt;
        vd(k+1) = vd(k) + dt*dvd_dt;

        Qctrl(k) = cstr.Q0 + u;
        Qact(k)  = Q;
        CAi(k)   = CAi_tmp;
        refCB(k) = CB_ref;
        e(k)     = CB_ref - CB(k);
        uDev(k)  = u;
    end

    Qctrl(end) = Qctrl(end-1);
    Qact(end)  = Qact(end-1);
    CAi(end)   = CAi(end-1);
    refCB(end) = refCB(end-1);
    e(end)     = refCB(end) - CB(end);
    uDev(end)  = uDev(end-1);

    out = packageCaseOutput(t, CA, CB, refCB, e, Qact, Qctrl, CAi, uDev, [], caseDef);
end

function out = simulateCSTRCaseNN(prob, caseDef, simopt, nnFcn)
    cstr = prob.cstr;
    ctrl = prob.ctrl;
    nn   = prob.nn;

    dt = simopt.dt;
    t  = (0:dt:simopt.t_end).';
    N  = numel(t);

    CA    = zeros(N,1);
    CB    = zeros(N,1);
    Qctrl = zeros(N,1);
    Qact  = zeros(N,1);
    CAi   = zeros(N,1);
    refCB = zeros(N,1);
    e     = zeros(N,1);
    uN    = zeros(N,1);
    uTv   = zeros(N,1);
    Delta = zeros(N,1);

    % Virtual teacher states
    xiV = zeros(N,1);
    vdV = zeros(N,1);

    % NN regressor states
    eHist = zeros(nn.ws,1);
    ieRaw = 0;
    uPrev = 0;

    CA(1) = cstr.CA0;
    CB(1) = cstr.CB0;

    for k = 1:N-1
        tk = t(k);

        if tk < caseDef.t_sp
            CB_ref = cstr.CB0;
        else
            CB_ref = caseDef.CB_ref_after;
        end

        dQ = 0;
        if tk >= caseDef.t_dQ
            dQ = caseDef.dQ;
        end

        dCAi = 0;
        if tk >= caseDef.t_dCAi
            dCAi = caseDef.dCAi;
        end

        CAi_tmp = cstr.CAi0 + dCAi;

        y = CB(k) - cstr.CB0;
        r = CB_ref - cstr.CB0;
        ek = r - y;

        ieRaw = ieRaw + dt*ek;
        phik = [ek; eHist(1:nn.ws-1); ieRaw; uPrev];

        uTvirt = ctrl.Kp*(ctrl.beta*r - y + xiV(k) - vdV(k));
        u = nnFcn(phik, uTvirt, k);

        Q = cstr.Q0 + u + dQ;
        Q = min(max(Q, simopt.Q_min), simopt.Q_max);

        dCA_dt = -cstr.k1*CA(k) - cstr.k3*CA(k)^2 ...
                 + (Q/cstr.V)*(CAi_tmp - CA(k));

        dCB_dt =  cstr.k1*CA(k) - cstr.k2*CB(k) ...
                 - (Q/cstr.V)*CB(k);

        dxi_dt = ek/ctrl.Ti;
        dvd_dt = (ctrl.Td/ctrl.Tf)*dCB_dt - (1/ctrl.Tf)*vdV(k);

        CA(k+1) = CA(k) + dt*dCA_dt;
        CB(k+1) = CB(k) + dt*dCB_dt;

        xiV(k+1) = xiV(k) + dt*dxi_dt;
        vdV(k+1) = vdV(k) + dt*dvd_dt;

        Qctrl(k) = cstr.Q0 + u;
        Qact(k)  = Q;
        CAi(k)   = CAi_tmp;
        refCB(k) = CB_ref;
        e(k)     = CB_ref - CB(k);
        uN(k)    = u;
        uTv(k)   = uTvirt;
        Delta(k) = u - uTvirt;

        eHist = [ek; eHist(1:end-1)];
        uPrev = u;
    end

    Qctrl(end) = Qctrl(end-1);
    Qact(end)  = Qact(end-1);
    CAi(end)   = CAi(end-1);
    refCB(end) = refCB(end-1);
    e(end)     = refCB(end) - CB(end);
    uN(end)    = uN(end-1);
    uTv(end)   = uTv(end-1);
    Delta(end) = Delta(end-1);

    extra = struct();
    extra.uN = uN;
    extra.uTv = uTv;
    extra.Delta = Delta;

    out = packageCaseOutput(t, CA, CB, refCB, e, Qact, Qctrl, CAi, uN, extra, caseDef);
end

function out = packageCaseOutput(t, CA, CB, refCB, e, Qact, Qctrl, CAi, uDev, extra, caseDef)
    idx_r   = (t >= caseDef.t_sp)   & (t < caseDef.t_dQ);
    idx_Q   = (t >= caseDef.t_dQ)   & (t < caseDef.t_dCAi);
    idx_CAi = (t >= caseDef.t_dCAi);

    metrics = struct();
    metrics.Jer       = localIAE(t, e, idx_r);
    metrics.TVur      = localTV(Qact, idx_r);
    metrics.JeQ       = localIAE(t, e, idx_Q);
    metrics.TVuQ      = localTV(Qact, idx_Q);
    metrics.JeCAi     = localIAE(t, e, idx_CAi);
    metrics.TVuCAi    = localTV(Qact, idx_CAi);
    metrics.IAE_total = trapz(t, abs(e));
    metrics.TVu_total = sum(abs(diff(Qact)));
    metrics.maxAbsE   = max(abs(e));
    metrics.maxAbsQDev= max(abs(Qact - 40));  % overwritten below if needed

    out = struct();
    out.t       = t;
    out.CA      = CA;
    out.CB      = CB;
    out.refCB   = refCB;
    out.e       = e;
    out.Q       = Qact;
    out.Qctrl   = Qctrl;
    out.CAi     = CAi;
    out.uDev    = uDev;
    out.metrics = metrics;
    out.caseDef = caseDef;

    if ~isempty(extra)
        f = fieldnames(extra);
        for i = 1:numel(f)
            out.(f{i}) = extra.(f{i});
        end
    end
end

function val = localIAE(t, e, idx)
    tt = t(idx);
    ee = abs(e(idx));
    if numel(tt) < 2
        val = NaN;
    else
        val = trapz(tt, ee);
    end
end

function val = localTV(u, idx)
    uu = u(idx);
    if numel(uu) < 2
        val = NaN;
    else
        val = sum(abs(diff(uu)));
    end
end

function plotCaseComparison(caseDef, models, caseResults, figDir)
    nModels = numel(models);

    figure('Name',['Nonlinear CSTR validation - ' caseDef.name], ...
           'Position',[60 60 1500 900]);

    tiledlayout(4,1,'TileSpacing','compact','Padding','compact');

    %% Output
    nexttile;
    hold on;
    for m = 1:nModels
        out = caseResults(m).out;
        if m == 1
            plot(out.t, out.refCB, 'k--', 'LineWidth', 1.1);
        end
        plot(out.t, out.CB, 'LineWidth', 1.2);
    end
    xline(caseDef.t_sp,'k:','SP');
    xline(caseDef.t_dQ,'k:','dQ');
    xline(caseDef.t_dCAi,'k:','dC_{Ai}');
    grid on;
    xlabel('Time [min]');
    ylabel('C_B [gmol/L]');
    title([caseDef.name ' - Output response']);
    legendLabels = [{'C_{B,r}'}, cellfun(@(x) x.name, models, 'UniformOutput', false)];
    legend(legendLabels, 'Location','best');

    %% Error
    nexttile;
    hold on;
    for m = 1:nModels
        out = caseResults(m).out;
        plot(out.t, out.e, 'LineWidth', 1.2);
    end
    xline(caseDef.t_sp,'k:');
    xline(caseDef.t_dQ,'k:');
    xline(caseDef.t_dCAi,'k:');
    grid on;
    xlabel('Time [min]');
    ylabel('e [gmol/L]');
    title([caseDef.name ' - Tracking error']);
    legend(cellfun(@(x) x.name, models, 'UniformOutput', false), 'Location','best');

    %% Manipulated variable
    nexttile;
    hold on;
    for m = 1:nModels
        out = caseResults(m).out;
        plot(out.t, out.Q, 'LineWidth', 1.2);
    end
    yline(40,'k:','Q_0');
    xline(caseDef.t_sp,'k:');
    xline(caseDef.t_dQ,'k:');
    xline(caseDef.t_dCAi,'k:');
    grid on;
    xlabel('Time [min]');
    ylabel('Q [L/min]');
    title([caseDef.name ' - Manipulated variable']);
    legend([cellfun(@(x) x.name, models, 'UniformOutput', false), {'Q_0'}], 'Location','best');

    %% Delta for NN models
    nexttile;
    hold on;
    hasDelta = false;
    deltaLabels = {};
    for m = 1:nModels
        out = caseResults(m).out;
        if isfield(out, 'Delta')
            plot(out.t, out.Delta, 'LineWidth', 1.2);
            hasDelta = true;
            deltaLabels{end+1} = models{m}.name; 
        end
    end
    xline(caseDef.t_sp,'k:');
    xline(caseDef.t_dQ,'k:');
    xline(caseDef.t_dCAi,'k:');
    grid on;
    xlabel('Time [min]');
    ylabel('\Delta = u^N-u^{T,v}');
    title([caseDef.name ' - Dynamic imitation error']);
    if hasDelta
        legend(deltaLabels, 'Location','best');
    end

    saveas(gcf, fullfile(figDir, [caseDef.shortName '_PID_vs_NN.png']));
end

function writeLatexMetricsTable(texFile, MetricsTable, models, caseDefs)
    metricNames = {'Jer','TVur','JeQ','TVuQ','JeCAi','TVuCAi'};
    metricLatex = {'$J_{er}$', '$\mathrm{TV}_{ur}$', ...
                   '$J_{eQ}$', '$\mathrm{TV}_{uQ}$', ...
                   '$J_{eCAi}$', '$\mathrm{TV}_{uCAi}$'};

    fid = fopen(texFile, 'w');
    if fid < 0
        error('Could not open %s for writing.', texFile);
    end

    nModels = numel(models);
    nCases  = numel(caseDefs);

    fprintf(fid, '\\begin{table}[!ht]\n');
    fprintf(fid, '\\centering\n');
    fprintf(fid, '\\caption{Nonlinear CSTR performance metrics for the PID teacher and the NN controllers.}\n');
    fprintf(fid, '\\label{tab:cstr_pid_nn_nonlinear_metrics}\n');
    fprintf(fid, '\\scriptsize\n');
    fprintf(fid, '\\resizebox{\\textwidth}{!}{%%\n');

    colSpec = 'c';
    for c = 1:nCases
        colSpec = [colSpec repmat('c',1,nModels)]; 
    end

    fprintf(fid, '\\begin{tabular}{%s}\n', colSpec);
    fprintf(fid, '\\hline\n');

    fprintf(fid, '& ');
    for c = 1:nCases
        if c > 1, fprintf(fid, ' & '); end
        fprintf(fid, '\\multicolumn{%d}{c}{%s}', nModels, caseDefs(c).name);
    end
    fprintf(fid, ' \\\\\n');

    fprintf(fid, '\\cline{2-%d}\n', 1+nModels*nCases);

    fprintf(fid, 'Metric');
    for c = 1:nCases
        for m = 1:nModels
            fprintf(fid, ' & %s', models{m}.latex);
        end
    end
    fprintf(fid, ' \\\\\n');
    fprintf(fid, '\\hline\n');

    for r = 1:numel(metricNames)
        metName = metricNames{r};
        fprintf(fid, '%s', metricLatex{r});

        for c = 1:nCases
            caseName = caseDefs(c).name;

            vals = nan(1,nModels);
            for m = 1:nModels
                idx = strcmp(MetricsTable.Case, caseName) & ...
                      strcmp(MetricsTable.Controller, models{m}.name);
                vals(m) = MetricsTable.(metName)(idx);
            end

            minVal = min(vals);

            for m = 1:nModels
                txt = sprintf('%.4f', vals(m));
                if abs(vals(m)-minVal) <= 1e-10
                    txt = ['\textbf{' txt '}'];
                end
                fprintf(fid, ' & %s', txt);
            end
        end

        fprintf(fid, ' \\\\\n');
    end

    fprintf(fid, '\\hline\n');
    fprintf(fid, '\\end{tabular}%%\n');
    fprintf(fid, '}\n');
    fprintf(fid, '\\end{table}\n');

    fclose(fid);
end