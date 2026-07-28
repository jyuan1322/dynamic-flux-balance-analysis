function plotStacks (T, P, Znorm, cfg_pks, cfg_ids, cMap, outf)
% Plot 13C stacks with colored peaks.
% The time axis is normalized to the metabolic onset of cfg_pk.
% SEE Fig. 2a,c,e,g,h; ED Figs. 2-4
%
% Parameters:
%  - T: time axis vector for NMR data, normalized to overall metabolic rate
%  - P: ppm shift axis vector for NMR data
%  - Znorm: NMR signal, normalized by RMSE of 130-160ppm region of each spectrum
%  - cfg_pks: expected ppm shifts of reference peaks
%  - cfg_ids: numerical indices grouping cfg_pks by compound
%  - cMap is the colormap for the metabolite labels
%  - outf is the path to which the output files are written, including
%        filename stem
%
% This script requires 1H and 13C spectra, given by <stem + "_1H.xlsx"> and
% <stem + "_13C.xlsx">, to be present in the NMRdata folder.
%
% Outputs:
%  - <stem + "_13Cn.png"> is the surface plot of the timecourse
%  - <stem + "_13Cspecs.svg"> is the final 1D spectrum with color-coded peaks
%
r = 4.5;
tolerance = 0.45;
ppm_min = 0.0;
ppm_max = 200.0;
isotope = '13C';
t_min = 0;
t_max = 36; % 36;

Znorm = Znorm(T >= t_min & T <= t_max, P >= ppm_min & P <= ppm_max);
T = T(T >= t_min & T <= t_max);
P = P(P >= ppm_min & P <= ppm_max);

% handle colors
C = max(log2(abs(Znorm)), ones(size(Znorm, 1), size(Znorm, 2)));
C = C ./ max(C, [], 'all');
C2 = log2(abs(Znorm));
C2 = C2 ./ (max(C2(:, P < ppm_max & P > ppm_min), [], 'all') - 0.1);

% load reference ppm from cfg sheet
unq_nms = unique(cfg_ids, 'stable');

% initialize some variables for peak ID
all_shifts = [];   % vector of ppm values for all peaks
all_signal = [];   % vector of heights for all peaks
all_time = [];   % vector timepoints for peaks
all_cpd_ids = [];   % compound id for each peak

% get peak data, iterating over time series
for i = 1:size(Znorm,1)
  Zvec = Znorm(i, :);
  sigs = zeros(size(cfg_pks, 1), 1); % peak signals at this timepoint
  incl = zeros(size(cfg_pks, 1), 1); % indicates whether each peak meets the signal
  thresh = r;
  for j = 1:size(cfg_pks,1)
    range = P > cfg_pks(j) - tolerance & P < cfg_pks(j) + tolerance;
    result = max(Zvec .* range');
    if result >= thresh
      sigs(j) = result;
      incl(j) = 1;
    end
  end
  incl = logical(incl);
  all_shifts = [all_shifts; cfg_pks(incl)];  % vector of ppm values for all peaks
  all_signal = [all_signal; sigs(incl)];     % vector of heights for all peaks
  all_time = [all_time; ones(sum(incl), 1)*T(i)];
  all_cpd_ids = [all_cpd_ids; cfg_ids(incl)];   % compound id for each peak
end

% PLOT STACK %
disp("Plotting...");
% 1. PLOT SURFACES %
% set figure resolution
wi = 4.5;   % width in inches
hi = 1.8;     % height in inches
f1 = figure('PaperUnits', 'inches', 'PaperPosition', [0 0 4.5 hi]);
f2 = figure('PaperUnits', 'inches', 'PaperPosition', [0 0 wi 0.95]);

% configure 1D panel
ax = subplot(1, 1, 1);
hold(ax, 'on');

% configure stack
set(0, 'currentfigure', f1);
ax = subplot(1, 1, 1);
hold(ax, 'on');
view(10, 10); % view(4.2, 30); %4.2 20
Zsurf = Znorm;

% configure colormap
nc = (4*ones(100, 3) + gray(100)) ./5;
for j = 1:size(unq_nms, 1)
  nc = [nc; (ones(100, 3) + gray(100)) .* 0.25 .* ([1 1 1] + cMap(j, :))];
end

% plotting loop, iterate over labeled peaks
if strcmp(isotope, '13C')
  for j = 1:size(cfg_pks,1)
    % select peak environs and erase in baseline plot
    rng = find(P > cfg_pks(j) - tolerance & P < cfg_pks(j) + tolerance);
    Zarea = Zsurf(:, rng);
    Zsurf(:, rng(2:size(rng, 1) -1)) = nan;
    % plot peak region in stack and panel
    p = surf(P(rng), T, Zarea, C(:, rng) + cfg_ids(j) - 0.01, 'EdgeColor', 'None');
    p.FaceAlpha = 0.15;
    set(0, 'currentfigure', f2);
    ax = subplot(1, 1, 1);
    if (cfg_pks(j) == 24.5) % special case for leucine isocap-isoval overlap
      plot(P(rng), Zarea(end, :), 'Color', "#8C8C8C", 'LineWidth', 0.5);
    else
      plot(P(rng), Zarea(end, :), 'Color', cMap(cfg_ids(j), :), 'LineWidth', 0.5);
    end
    set(0, 'currentfigure', f1);
  end
end
% plot baselines
if strcmp(isotope, '13C')
  p = surf(P, T, Zsurf, C, 'EdgeColor', 'none');
else
  p = surf(P, T, Zsurf, C, 'LineWidth', 0.001);
end
set(0, 'currentfigure', f2);
ax = subplot(1, 1, 1);
plot(P, Zsurf(end, :), 'Color', "#8C8C8C", 'LineWidth', 0.5);
ts = [T(1) T(end)];

% 2. ADJUST PLOT PARAMETERS %
% panel
for i = [1]
  box on;
  xlim(ax, [ppm_min ppm_max]); % [0 200]
  ylim(ax, [-inf 2*max(Znorm(end, :), [], 'all')]);
  yticks(ax, []);
  ax.LineWidth = 0.5;
  ax.FontSize = 5;
  ax.XDir = 'reverse';
  set(ax, 'TickDir', 'out');
  chi = get(ax, 'Children');
  set(ax, 'Children', flipud(chi));
  set(ax, 'Layer', 'top');
end
han=axes(f2,'visible','off');

% stack
set(0, 'currentfigure', f1);
colormap(nc);
caxis([0 size(unq_nms, 1)+1]);
p.LineWidth = 0.2;
p.FaceAlpha = 1;
Ax = gca;
Ax.ZAxis.Visible = 'off';
grid(Ax, 'off');
Ax.XDir = 'reverse';
Ax.Color = 'none';
Ax.LineWidth = 0.5;
Ax.FontSize = 7;
xlim([ppm_min ppm_max]);
ylim([t_min t_max]);
% zlim([0 1.7*max(Znorm(end, :), [], 'all')]);  % 1.3x Pro, Glc, inf Leu
zlim([0 inf]);
Ax.Clipping = 'off';
set(Ax, 'Layer', 'top');
yticks(t_min:12:t_max);
if strcmp(isotope, '13C')
  step = 50;
  xticks(ppm_min:50:ppm_max);
  Ax.XAxis.MinorTick = 'on';
  Ax.XAxis.MinorTickValues = Ax.XAxis.Limits(1):10:Ax.XAxis.Limits(2);
else
  xticks(ppm_min:1:ppm_max);
end

set(Ax,'fontname','Arial');
pbaspect([3 2 1]);

shading(Ax, 'interp');

% 3. PLOT STEMS %
for i = 1:size(unq_nms, 1)
  color = cMap(i, :);
  choice = all_cpd_ids == i;
  p_shift = all_shifts(choice);
  p_time = all_time(choice);
  p_signal = all_signal(choice);
  if cfg_pks(i) == 24.5 % leucine overlap peak
    stem3(p_shift(p_time <= 6), p_time(p_time <= 6), p_signal(p_time <= 6),...
          'filled',...
          'Color', cMap(1, :),...
          'MarkerSize', 2,...
          'LineWidth', 0.1);
    stem3(p_shift(p_time > 6), p_time(p_time > 6), p_signal(p_time > 6),...
          'filled',...
          'Color', color,...
          'MarkerSize', 2,...
          'LineWidth', 0.1);
  else
    stem3(p_shift, p_time , p_signal,...
          'filled',...
          'Color', color,...
          'MarkerSize', 2,...
          'MarkerFaceColor', color,...
          'LineWidth', 0.1);
  end
end

% 4. PLOT RIDGES %
for i = 1:size(cfg_pks)
  cpd = cfg_ids(i);
  choice = all_shifts == cfg_pks(i);
  if sum(choice) > 1
    p_time = min(all_time(choice), [], 'all'):0.1:max(all_time(choice), [], 'all');
    fit_fcn = fit(all_time(choice), all_signal(choice), 'smoothingspline');
    p_shift = ones(size(p_time, 2), 1)*cfg_pks(i);
    p_curve = fit_fcn(p_time);
    if cfg_pks(i) == 24.5 % leucine overlap peak
      plot3(p_shift(p_time <= 6), p_time(p_time <= 6), p_curve(p_time <= 6),...
            'Color', cMap(1, :), 'linewidth', 1);
      plot3(p_shift(p_time > 6), p_time(p_time > 6), p_curve(p_time > 6),...
            'Color', cMap(cpd, :), 'linewidth', 1);
    else
      plot3(p_shift, p_time, p_curve, 'Color', cMap(cpd, :), 'linewidth', 1);
    end
  end
end

%view(4.2, 30); % 10, 30
print(f1, outf + "_" + isotope + "n", "-dpng", "-r1200");
if strcmp(isotope, '13C')
  saveas(f2, outf + "_13Cspecs.svg");
end
end
