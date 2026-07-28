function plotAlaTraces (fn, sns, cmap, outf)
% Plots NMR traces of the alpha carbon alanine peaks, with splitting lines.
% SEE Fig. 5a,c
%
% Parameters:
% - fn : the filename of the spreadsheet
% - sns : array of the data sheet names
% - cmap : colormap for splitting lines
% - outf : path to the rendered image
%
% Note: this code is quite specific to the alanine alpha carbon peak.
%   However, the splitting patterns could be modified to reflect other
%   peaks. To accomplsh this, one would need to modify the variables in
%   the %% PLOT SPLITS %% section, and the subsequent dashline function calls.
%
% CRITICAL: Requires DASHLINE function v1.0.0.0 by Edward Abraham on the
%   MATLAB Central File Exchange.
%   https://www.mathworks.com/matlabcentral/fileexchange/1892-dashline
%
% Copyright 2021-2022 Massachusetts Host-Microbiome Center
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.
%

% Set up figure with tiled plot and axes labels
% set figure resolution
wi = 2.7;   % width in inches
hi = 3.5;     % height in inches
f = figure('PaperUnits', 'inches', 'PaperPosition', [0 0 wi hi]);
t = tiledlayout(numel(sns), 1, 'Padding', 'none', 'TileSpacing', 'none');

% load plot each trace specified in <sns>
for i = 1:numel(sns)
    % load ppm shifts <P>, label <L>, and signals <Z>
    P = readmatrix(fn, 'Sheet', sns(i), 'range', 'A:A', 'NumHeaderLines', 1);
    P = P - 0.02; % adjust 15N
    Z = readmatrix(fn, 'Sheet', sns(i), 'range', 'B:D', 'NumHeaderLines', 1);
    M = mean(Z,2);
    E = std(Z,0,2)/sqrt(3); % SEM
    Zl = M - E;
    Zu = M + E;

    % set plot style
    ax = nexttile;
    hold(ax, 'on');
    ax.XDir = 'reverse';
    ax.LineWidth = 0.5;
    ax.FontSize = 7;
    ax.YAxis.Visible = 'off';
    ax.TickDir = 'out';
    set(ax,'fontname','Arial');
    yticks([]);
    ymin = min(Zl, [], 'all');
    ymax = max(Zu, [], 'all');
    ylim([ymin 1.5*ymax]);
    xlim([53 53.8]);
    xticks([53:0.2:53.8]);
    if i ~= 2
        xticklabels([]); % only show xtick labels on second plot
    else
        xticklabels({'53.0', '53.2', '53.4', '53.6', '53.8'});
    end

    %% PLOT TRACE %%
    % plot SEM
    fill([P(1:end-1),P(1:end-1),P(2:end),P(2:end)]',...
         [Zl(1:end-1),Zu(1:end-1),Zu(2:end),Zl(2:end)]',...
         [0.4 0.4 0.4],...
         'FaceAlpha', 0.5,...
         'LineStyle', 'none');
    % plot trace
    plot(P, M, 'Color', 'k', 'LineWidth', 1);

    %% PLOT SPLITS %%
    basis = 53.7031;
    c1 = -0.3588; % split distance from 13C-C1
    c3 = -0.2306; % split distance from 13C-C3
    n2 = -0.0371; % split distance from C2-N
    s23 = -0.1685; % shift from leftmost peak to left peak of 2-3 split
    sv = [basis basis+c3 basis+c1 basis+c1+c3];  % [U-13C]Ala
    sv2= [basis+s23 basis+c3+s23];  % [2,3-13C]Ala
    % set colormap
    col1 = cmap(1, :); % red
    col2 = cmap(2, :); % orange
    col3 = cmap(3, :); % purple
    col4 = cmap(4, :); % blue
    col5 = cmap(5, :); % yellow-green
    col6 = cmap(6, :); % green

    % plot split lines
    for j = 1:numel(sv)
        if i == 2
            dashline([sv(j) sv(j)+n2 sv(j)+n2], [1.15*ymax 1.05*ymax ymin], 1, 1, 1, 1, 'Color', col4, 'LineWidth', 1);
            dashline([sv(j) sv(j)], [ymin 1.15*ymax], 1, 1, 1, 1, 'Color', col3, 'LineWidth', 1);
            dashline([sv(j) sv(j)], [1.15*ymax 1.5*ymax], 1, 1, 1, 1, 'Color', col1, 'LineWidth', 1);
        elseif i == 1
            dashline([sv(j) sv(j) sv2(2-mod(j, 2))], [ymin 1.05*ymax 1.15*ymax], 1, 1, 1, 1, 'Color', col1, 'LineWidth', 1);
        else
            dashline([sv(j) sv(j)], [ymin 1.2*ymax], 1, 1, 1, 1, 'Color', col1, 'LineWidth', 1);
        end
    end
    for j = 1:numel(sv2)
        if i == 2
            dashline([sv2(j) sv2(j)+n2 sv2(j)+n2], [1.15*ymax 1.05*ymax ymin], 1, 1, 1, 1, 'Color', col6, 'LineWidth', 1);
            dashline([sv2(j) sv2(j)], [ymin 1.15*ymax], 1, 1, 1, 1, 'Color', col5, 'LineWidth', 1);
            dashline([sv2(j) sv2(j)], [1.15*ymax 1.5*ymax], 1, 1, 1, 1, 'Color', col2, 'LineWidth', 1);
        elseif i == 1
            dashline([sv2(j) sv2(j) 53.425], [ymin 1.25*ymax 1.35*ymax], 1, 1, 1, 1, 'Color', col2, 'LineWidth', 1);
        else
            dashline([sv2(j) sv2(j)], [ymin 1.2*ymax], 1, 1, 1, 1, 'Color', col2, 'LineWidth', 1);
        end
    end
end

print(f, outf, "-dsvg");

end
