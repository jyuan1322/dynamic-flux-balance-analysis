function plotFluxBidirectional (stem, snin, snout)
% Stacked area plot, showing proportions of flux for a metabolite.
% SEE Fig. 3a
%
% Parameters:
%  - stem is the filename stem of the excel spreadsheet
%  - snin is the sheetname for influx
%  - snout is the sheetname for outflux
%
% Output:
%  - <stem + ".svg"> is the area plot
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

fn = stem + ".xlsx";

disp("Loading data...");
T = readmatrix(fn, 'Sheet', snin, 'Range', 'A:A', 'NumHeaderLines', 2);
Ni = readmatrix(fn, 'Sheet', snin, 'Range', 2, 'OutputType', 'string');
Ni = Ni(2:end);
Vi = readmatrix(fn, 'Sheet', snin, 'Range', 'B3');

To = readmatrix(fn, 'Sheet', snout, 'Range', 'A:A', 'NumHeaderLines', 2);
No = readmatrix(fn, 'Sheet', snout, 'Range', 2, 'OutputType', 'string');
No = No(2:end);
Vo = readmatrix(fn, 'Sheet', snout, 'Range', 'B3');

if ~isequal(T, To)
    disp('Abort: timescales not equal.');
    return
end

Vi = abs(Vi);
Vo = -1*abs(Vo);
%vsum = sum(V, 2);
%A = V ./ vsum;

cmap = getColor('custom');
%% color order for specific metabolites %%
cmapAla = [ % for alanine
    [74 153 255];  % ALT
    [147 206 255]; % VPT
    [211 255 255]; % SPT
    [0 127 255];   % Tsp_alaL
]./255.;
cmapAce = cmap([2 6 7 8 9], :); % for acetate
cmapPyr = [
    [28 18 195]; %pep
    [28 18 195]; % [72 84 248]; %pts
    [251 129 255];%pfo
    [74 153 255];  % ALT
    [74 153 255]; % [147 206 255]; % VPT
    [74 153 255]; % [211 255 255]; % SPT
    [162 62 0]; % LDH
]./255.;
cmapAtp = [
    [28 18 195]; % [72 84 248]; %3pgk
    [28 18 195]; %pyr kinase
    [253 0 80]; % acetate kinase
    [200 200 200]; % ATP synthase
    [75 223 46]; % oxleu
    [0 128 128]; % oxval
    [245 202 61]; % buty
    [50 50 50]; % oxthr
    [50 50 50]; % oxile
    [53 236 191];  % WLP
    [28 18 195]; %glyco
    [0 128 128]; % leuTSP
    [0 0 128];   % proTSP
    [127 127 127]; % valtsp
    [127 127 127]; % thrtsp
    [127 127 127]; % glytsp
    [74 153 255]; % Glu-A L
    [72 84 248]; % glyco
]./255.;
cmapNadh = [
    [28 18 195]; %3pgk
    [24 173 0]; % nitrite
    [74 153 255];  % GDH
    [127 127 127]; % Rnf
    [28 18 195]; %3pgk
    [253 0 80]; % mannitol
    [24 173 0]; % nitrite
    [74 153 255];  % GDH
    [127 127 127]; % Rnf
]./255.;

if strcmp(stem, "pyrflux")
    cmap = cmapPyr;
    mlim = [-2.2 2.2];
    mtik = [-2.2 0 2.2];
elseif strcmp(stem, "acflux")
    cmap = cmapAce;
    mlim = [-3 3];
    mtik = [-3 0 3];
elseif strcmp(stem, "alaflux")
    cmap = cmapAla;
    mlim = [-.6 .6];
    mtik = [-0.6 0 0.6];
elseif strcmp(stem, "atpflux")
    cmap = cmapAtp;
    mlim = [0 10];
    mtik = [0 2 4 6 8 10];
elseif strcmp(stem, "nadhflux")
    cmap = cmapNadh;
    mlim = [-6 6];
    mtik = [-5 0 5];
end

disp("Plotting...");
% set figure resolution
wi = 1.4;     % width in inches
hi = 1.5;     % height in inches
dpi = 300;  % dpi resolution
f = figure('PaperUnits', 'inches', 'PaperPosition', [0 0 wi hi]);
hold on;
box off;
a1 = area(T, Vi, 'LineStyle', 'none', 'FaceAlpha', 1);
a2 = area(T, Vo, 'LineStyle', 'none', 'FaceAlpha', 1);
if ~strcmp(stem, "atpflux")
    yline(0, '--', 'LineWidth', 0.5, 'Color', "#8C8C8C");
end
xlim([0 36]);
ylim(mlim);
yticks(mtik);
%yticklabels({'-.6', '-.4', '-.2', '0', '.2', '.4', '.6'});
xticks(0:12:36);
%xlabel('time (h)');
%ylabel('outflux        influx');
ax = gca;
ax.LineWidth = 0.5;
ax.FontSize = 5;
ax.TickDir = 'out';
set(ax,'fontname','Arial');

%ax.FontWeight = 'bold';
colororder(cmap);

%legend(N, 'Location', 'northeast');
disp("Writing plot...");
print(f, stem, "-dsvg");
disp("Done.");
end
