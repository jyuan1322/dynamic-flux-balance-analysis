 function [T, P, Znorm, cfg_pks, cfg_ids, unq_nms] = load13CData(stem, cfg_pk, cfg_nm, outpath)
% Load 13C-NMR data stack.
% The time axis is normalized to the metabolic onset of cfg_pk.
%
% Parameters:
%  - stem is the filename stem of the run
%  - cfg_pk is the 1H chemical shift used for time axis calibration
%  - cfg_nm is the name of the chemical used for time axis calibration
%  - outpath is the filepath for the output
%
% This script requires 1H and 13C spectra, given by <stem + "_1H.xlsx"> and
% <stem + "_13C.xlsx">, to be present in the NMRdata folder.
%
% Returns:
%  - T: time axis vector for NMR data, normalized to overall metabolic rate
%  - P: ppm shift axis vector for NMR data
%  - Znorm: NMR signal, normalized by RMSE of 130-160ppm region of each spectrum
%  - cfg_pks: expected ppm shifts of reference peaks
%  - cfg_ids: numerical indices grouping cfg_pks by compound
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
ppm_min = 0.0;   % minimum chemical shift
ppm_max = 200.0; % maximum chemical shift

t_min = 0;       % minimum time
t_max = 36;      % maximum time

% Scale time axis by glucose metabolic bounds. %
% these references numbers are the output from getTimescale run on the glucose dataset,
% using isocaproate (ppm=0.747) as the reference peak
y1 = 4.6548; % metabolism start time
y2 = 14.3857; % metabolism end time

% get metabolic time scale
[x1 x2] = getTimescale(stem + "_1H.xlsx", cfg_pk, cfg_nm, outpath + stem + "_1H_isc.png");
ts_fcn = @(t) t - x1 + y1;
% ts_fcn = @(t) ((y2 - y1)/(x2 - x1)).*(t - x1) + y1;

% load spectra and scale time axis
fname = stem + "_13C.xlsx";
T = readmatrix(fname, 'Sheet', 1, 'Range', '2:2')';
T = ts_fcn(T(2:end)); % scale time axis
P = readmatrix(fname, 'Sheet', 1, 'Range', 'A:A', 'NumHeaderLines', 3);
Z = readmatrix(fname, 'Sheet', 1, 'Range', 'B4')';

% trim data bounds
Z = Z(T <= t_max & T >= t_min, P >= ppm_min & P <= ppm_max);
P = P(P >= ppm_min & P <= ppm_max);
T = T(T <= t_max & T >= t_min);

% normalize spectrum signal by noise level (RMSE)
noise = zeros(size(Z, 1), 1);
for i = 1:size(Z, 1)
  col = Z(i, P >= 130 & P <= 160); % Empty region in 13C-NMR
  noise(i, 1) = sqrt(mean((col - mean(col)).^2)); % RMSE
end
Znorm = Z ./ noise;

% load reference ppm from cfg sheet
cfg_pks = readmatrix(fname, 'Sheet', 2, 'Range', 'A:A');
cfg_nms = readmatrix(fname, 'Sheet', 2, 'Range', 'B:B', 'OutputType', 'string');
unq_nms = unique(cfg_nms, 'stable');
cfg_ids = zeros(size(cfg_pks, 1), 1);
for i = 1:size(unq_nms, 1)
  choice = cfg_nms == unq_nms(i);
  cfg_ids = cfg_ids + i*(choice);
end

end
