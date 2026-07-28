function [t5p t95p] = getTimescale (fn, cfg_pk, cfg_nm, outf)
% Calculate the timing and rate of metabolism using trajectory of cfg_pk.
% Returns time when 5% and 95% of cfg_pk is metabolized.
%
% Parameters:
%  - fn is the filename of the 1H spectrum
%  - cfg_pk is the 1H chemical shift used for time axis calibration
%  - cfg_nm is the name of the chemical used for time axis calibration
%  - outf is the filename of the output file
%
% Outputs:
%  - <outf> is a plot of the logistic curve fit for cfg_pk,
%    to be used for visual validation
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

input = false; % is cfg_pk an input metabolite, like glucose or proline?

r = 0.01;  % number of RMSE for minimum peak prominence = 3
tolerance = 0.01; % 0.5 for 13C, 0.05 for 1H, range around reference ppm

% load 1H spectra
T = readmatrix(fn, 'Sheet', 1, 'Range', '2:2')';
T = T(2:end);
P = readmatrix(fn, 'Sheet', 1, 'Range', 'A:A', 'NumHeaderLines', 3);
Z = readmatrix(fn, 'Sheet', 1, 'Range', 'B4')';

% normalize spectra
meanDev = zeros(size(Z, 1), 1);
for i = 1:size(Z, 1)
    col = Z(i, :);
    meanDev(i, 1) = sqrt(mean((col - mean(col)).^2));
end
meanMeanDev = mean(meanDev);
Znorm = Z ./ meanMeanDev;

% initialize some variables for peak ID
pks = zeros(size(T, 1), 1);   % vector of peak heights
itg = zeros(size(T, 1), 1);   % vector of peak integrals
tim = zeros(size(T, 1), 1);   % vector of peak timepoints
inc = zeros(size(T, 1), 1);   % include if peak is adequately prominent

% identify peaks
for i = 1:size(Znorm,1)
    Zvec = Znorm(i, :);
    pk = 0;
    incl = 0;
    thresh = r;
    range = P > cfg_pk - tolerance & P < cfg_pk + tolerance;
    result = max(Zvec .* range');
    if result >= thresh
        pks(i) = result;
        itg(i) = -1*trapz(P(range), Zvec(range));
        tim(i) = T(i);
        inc(i) = 1;
    end
end

%[ht, i] = min(pks);
%inc(1:i-1) = 0;
inc = logical(inc);
all_pks = pks(inc);  % filtered vector of heights
all_itg = itg(inc);  % filtered vector of integrals
all_tim = tim(inc);  % filtered vector of timestamps

%all_pks = all_itg;   % comment this line to use peak heights

% logistic fit and plot
f = figure('Position', [10 10 900 780]);
hold on;
range = 0:0.5:64;
basecolor = [0, 0.4470, 0.7410];
scatter(all_tim, all_pks, 30, 'filled', 'MarkerFaceColor', basecolor);
if (input == true)
    g = fittype( @(L, k, x0, C, x) L./(1 + exp(-1*k*(x - x0))) + C);
    fit_fcn = fit(all_tim, all_pks, g, 'start', [0.25 0.25 21 0.05]);
else
    g = fittype( @(L, k, x0, x) L./(1 + exp(-1*k*(x - x0))) );
    fit_fcn = fit(all_tim, all_pks, g, 'start', [15 0.2 5]);
end
plot(range, fit_fcn(range), '--', 'Color', basecolor, 'linewidth', 2);
coeffs = coeffvalues(fit_fcn);
L = coeffs(1);
k = coeffs(2);
x0 = coeffs(3);
disp([L k x0]);
% get time bounds of metabolism using inverse logistic
invlog = @(d) (log(d*L) - log((1-d)*L))/k + x0;
t5p = invlog(0.05); %t5p
t95p = invlog(0.95);  %t95p
xline(t5p, ':k', 't=' + string(t5p), 'HandleVisibility', 'off');
xline(t95p, ':k', 't=' + string(t95p), 'HandleVisibility', 'off');

disp([t5p t95p]);

% ax parameters
xlim([0 72]);
ylabel('peak signal');
xlabel('time (h)');
ylim([0 inf]);
ax = gca;
ax.LineWidth = 3;
ax.FontSize = 42;
ax.FontWeight = 'bold';

saveas(f, outf);
end
