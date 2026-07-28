% SCRIPT TO MAKE NMR STACK %
% OUTPUT: render.png %
disp("Loading parameters...");
fn = "20210525_13CGlc_15NLeu_13Ca.xlsx"; % "20210519_13CGlc_13Cc.xlsx"; %"20210525_13CGlc_15NLeu_13Ca.xlsx"; % xlsx input file, change this
outf = "20210525_13CGlc_15NLeu_13Cwf.png";

ppm_min = 53.0;   % minimum ppm shift displayed
ppm_max = 53.8; % maximum ppm shift displayed
t_min = 0;     % minimum time displayed ; 26
t_max = 24;    % maximum time displayed ; 50
yaw = 10;     % camera rotation angle, yaw left
pitch = 30;    % camera rotation angle, pitch down

addpath("../../../data");

cmap = getColor('splitting'); % default, paired
if contains(fn, '15N'); % colors for 15N-coupled peaks
    cmap = cmap(3:end, :);
end

plotRegionAla(fn, ppm_min, ppm_max, t_min, t_max, yaw, pitch, cmap, outf);
