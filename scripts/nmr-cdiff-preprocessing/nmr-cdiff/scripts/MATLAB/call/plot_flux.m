% SCRIPT TO MAKE NMR STACK %
% OUTPUT: render.png %

fn = "fluxes.xlsx"; % xlsx input file, change this
outf = "fluxes";    % filename for output
data_sheetname = "fluxes";
lbdv_sheetname = "fluxlb";
ubdv_sheetname = "fluxub";

addpath("../../../data"); % add "data" directory to path, change if necessary

cmap = getColor('custom');  % default, paired
cmap = cmap([1 2 7 6 17 10 12 13 15 8 9 4 18 26 27 22 23 21 20 24 18], :);

plotGroupedRxns(fn, data_sheetname, lbdv_sheetname, ubdv_sheetname, cmap, outf);
