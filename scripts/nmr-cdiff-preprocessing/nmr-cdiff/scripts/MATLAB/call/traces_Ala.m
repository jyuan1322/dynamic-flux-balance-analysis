% SCRIPT TO MAKE PLOT OF ALANINE TRACES WITH SPLITTING LINES %
% See Fig3

fn = "alatrace.xlsx"; % xlsx input file, change this
outf = "alatraces.svg";
sns = ["13C" "15N13C"];

addpath("../../../data");

cmap = getColor('splitting');

plotAlaTraces(fn, sns, cmap, outf);
