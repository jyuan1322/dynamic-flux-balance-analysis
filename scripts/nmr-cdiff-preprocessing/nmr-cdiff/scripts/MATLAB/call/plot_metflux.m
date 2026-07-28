% SCRIPT TO MAKE NMR STACK %
% OUTPUT: pyrflux.png %

disp("Loading parameters...");

stem = "atpflux"; % xlsx input file, change this
snin = "in"; % "in"
snout = "out"; % "out"

addpath("../../../data");

plotFluxBidirectional(stem, snin, snout);
