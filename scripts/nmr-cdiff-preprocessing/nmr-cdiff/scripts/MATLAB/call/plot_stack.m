% Plot 13C stacks.
% Time axis normalized to isocaproate metabolic onset.
% Specify metabolite with "met", add use cases with elseif.

met = "glc"; % Metabolite

cfg_pk = 0.8642;
if strcmp(met, "glc")
    stem = "20210519_13CGlc";
    cidx = [1 7 10 2 8 5 9];% [1 7 4 10 2 8 5 9 16];
elseif strcmp(met, "pro")
    stem = "20201230_13CPro";
    cidx = [14 15];
elseif strcmp(met, "leu")
    stem = "20210322_13CLeu";
    cfg_pk = 0.76;
    cidx = [11 12 13 4 2 17];
end

outpath = "";
cfg_nm = "Isocaproate";

addpath("../../../data");

cmap = getColor('custom'); % default, paired
cmap = cmap(cidx, :);

[T, P, Znorm, cfg_pks, cfg_ids, unq_nms] = load13CData(stem, cfg_pk, cfg_nm, outpath);
plotStacks(T, P, Znorm, cfg_pks, cfg_ids, cmap, outpath);
