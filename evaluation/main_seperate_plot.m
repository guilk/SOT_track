clear
close all;
clc

addpath('./util');
addpath('./rstEval');

seqs = config_benchmark('cvpr13');

for ind_seqs = 1:length(seqs)
% for ind_seqs = 1:1
    disp(seqs{ind_seqs}.name);
    per_seperate_plot({seqs{ind_seqs}});
end