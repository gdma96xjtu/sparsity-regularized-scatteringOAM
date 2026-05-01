clc;clear all;close all;
load('spr_real.mat');
load('spr_imag.mat');
figure;
h=bar(spr_real);
hold on
h=bar(spr_imag);

set(gca,'XTickLabel',[],'YTickLabel',[]);
axis normal;
box off
set(gca, 'LooseInset', [0,0,0,0]);
% legend
% saveas(gca,strcat('ComplexUniform8bit.png'));