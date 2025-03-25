close all
clear
clc
fitfun = @Chung_Reynolds;
dim=30;
Max_iteration=1000;
SearchAgents_no=30;
lb=-100;
ub=100;
tlt='Chung Reynolds';
i=1;
[Xvalue,Xfood,CNVG] = MESO(SearchAgents_no,Max_iteration,lb,ub,dim,fitfun)
figure,
plot(CNVG,'Color', 'r')
xlim([1 1000]);
