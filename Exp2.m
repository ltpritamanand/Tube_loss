clear all;
close all;
%% 
N=500;   % Number of Training Points
Nt=1000;  % Number of Testing Points
tau=0.8;  % Target PICP
%% Parameters
para.eta =0.5;  % Learning Rate
para.eps1=0.09;  % Decay Rate
para.iter = 5000; % Max_iteration
para.kernel=2;    % RBF kernel
para.p1 = 2^-1;    % RBF kernel Parameter
para.lambda = 1;   % Regularization
para.delta = 0;    % delta parameter of tube loss
para.r = 0.6;      % r parameter of tube loss
%%
res.error = [];
res.Loss = [];
res.Length = [];
res.CP = [];
res.cp_err = [];
res.lq = [];
res.uq = [];
res.r = [];
%%
for h=1:1
    b = -4; a = 4;
    rng(2);
    X = a + (b-a).*rand(N,1);
    MX= sin(X)./X;
    noise =  chi2rnd(3,N,1); % Noise generate  from Chisquare distribution
    Y= MX+noise;
    test = a + (b-a).*rand(Nt,1);
    MX_test= sin(test)./test;
    noise_test= chi2rnd(3,Nt,1); 
    Y_test= MX_test+noise_test;
    train= X(1:N,:);
    ytrain=Y(1:N,:);
    
   
%%   
    [w1,w2,b1,b2,error,loss,CI,coverage_confi,m1,m2,m3,m4] = cce_withR_m2(X,Y,test,Y_test,tau,para);  
    res.error(end+1,1) = error;
    res.Loss(end+1,1) = loss;
    res.Length(end+1,1) = CI;
    res.CP(end+1,1) = coverage_confi;
    m12 = length(m1)/length(m2);
    m34 = length(m3)/length(m2);
    lowq = (length(m3)/Nt)*100;
    uppq = 100-(length(m4)/Nt)*100;
    res.lq(end+1,1) = lowq;
    res.uq(end+1,1) = uppq;
    res.r(end+1,1) = para.r;
    %%
    [coverage_confi,lowq,uppq,error,CI];
  fprintf('\n PICP = %2.2f, MPIW= %2.2f\n',coverage_confi,CI);
  
end 
