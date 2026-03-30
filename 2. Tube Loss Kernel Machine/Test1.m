clear all;
close all;
%% 
time=0;
tic;
N=500; % Number of training points
Nt=1000; % Number of Testing Points
sigma=0.8; % Sigma in Normal noise
tau=0.80; % Target PI
%% Model Parameter
para.eta =0.1;   % Learning  Rate
para.eps1=0.05;   % Decay Rate
para.iter = 5000; % Number of Iteration
para.kernel=1;    % Linear Kernel
para.p1 = 2^-4;   % RBF kernel parameter (NA)
para.lambda =1;   % Regularization Parameter
para.delta = 0;   % Delta of Tube Loss
para.r=0.6;       % r of Tube Loss
%%
res.error = [];
res.Loss = [];
res.Length = [];
res.CP = [];
%% 
for h=1:10
    b = 0; a = 1;
    rng(h);
    X = a + (b-a).*rand(N,1);  
    Y = sin(X)./X  + sigma*randn(N,1);
    train= X(1:N,:);
    ytrain=Y(1:N,:);
    test = a + (b-a).*rand(Nt,1);
    Y_test = sin(test)./test  + sigma*randn(Nt,1);
    median= sin(test)./test + norminv(0.5,0,sigma);
    [w1,w2,b1,b2,error,loss,CI,coverage_confi] = cce_withR_m1(X,Y,test,Y_test,tau,para);
    res.error(end+1,1) = error;
    res.Loss(end+1,1) = loss;
    res.Length(end+1,1) = CI;
    res.CP(end+1,1) = coverage_confi;
end
%% 
 fprintf("  \n PICP = %2.3f + %2.3f, \n MPIW = %2.3f + %2.3f,\n  Error= %2.3f, %2.3f\n", mean(res.CP),std(res.CP),mean(res.Length),std(res.Length),mean(res.error),std(res.error));
 time= time+toc;