tic
clear all;
close all;
%%
%  machinecpu=xlsread('spnbmd1.csv');
%  n = length(machinecpu);
%  x = machinecpu(:,1:end-1);
%  y1 = machinecpu(:,end); 
 %%
 load servo.mat;
 machinecpu =servo;
 x = machinecpu(:,2:end);
 y1 = machinecpu(:,1);
  n=length(y1);
 t= randperm(n);
 x=x(t,:);
 y1= y1(t,:);
  ktest=  x(151:end,:);
  yktest = y1(151:end,:);%%
  x= x(1:150,:);
  y1= y1(1:150,:);
   n = length(y1);
tau=0.80;
%%
% para.eta =0.2; 
% para.eps1=0.001; %0.0001 
% para.iter = 5000;
% para.kernel=2;
% para.p1 = 2^3;
% para.lambda = 0.009;
% para.delta = 0.01;
% para.r=0.6;

para.eta =0.6; 
para.eps1=0.05; %0.0001 
para.iter = 5000;
para.kernel=2;
para.p1 = 2^-7;
para.lambda = 0;
para.delta = 0.13;
para.r=0.5;
r=para.r;
%%
res.error = [];
res.Loss = [];
res.Length = [];
res.CP = [];
res.cp_err = [];
res.lq = [];
res.uq = [];
%%
X = x;
Y = y1;
fold=10;
indx = [0:fold];
indx = floor(n*indx/fold);
w1 = zeros(indx(end-1),fold);
w2 = zeros(indx(end-1),fold);

for i=1:fold
    Ctest = X(indx(i)+1:indx(i+1),:);
    dtest = Y(indx(i)+1:indx(i+1));
    
    Ctrain = X(1:indx(i),:);
    Ctrain = [Ctrain;X(indx(i+1)+1:n,:)];
    dtrain = [Y(1:indx(i));Y(indx(i+1)+1:n,:)];
    
    train= Ctrain;
    ytrain=dtrain;
    
    test =Ctest;
    Y_test = dtest;
    [w1(:,i),w2(:,i),b1(i),b2(i),error,loss,CI,coverage_confi,m1,m2,m3,m4] = cce_withR_m1(train,ytrain,test,Y_test,tau,para);

    %[w1,w2,b1,b2,error,loss,CI,coverage_confi,m1,m2,m3,m4] = cce1(train,ytrain,test,Y_test,tau,para);
    err1 = abs(coverage_confi - tau);
    lowq = (length(m3)/length(Y_test))*100;
    uppq = 100-(length(m4)/length(Y_test))*100;
    fprintf("Coverage=%f,  Coverage Length=%f \n",coverage_confi,CI);
   % fprintf("Error = %f, Conf. Loss= %f Conf. Length=%f, Cov. Prob= %f, CP_error=%f \n",error,loss,CI,coverage_confi,err1);
    res.CP(end+1,1) = coverage_confi;
    res.cp_err(end+1,1) =  abs(coverage_confi - tau);
    res.Length(end+1,1) = CI;
    res.Loss(end+1,1) = loss;
end
%%
fprintf("Coverage: Mean=%f, Std=%f\n",mean(res.CP),std(res.CP));
fprintf("Error: Mean=%f, Std=%f\n",mean(res.cp_err),std(res.cp_err));
fprintf("Length: Mean=%f, Std=%f\n",mean(res.Length),std(res.Length));
fprintf("Loss: Mean=%f, Std=%f\n",mean(res.Loss),std(res.Loss));
toc;

w1 =mean(w1,2);
w2=mean(w2,2);
b1=mean(b1);
b2=mean(b2);

if(para.kernel==1)
    for i=1:size(ktest,1)
        for j=1:length(dtrain)
            Htest(i,j) = svkernel('linear',ktest(i,:), x(j,:),para.p1);
        end
    end
end
if(para.kernel==2)
    for i=1:size(ktest,1)
        for j=1:length(dtrain)
            Htest(i,j) = svkernel('rbf',ktest(i,:), x(j,:),para.p1);
        end
    end
end
if(para.kernel==3)
    for i=1:size(ktest,1)
        for j=1:length(dtrain)
            Htest(i,j) = svkernel('fourier',ktest(i,:), x(j,:),para.p1);
        end
    end
end
    

%%
m1= find(yktest < (Htest*w1+b1) & yktest > (Htest*w2+b2) &   yktest > (Htest*(r*w1+(1-r)*w2)+ (r*b1+(1-r)*b2)));
m2 = find(yktest < (Htest*w1+b1) & yktest >  (Htest*w2+b2) &   yktest < (Htest*(r*w1+(1-r)*w2)+ (r*b1+(1-r)*b2)));
m3 =  find(yktest < (Htest*w2+b2));
m4=  find(yktest > (Htest*w1+b1));
coverage_confi = (length(m1)+length(m2))/size(ktest,1)
err1=abs(tau-coverage_confi);
testCI=sum((Htest*(w1-w2)+(b1-b2)))/length(yktest)
