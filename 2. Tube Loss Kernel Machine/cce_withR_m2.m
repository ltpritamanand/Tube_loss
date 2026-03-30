function [w1,w2,b1,b2,error,loss,CI,coverage_confi,m1,m2,m3,m4,count,pred1,pred2] = cce_withR_m(train,y,test,ytest,tau,para)
n=size(train,1);
e=ones(n,1);
e2=ones(length(ytest),1);
kernel=para.kernel;
p1= para.p1;
r = para.r;
eta=para.eta;
eps1= para.eps1;
max_iter=para.iter;
lambda=para.lambda;
delta = para.delta;
count=  ones(max_iter,1);
%% Intialization
rng(1);
w1= rand(n,1)*0; b1 = 6; % Intialization effect the numerical results
w2= rand(n,1)*0; b2=  0;
 f_gradw = zeros(n,n);
 f_gradb = zeros(n,1);
 s_gradw = zeros(n,n);
 s_gradb = zeros(n,1);
 flag=0;
%% Kernel construction
if(kernel==1)
    for i=1:n
        for j=1:n
            H(i,j) = svkernel('linear',train(i,:), train(j,:),p1);
        end
    end
end
if(kernel==2)
    for i=1:n
        for j=1:n
            H(i,j) = svkernel('rbf',train(i,:), train(j,:),p1);
        end
    end
end


if(kernel==3)
    for i=1:n
        for j=1:n
            H(i,j) = svkernel('fourier',train(i,:), train(j,:),p1);
        end
    end
end


%%
if(kernel==1)
    for i=1:size(test,1)
        for j=1:n
            Htest(i,j) = svkernel('linear',test(i,:), train(j,:),p1);
        end
    end
end
if(kernel==2)
    for i=1:size(test,1)
        for j=1:n
            Htest(i,j) = svkernel('rbf',test(i,:), train(j,:),p1);
        end
    end
end
if(kernel==3)
    for i=1:size(test,1)
        for j=1:n
            Htest(i,j) = svkernel('fourier',test(i,:), train(j,:),p1);
        end
    end
end



%%
fprintf("\nMinimizing the  Continious Confidendence Loss");

for i=1:max_iter  
m1= find(y < H*w1+b1 & y > H*w2+b2 &   y > H*(r*w1+(1-r)*w2)+ (r*b1+(1-r)*b2));
m2 = find(y < H*w1+b1 & y >  H*w2+b2 &   y < H*(r*w1+(1-r)*w2)+ (r*b1+(1-r)*b2));
m3 =  find(y < H*w2+b2);
m4=  find(y > H*w1+b1);

loss1 = sum((1-tau)*(H(m1,:)*w1+b1)-y(m1));
loss2= sum((1-tau)*( y(m2)-(H(m2,:)*w2+b2)));
loss3= sum(tau*(H(m3,:)*w2+b2-y(m3)));
loss4= sum(tau*(y(m4)-(H(m4,:)*w1+b1)));
loss_inf(i)=(loss1+loss2+loss3+loss4)/length([m1;m2;m3;m4]) ;



f_gradw(:,m1) = H(:,m1)*(1-tau);
f_gradw(:,m2) = zeros(n,size(m2,1));
f_gradw(:,m3) = zeros(n,size(m3,1));
f_gradw(:,m4) = -H(:,m4)*(tau);


f_gradb(m1) = ones(size(m1,1),1)*(1-tau);
f_gradb(m2) = zeros(size(m2,1),1);
f_gradb(m3) = zeros(size(m3,1),1);
f_gradb(m4) = -ones(size(m4,1),1)*(tau);

s_gradw(:,m1) = zeros(n,size(m1,1));
s_gradw(:,m2) = H(:,m2)*(tau-1);
s_gradw(:,m3) = H(:,m3)*(tau);
s_gradw(:,m4) = zeros(n,size(m4,1));

s_gradb(m1) = zeros(size(m1,1),1);
s_gradb(m2) = ones(size(m2,1),1)*(tau-1);
s_gradb(m3) =  ones(size(m3,1),1)*(tau);
s_gradb(m4) =  zeros(size(m4,1),1);



   if ( min((H*w1+b1) - (H*w2+b2))> 0)
       count(i) = 0;
   else
       count(i) =1 ;
   end
   
   
   if(i>5)
       if(sum(count(i-5:i))==0)
       flag=1;
       else
           flag=0;
       end
   end
   
w2= w2-eta*( ( (delta/n)* (flag)*(-H'*e) +  (lambda/n)*w2) +mean(s_gradw,2));
b2 = b2-eta*( (delta/n)* (flag)*(-e'*e) + mean(s_gradb));
w1= w1-eta*( ( (delta/n)*(flag)* (H'*e) +  (lambda/n)*w1) + mean(f_gradw,2));
b1 = b1-eta*( (delta/n)*(flag)*(e'*e) + mean(f_gradb));
eta= eta/(1+ eps1);
if (eta <1e-5)
    break
end
  grad_inf(i)=norm([f_gradb,f_gradw])+norm([s_gradb,s_gradw]);

end
 






m1= find(ytest < (Htest*w1+e2*b1) & ytest > (Htest*w2+e2*b2) &   ytest > (Htest*(r*w1+(1-r)*w2)+ e2*(r*b1+(1-r)*b2)));
m2 = find(ytest < (Htest*w1+e2*b1) & ytest >  (Htest*w2+e2*b2) &   ytest < (Htest*(r*w1+(1-r)*w2)+ e2*(r*b1+(1-r)*b2)));
m3 =  find(ytest < (Htest*w2+e2*b2));
m4=  find(ytest > (Htest*w1+e2*b1));
coverage_confi = (length(m1)+length(m2))/size(test,1);
error=abs(tau-coverage_confi);
loss1 = sum((1-tau)*(Htest(m1,:)*w1+b1)-ytest(m1));
loss2= sum((1-tau)*( ytest(m2)-(Htest(m2,:)*w2+b2)));
loss3= sum(tau*(Htest(m3,:)*w2+b2-ytest(m3)));
loss4= sum(tau*(ytest(m4)-(Htest(m4,:)*w1+b1)));
loss=(loss1+loss2+loss3+loss4)/length([m1;m2;m3;m4]) ;
CI=sum((Htest*(w1-w2)+(b1-b2)))/length(ytest);
%% Plotting results
plot(test,ytest,'ro');
[vv,jj]=sort(test);
hold on;
plot(test(jj),Htest(jj,:)*w1+b1,'b-','LineWidth',2);
hold on;
plot(test(jj),Htest(jj,:)*w2+b2,'b-','LineWidth',2);
hold on;
plot(test(jj),(Htest(jj,:)*(r*(w1+w2))+(r*(b1+b2))),':','LineWidth',3);
pred1 = Htest*w1+b1;
pred2 = Htest*w2+b2;
end