% heatmap of nlz over parameters of drift process

oss = linspace(0.01,0.1,10);
lss = linspace(5,50,10);
ts = linspace(0,90,10)+90;
nlzs = zeros(10);

for i=1:10
   for j=1:10
      tmp = theta;
      tmp.cov(14) = ts(i);
      tmp.cov(15) = log(lss(j));
%       tmp.cov(16) = log(oss(i));
      nlzs(i,j)=gp(tmp,inference_method, mean_function,covariance_function,[],x,y);
   end
end
figure(1);
clf;
heatmap(nlzs, 'XData', lss, 'YData', ts);