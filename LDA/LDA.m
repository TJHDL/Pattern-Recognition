%加载数据及样本点描绘
data = load('data.txt');
Positive = find(data(:,3)==1);
Negative = find(data(:,3)==0);
X1 = data(Positive,1:2);
X0 = data(Negative,1:2);
hold on;
plot(X1(:,1),X1(:,2),'r+','markerfacecolor',[1,0,0]);
plot(X0(:,1),X0(:,2),'bo','markerfacecolor',[0,0,1]);
grid on;

%计算各均值向量
Pos_ave = mean(X1);     %好瓜样本的均值向量
Neg_ave = mean(X0);     %坏瓜样本的均值向量
All_ave = mean([X1;X0]); %所有示例的均值向量

%建立运算过程中各值所对应的矩阵
p = size(X1,1);
q = size(X0,1);
Pos_ave_p = repmat(Pos_ave,p,1);
Neg_ave_q = repmat(Neg_ave,q,1);
All_ave_p = repmat(All_ave,p,1);
All_ave_q = repmat(All_ave,q,1);

%计算类内散度矩阵
Sw1 = (X1-Pos_ave_p)'*(X1-Pos_ave_p);
Sw0 = (X0-Neg_ave_q)'*(X0-Neg_ave_q);
Sw = Sw1+Sw0;

%计算类间散度矩阵
Sb1 = p*(Pos_ave_p-All_ave_p)'*(Pos_ave_p-All_ave_p);
Sb0 = q*(Neg_ave_q-All_ave_q)'*(Neg_ave_q-All_ave_q);
Sb = Sb1+Sb0;

%计算最大特征值及其对应的最大特征向量(w)
[vector,eigen] = eig(inv(Sw)*Sb);
a = max(max(eigen));
[~,j] = find(eigen==a);
max_vector = vector(:,j);

%计算投影直线方程
k = max_vector(2)/max_vector(1);
b = 0;
x = 0:1;
y = k*x + b;
hold on;
plot(x,y);

%计算示例在直线上的投影点位置
Projection_point1=zeros(p,2);
for i=1:p
    x1 = (X1(i,1)+k*X1(i,2))/(k^2+1);
    y1 = k*x1+b;
    Projection_point1(i,1)=x1;
    Projection_point1(i,2)=y1;
end 

Projection_point0=zeros(q,2);
for i=1:q
    x0 = (X0(i,1)+k*X0(i,2))/(k^2+1);
    y0 = k*x0+b;
    Projection_point0(i,1)=x0;
    Projection_point0(i,2)=y0;
end 
hold on;
plot(Projection_point1(:,1),Projection_point1(:,2),'rx','markerfacecolor',[1,0,0]);
plot(Projection_point0(:,1),Projection_point0(:,2),'bp','markerfacecolor',[0,0,1]);
axis([0 1 0 1]);
xlabel('密度');
ylabel('含糖率');
title('西瓜LDA');