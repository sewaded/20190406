tic
clc
format long
% data20190406=xlsread('C:\20190406.xlsx',1,'E2:BF1029');
load data20190406
data=data20190406; %读数据
l=length(data);
train=floor( .1 *l); %这个train就是训练集的总量
target=1;
for o1=1:train
    ran=ceil(rand*(l-o1+1)); %随机取train集
    x(:,o1)=data(ran,5:58)'; 
    y(:,o1)=2*(data(ran,4)==target)'-1;
    data(ran,:)='';
end
[d,n]=size(x);
ran=100;
n1=1;
% b1=2*ran*(rand(n1,1)-.5);
prec=100; %期望的误差收敛要求，但事实上碰不得
step0=1; %初始步长，我后面用的是适应性步长
y1=2*sigmoid(w1*[x;ones(1,n)])-1;
%     loss=sum(power(y1-y,2));
Q=loss(w1);
r=0; %迭代次数记录
Qmin=Q;
W=zeros(n1,d+1); 
for o=1:6
    if o<6
        disp(strcat('Round',num2str(o)))
        pause(1)
    end
    step=step0; %步长还原
    reverse=1; %这个东西很棒
    %Gravity-Reverse方法，这是我自己平时用的一个能使得梯度法或是类梯度法自动跳出局域最优并尽可能多的搜索全域的方法
    w1=2*ran*(rand(n1,d+1)-.5); %重新初始化
    Q=loss(w1); %loss的function
    while(Q>prec && r<o*600) 
        Qo=Q; %记录当前误差值
    %     y1=2*sigmoid(w1*[x;ones(1,n)])-1;
    % %     loss=sum(power(y1-y,2));
    %     Q=loss(w1);
        for o1=1:n1
            for o2=1:d
                dw1(o1,o2)=2*(y1(o1,o2)-y(o1,o2))*y(o1,o2)*(1-y(o1,o2))*sum(x(o2,:)); %求梯度
            end
            for o2=d+1:d+1
                dw1(o1,o2)=2*(y1(o1,o2)-y(o1,o2))*y(o1,o2)*(1-y(o1,o2))*n; %我的b（bias）是放在w（weight）的最后一行的
            end
        end
        w1o=w1;
        w1=w1+step*reverse*dw1; % 参数更新
        y1=2*sigmoid(w1*[x;ones(1,n)])-1; %我用的是sigmoid，sigmoid求导比较好使
    %     loss=sum(power(y1-y,2));
        Q=loss(w1); %目标值更新
        step=step*1.1; %步长自动增加，后面会有减小判定，这样可以做到适应性步长
        Qmin=min(Qmin,Q);
        disp(strcat('Q:',num2str(Q),' & Qmin: ',num2str(Qmin),' & step: ',num2str(step))) %显示
        W=W+(Q<Qmin)*(w1-W); %跟踪最优参数组
        if Q*reverse>Qo*reverse %目标变差的话就取消这一步变动并且使步长打折
            w1=w1o;
            step=step/2;
        end
        if step<.00000001 || step>100 %判定收敛则Gravity-Reverse，跳出当前局域

            step=step0;
            reverse=-1*reverse;
        end
        r=r+1; %记录迭代次数
    end
end
disp('Test Phase') %下面是验证
toc
pause(1)
r=0;
x=data(:,5:58)';
y=2*(data(:,4)==target)'-1;
[d,n]=size(x);
for o1=1:n
    test(o1)=round(sigmoid(W*[x(:,o1);ones(1,1)]))==(y(o1)>0);
end
mean(test)
