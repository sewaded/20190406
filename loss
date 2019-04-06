function [loss]=loss(w1)
    load data20190405
    data=data20190405;
    x=data(:,5:58)';
    y=2*data(:,3)'-3;
    [~,n]=size(x);
    y1=2*sigmoid(w1*[x;ones(1,n)])-1;
    loss=sum(power(y1-y,2));
end
