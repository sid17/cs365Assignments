## Data (200 points)
t1 = textread('dataset/h1_train_big.csv', '%s', 'delimiter', '\n');
A=[];
B=[];
for i = 1:length(t1)
    j = findstr(t1{i}, ',')(1);
    A= [A,str2num(t1{i}(1:j - 1))];
    B =[B,str2num(strtrim(t1{i}(j + 1:end)))];
end
x = A;
y = B;
## constant
pp0 = polyfit (x, y, 0);
## linear
pp1 = polyfit (x, y,  1);
## quadratic
pp2 = polyfit (x, y,  2);
## cubic
pp3 = polyfit (x, y,  3);
## degree 5
pp5 = polyfit (x, y,  5);
## degree 9
pp9 = polyfit (x, y, 9);

## Plot
xx = linspace (min(A)-2, max(A)+2, 10000);
y0 = polyval (pp0, xx);
y1 = polyval (pp1, xx);
y2 = polyval (pp2, xx);
y3 = polyval (pp3, xx);
y5 = polyval (pp5, xx);
y9 = polyval (pp9, xx);


plot (x, y, '.', xx, [y0],'LineWidth',2)
legend ({'data', 'order 0'})
title('Polynomial Fitting: Degree 0')
xlabel('Time') % x-axis label
ylabel('Height') % y-axis label

print -djpg image0_big.jpg


plot (x, y, '.', xx, [y1],'LineWidth',2)
legend ({'data', 'order 1'})
title('Polynomial Fitting: Degree 1')
xlabel('Time') % x-axis label
ylabel('Height') % y-axis label

print -djpg image1_big.jpg

plot (x, y, '.', xx, [y2],'LineWidth',2)
legend ({'data', 'order 2'})
title('Polynomial Fitting: Degree 2')
xlabel('Time') % x-axis label
ylabel('Height') % y-axis label

print -djpg image2_big.jpg


plot (x, y, '.', xx, [y3],'LineWidth',2)
legend ({'data', 'order 3'})
title('Polynomial Fitting: Degree 3')
xlabel('Time') % x-axis label
ylabel('Height') % y-axis label

print -djpg image3_big.jpg


plot (x, y, '.', xx, [y5],'LineWidth',2)
legend ({'data', 'order 5'})
title('Polynomial Fitting: Degree 5')
xlabel('Time') % x-axis label
ylabel('Height') % y-axis label

print -djpg image5_big.jpg


plot (x, y, '.', xx, [y9],'LineWidth',2)
legend ({'data', 'order 9'})
title('Polynomial Fitting: Degree 9')
xlabel('Time') % x-axis label
ylabel('Height') % y-axis label
% plot (x, y, '.', xx, [y1],'LineWidth',2)
print -djpg image9_big.jpg

t1 = textread('dataset/h1_validate.csv', '%s', 'delimiter', '\n');
A=[];
B=[];
for i = 1:length(t1)
    j = findstr(t1{i}, ',')(1);
    A= [A,str2num(t1{i}(1:j - 1))];
    B =[B,str2num(strtrim(t1{i}(j + 1:end)))];
end
x_validate = A;
y_validate = B;

y0 = polyval (pp0, x_validate);
y1 = polyval (pp1, x_validate);
y2 = polyval (pp2, x_validate);
y3 = polyval (pp3, x_validate);
y5 = polyval (pp5, x_validate);
y9 = polyval (pp9, x_validate);

N=size(y0,2)

X=[sqrt(sum((y_validate-y0).^2)/N),sqrt(sum((y_validate-y1).^2)/N),sqrt(sum((y_validate-y2).^2)/N),sqrt(sum((y_validate-y3).^2)/N),sqrt(sum((y_validate-y5).^2)/N),sqrt(sum((y_validate-y9).^2)/N)]
Y=[0,1,2,3,5,9]
plot(Y,X,'LineWidth',2)
title('Plot of Error vs Polynomial Degree')
xlabel('Degree') % x-axis label
ylabel('Error') % y-axis label
print -djpg error_s_v.jpg

fprintf('|Degree|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|\n');
fprintf('|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|\n');
fprintf('|%d|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|\n',0,pp0(1),0,0,0,0,0,0,0,0,0);
fprintf('|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|\n');
fprintf('|%d|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|\n',1,pp1(2),pp1(1),0,0,0,0,0,0,0,0);
fprintf('|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|\n');
fprintf('|%d|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|\n',2,pp2(3),pp2(2),pp2(1),0,0,0,0,0,0,0);
fprintf('|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|\n');
fprintf('|%d|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|\n',3,pp3(4),pp3(3),pp3(2),pp3(1),0,0,0,0,0,0);
fprintf('|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|\n');
fprintf('|%d|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|\n',5,pp5(6),pp5(5),pp5(4),pp5(3),pp5(2),pp5(1),0,0,0,0);
fprintf('|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|\n');
fprintf('|%d|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|\n',9,pp9(10),pp9(9),pp9(8),pp9(7),pp9(6),pp9(5),pp9(4),pp9(3),pp9(2),pp9(1));
fprintf('\n\n|Training Set|Deg0|Deg1|Deg2|Deg3|Deg5|Deg9|\n');
fprintf('|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|\n');
fprintf('|Large|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|%0.3f|\n',X(1),X(2),X(3),X(4),X(5),X(6));
