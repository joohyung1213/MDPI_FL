clearvars
full = csvread('Full_power.csv');
proposed = csvread('Proposed_power.csv');
static = csvread('Static_power.csv');
rand = csvread('Random_power.csv');

%full = transpose(full);
%proposed = transpose(proposed);
%static = transpose(static);

x = [0:1:999];
p1 = plot(x, full);
hold on
p2 = plot(x, static);
hold on
p4 = plot(x, proposed);
hold on
p3 = plot(x, rand);

hline = refline([0 2000]);
hline.Color = 'black';
hline.LineWidth = 1.5;
hline.LineStyle = '--';
% full
p1.LineWidth = 4;
str = '#C22D5A';
%str= '#7E2F8E';
color = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
p1.Color = color;

% static
p2.LineWidth = 4;
str = '#3DDBA8';
% str = '#f2a900';
color = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
p2.Color = color;

%proposed
p3.LineWidth = 4;
%str = '#C22D5A';
str= '#f2a900';
color = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
p3.Color = color;

%rand
p4.LineWidth = 6;
str = '#7E2F8E';
color = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
p4.Color = color;

grid on

ax = gca;

xlabel('시간 (sec)') 
ylabel('큐 길이 (MB)') 
xticks(0:100:1000)
yticks(0:2000:10000)
ax.FontSize = 30;

ylim([0, 10000])
ax.YAxis.Exponent = 3;

lgd = legend([p1 p2 p3 p4], {'최대 선택', '고정 개수 선택', '제안하는 클라이언트 수 결정 + 임의 선택', '제안하는 클라이언트 수 결정 + 제안하는 클라이언트 선택'});
ldg.FontSize = 40;