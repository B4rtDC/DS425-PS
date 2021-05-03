%% load up and prepare the data + general settings
clear;close all;clc
d_BTC = flip(readtimetable('./Data/gemini_BTCUSD_1hr.csv'));

%%
indstart = 38000;%44150+5;%45873;
indend = 48032+5;
data = d_BTC(indstart:indend, "Close");
fprintf('start date: %s\nend date: %s\n',datestr(d_BTC.Date(indstart),'yyyy-mm-dd HH:MM'),...
                                       datestr(d_BTC.Date(indend),'yyyy-mm-dd HH:MM'));
% training/validation ratio
vrat = 0.95; limidx = floor(height(data)*vrat);
traindata= data.Close(1:limidx-1);
testdata = data.Close(limidx:end);

% detrend the data
[ff,S,M] = polyfit(1:length(traindata),traindata,3);
%a = ff(1); b = ff(2);
x = (1:height(data))';
traindata_prep = traindata - polyval(ff,x(1:limidx-1),[],M);
testdata_prep = testdata - polyval(ff,x(limidx:end),[],M); 
% illustrate
figure
subplot(1,2,1)
plot(1:limidx-1, traindata);
hold on
plot(limidx:height(data), testdata);
plot(x, polyval(ff,x,[],M))
plot(x(1:limidx-1), traindata_prep)
plot(x(limidx:end), testdata_prep)
title("original and detrended data")

% standardize the data
mu = mean(traindata_prep); s = std(traindata_prep);
traindata_prep = (traindata_prep - mu) / s;
testdata_prep = (testdata_prep - mu) / s;
% illustrate
subplot(1,2,2)
plot(x(1:limidx-1), traindata_prep);
hold on;
plot(x(limidx:end), testdata_prep);
title("standardized and detrended data")

%% INVESTER SETTINGS
% trading costs  (ration of transaction amount)
csell = 0.015;
cbuy = 0.045;
% lead time (Hr)
lead_time = 24;
% starting wallet for both techniques
startcash = 0;
startcoin = 10;


%% OMNISCIENT INVESTOR
w_perf = Wallet(startcash,startcoin);
% select only subset of data as this will be really used for the perfect
% information simulation of the portfolio growth
idx = 1:lead_time:length(testdata);
d = testdata(idx);
d_sell = d * (1-csell);
d_buy = d * (1-cbuy);

% rules:
% crypto in wallet & ratio (X+1)/X < (1 - csell) => sell
% no crypto & ratio ratio (X+1)/X > (1 + cbuy) => buy
V = zeros(length(d)-1,0);
fprintf('OMNISCIENT INVESTOR:\n')
for i = 1:length(d)-1
    if d(i+1) / d(i) < (1-csell)
        w_perf = sell(w_perf,d(i));
    elseif d(i+1) / d(i) > (1+cbuy)
        w_perf = buy(w_perf,d(i));
    end
    fprintf('%i: %1.2f cash, %1.2f BTC\n',i, w_perf.cash, w_perf.crypto)
    V(i) = value(w_perf,d(i));
end
fprintf('\n')

% illustrations
figure 
hold on
yyaxis left
plot(d,'-ok','displayname','rate')
ylabel('rate in $','fontsize', 12);
yyaxis right
ylabel('ratio','fontsize', 12)
plot(2:length(d),d(2:end)./d(1:end-1),'-o','color',[0 0.4470 0.7410],'displayname',sprintf('ratio (day X+%iHr)/(day X)',lead_time)); 
hold on;
plot((1+cbuy).*ones(size(d)),'--k','displayname','buy limit')
plot((1-csell).*ones(size(d)),':k','displayname','sell limit')
legend('fontsize', 12,'location','north');
title('ratio and true rate')

%% RNN INVESTOR
% 1. transform data in input and output vectors
xtrain = traindata_prep(1:end-1)';
ytrain = traindata_prep(2:end)';
xtest = [traindata_prep(end); testdata_prep(1:end-1)]';
ytest = testdata_prep(1:end)';

% 2. Model building stage
% build the network (1D input, 1D output, regression type)
layers = [sequenceInputLayer(1)...
          lstmLayer(100)...
          dropoutLayer(0.1)...
          batchNormalizationLayer()...
          fullyConnectedLayer(1)...
          regressionLayer];
% set the training options
options = trainingOptions('adam', ...
                          'GradientThreshold',1,...
                          'MaxEpochs',1000, ...
                          'LearnRateSchedule','piecewise', ...
                          'LearnRateDropPeriod',750, ...
                          'LearnRateDropFactor',0.9,...
                          'Verbose',0, ...
                          'Plots','none',...
                          'ExecutionEnvironment', 'gpu'...
                      );

%% 3. Model training stage
% actual training (takes a while)
try
    load('trained_net', 'net')
catch
    [net, info] = trainNetwork(xtrain,ytrain,layers,options);
    save('trained_net','net')
end
%
%% quicktest: model performance with actual updates:
resetState(net);
perfnet = predictAndUpdateState(net, xtrain);
% prediction using own value as reinforcement
yperf = zeros(size(ytest));

for i = 1:length(xtest)
    if i <= 1
        [perfnet, yperf(:,i)] = predictAndUpdateState(perfnet, xtest(:,i));
    else
        [perfnet, yperf(:,i)] = predictAndUpdateState(perfnet, yperf(:,i-1));
    end
end
yultimate = zeros(size(ytest));
for i = 1:length(xtest)
    [perfnet, yultimate(:,i)] = predictAndUpdateState(perfnet, xtest(:,i));
end
figure
plot(yperf,'--o'); hold on
plot(ytest,'-');
plot(yultimate,'--x')

%% 4. Use model for predictions
% starting point: initialize the network state, predict on the training data.
resetState(net);
res = zeros(size(ytest));
for i = 1:lead_time:floor(length(ytest)/lead_time)*lead_time
    [net,res(:,i:i+lead_time-1)] = predictme(net,xtest(:,i:i+lead_time-1),lead_time);
end
figure
subplot(1,3,1)
plot(res,'displayname','estimates')
hold on
plot(ytest,'displayname','reality')
legend()
title('model result')
subplot(1,3,2)
plot(res*s+mu,'displayname','estimates')
hold on
plot(ytest*s+mu,'displayname','reality')
legend()
title('model result - destandardized')
subplot(1,3,3)
plot(polyval(ff,length(xtrain)+1:length(xtrain)+length(xtest),[],M) + res*s+mu,'displayname','estimates')
hold on 
plot(polyval(ff,length(xtrain)+1:length(xtrain)+length(xtest),[],M) + ytest*s+mu,'displayname','reality')
legend()
title('model result - destandardized, trended')
%% 5. Use predictions for investment strategy
w_ann = Wallet(startcash,startcoin);
% rules:
% crypto in wallet & ratio (X+1)/X < (1 - csell) => sell
% no crypto & ratio ratio (X+1)/X > (1 + cbuy) => buy
VV = zeros(length(d)-1,0);
fprintf('ANN INVESTOR:\n')
res_invmap = polyval(ff,length(xtrain)+1:length(xtrain)+length(xtest),[],M) + res*s+mu;
for i = 1:length(d)-1
    ind = (i-1)*lead_time+lead_time;
    %res_invmap(ind) / d(i)
    if res_invmap(ind) / d(i) < (1-csell)
        w_ann = sell(w_ann,d(i));
        
    elseif res_invmap(ind) / d(i) > (1+cbuy)
        w_ann = buy(w_ann,d(i));
    end
    fprintf('%i: %1.2f cash, %1.2f BTC\n',i, w_ann.cash, w_ann.crypto)
    VV(i) = value(w_ann,d(i));
end

% portfolio growth
figure
plot([1 length(d)-1],[startcash+startcoin*d(1) startcash+d(end-1)/d(1)*V(1)],'--k','displayname','conservative')
hold on
plot(V,'displayname','perfect info','color',[0 0.4470 0.7410]);
plot(VV,'displayname','ANN info');
legend('location','northwest','fontsize', 12)
ylabel('wallet value')
title(sprintf("portfolio growth (lead time: %i Hr)", lead_time));



%% supporting functions
function [net,res] = predictme(net, xv, lead_time)
    % initialize
    res = [];
    lnet = net;
    % first do the prediction to the lead time
    [lnet, res(:,1)] = predictAndUpdateState(lnet, xv(:,1));
    for i = 2:lead_time
         [lnet, res(:,i)] = predictAndUpdateState(lnet, res(:,i-1));
    end
    % followed by actual update of the net
    for i = 1:lead_time
        net = predictAndUpdateState(net, xv(:,i));
    end
end


