% RNG Seed
seed = rng(999);

%Load 

%% Setup the Import Options and import the data

opts = delimitedTextImportOptions("NumVariables", 13);

% Specify range and delimiter
opts.DataLines = [1, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["Var1", "Var2", "Var3", "Var4", "start_time", "end_time", "idle_time", "mouse_wheel", "mouse_wheel_click", "mouse_click_left", "mouse_click_right", "mouse_movement", "keystroke"];
opts.SelectedVariableNames = ["start_time", "end_time", "idle_time", "mouse_wheel", "mouse_wheel_click", "mouse_click_left", "mouse_click_right", "mouse_movement", "keystroke"];
opts.VariableTypes = ["string", "string", "string", "string", "datetime", "datetime", "double", "double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, ["Var1", "Var2", "Var3", "Var4"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["Var1", "Var2", "Var3", "Var4"], "EmptyFieldRule", "auto");
opts = setvaropts(opts, "start_time", "InputFormat", "dd.MM.yyyy HH:mm:ss");
opts = setvaropts(opts, "end_time", "InputFormat", "dd.MM.yyyy HH:mm:ss");

%%% Import the data

% Get a list of all txt files in the current folder, or subfolders of it.

fds = fileDatastore('C:\Users\Tom\Desktop\Forecast Project\EPMDataset\EPM Dataset 2\Data\Processes\Session 3\', 'ReadFcn', @importdata);

fullFileNames = fds.Files;

numFiles = length(fullFileNames);

% Loop over all files reading them in and plotting them.



for k = 1 : numFiles

    fprintf('Now reading file %s\n', fullFileNames{k});

    data = readtable(fullFileNames{k}, opts);
    
    % Calculate duration

    start_time = datetime(data{:,1});

    end_time = datetime(data{:,2});

    full_duration = end_time - start_time;

    full_duration = seconds(full_duration);

    sum_duration = cumsum(full_duration);

    sum_duration = reshape(sum_duration, 1, []);
    
    % Calculate activity

    full_activity = data{:,5} + data{:,5} + data{:,7} + data{:,8};

    sum_activity = cumsum(full_activity);

    sum_activity = reshape(sum_activity, 1, []);


    % Combine data 

    newdata = [sum_duration(:), full_activity(:)];


    %Aggregate Data

    [unique_durations, ~, subs] = unique(sum_duration); 

    uniques_activity_sum = accumarray(subs, sum_activity);

    newdata_aggregated = [unique_durations(:), uniques_activity_sum(:)];
    
    % Scaling data to 10sec intervals

    unique_durations_ceiled = ceil(unique_durations / 20) ;

    [unique_durations_ceiled, ~, subs] = unique(unique_durations_ceiled); 

    uniques_activity_sum_ceiled = accumarray(subs, uniques_activity_sum);

    newdata_aggregated_ceil = [unique_durations_ceiled(:), uniques_activity_sum_ceiled(:)];


    % fill in zeroes for missing intervals

    pos = 0;
    
    for ia = unique_durations_ceiled
    
    pos = pos + 1;
    
        if pos == numel(unique_durations_ceiled)
            break
        end
    
    ia = ia; %ok
    
    ib = unique_durations_ceiled(pos +1);
    
    diff = ia+1:ib-1;
    
    
    zero = zeros(numel(diff),1);
    newrows = [diff', zero];
    
    
    newdata_aggregated_ceil = [newdata_aggregated_ceil;newrows];
    
    newdata_aggregated_ceil = sortrows(newdata_aggregated_ceil); 
    
    end
    
    newdata_aggregated_ceil(:,1) = [];

    finals = newdata_aggregated_ceil.'; 
   
    %%normalize
    finals = normalize(finals, 'scale');
      
    final{k} = finals;
    
    
    
    
end   

for k = 1 : numFiles


    % Train test Split

    numTimeStepsTrain{k} = floor(0.9*numel(final{k}));

    dataTrain{k} = final{k}(1:numTimeStepsTrain{k}+1);
    dataTest = final{k}(numTimeStepsTrain{k}+1:end);

    
    %Prepare Predictors and Responses

    
    XTrain{k} = dataTrain{k}(1:end-1);
    YTrain{k} = dataTrain{k}(2:end);

    XTest{k} = dataTest(1:end-1);
    YTest{k} = dataTest(2:end);
end




%% Clear temporary variables
    clear opts
    
        
% define network architecure

numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    reluLayer
    regressionLayer];


options = trainingOptions('adam', ...
    'MaxEpochs',100, ...
    'Shuffle','every-epoch',... 
    'MiniBatchSize',200, ...
    'GradientThreshold',2, ...
    'InitialLearnRate',0.01, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',15, ...
    'LearnRateDropFactor',0.001, ...
    'Verbose',0, ...
    'Plots','training-progress');
    


%train

net = trainNetwork(XTrain,YTrain,layers,options);


for i = 1 : numFiles
        
    %Forecast Future Time Step
    net = resetState(net);
    net = predictAndUpdateState(net,XTrain(i));
   
    XTest_pos = cell2mat(XTest(i));
    
    YPred = [];
    
    numTimeStepsTest = numel(XTest_pos);
    
    for j = 1:numTimeStepsTest
        [net,YPred(:,j)] = predictAndUpdateState(net,XTest_pos(:,j),'ExecutionEnvironment','cpu');
        
        % Baseline switch
        %YPred(:,j) = 0;
        
    end
    
    % offset correction
    YPred(:,1) = [];
    YTest{i}(:,end) = [];
    
    % calcuoate RMSE 
    rmse(i) = sqrt(mean((YPred-YTest{i}).^2));
    
    %plot
    if i == 40     
        %forcast
        figure
        plot(dataTrain{i}(1:end-1))
        hold on
        
        numTimeStepsTest = numTimeStepsTest - 1;
        idx = numTimeStepsTrain{i}:(numTimeStepsTrain{i}+numTimeStepsTest);
        
        
        plot(idx,[final{i}(numTimeStepsTrain{i}) YPred],'.-')
    
        hold off
        xlabel("20 Second Intervals")
        ylabel("Mouse Activity")
        title("Forecast for Subject " + i)
        legend(["Observed" "Forecast"])
        
        %compare
        figure
        subplot(2,1,1)
        plot(YTest{i})
        hold on
        plot(YPred,'.-')
        hold off
        legend(["Observed" "Forecast"])
        ylabel("Mouse Activity")
        title("Comparision for Subject " + i)

        subplot(2,1,2)
        stem(YPred - YTest{i})
        xlabel("20 Second Intervals")
        ylabel("Error")
        title("RMSE = " + rmse(i))
    end
    
end

After_training = sum(rmse)

   



