clear
clc
%Change this label to choose which data set to use%
Used_Data = 5;
%Data used for train%
if Used_Data == 1
    Data =load('datasets/ionosphere_train.mat');
elseif Used_Data == 2
    Data =load('datasets/isolet_train.mat');
elseif Used_Data == 3
    Data =load('datasets/liver_train.mat');
elseif Used_Data == 4
    Data =load('datasets/mnist_train.mat');
else
    Data =load('datasets/mushroom_train.mat');
end

[Data_Num,Feature_Num] = size(Data.X);

Num_Train_80 = fix(Data_Num*0.8);
t1 = clock;
Iterator_Max = 300;

Data_X_Train_80 = zeros(Num_Train_80,Feature_Num);
Data_Y_Train_80 = zeros(Num_Train_80,1);
Data_X_Cross_20 = zeros((Data_Num - Num_Train_80),Feature_Num);
Data_Y_Cross_20 = zeros((Data_Num - Num_Train_80),1);

Correct_Rate = zeros(20,1);

for k = 1:10;
    k
    A = randperm(Data_Num);
    for i = 1: Num_Train_80
        Data_X_Train_80(i,:) = Data.X(A(i),:);
        Data_Y_Train_80(i) = Data.Y(A(i));
    end
    for i = (Num_Train_80+1):Data_Num
        Data_X_Cross_20(i-Num_Train_80,:) = Data.X(A(i),:);
        Data_Y_Cross_20(i-Num_Train_80) = Data.Y(A(i));
    end
    
    for Hidden_Node = 1:20
        Hidden_Node
        Omega = ANN_Train(Data_X_Train_80,Data_Y_Train_80,Iterator_Max,Hidden_Node);
        Test_Result = ANN_Test(Omega,Data_X_Cross_20,Data_Y_Cross_20);
        Correct_Rate(Hidden_Node) = Correct_Rate(Hidden_Node) + Test_Result.Correct_Rate/10;
    end
    
end

[Hidden_Node] = find(Correct_Rate==max(Correct_Rate));
Omega = ANN_Train(Data_X_Train_80,Data_Y_Train_80,Iterator_Max,Hidden_Node(1));
t2 = clock;
t = etime(t2,t1);

%Data used for test%
if Used_Data == 1
    Test =load('datasets/ionosphere_test.mat');
elseif Used_Data == 2
    Test =load('datasets/isolet_test.mat');
elseif Used_Data == 3
    Test =load('datasets/liver_test.mat');
elseif Used_Data == 4
    Test =load('datasets/mnist_test.mat');
else
    Test =load('datasets/mushroom_test.mat');
end


Test_Result = ANN_Test(Omega,Test.X,Test.Y);

