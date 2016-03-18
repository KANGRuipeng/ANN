function Test_Result = ANN_Test(Omega,Test_X,Test_Y)

Correct_Num = 0;
N = length(Test_Y);
[Hidden_Nodes_Num,~] = size(Omega.FirstLayer);

Omega_First_Layer = Omega.FirstLayer;
Omega_Second_Layer = Omega.SecondLayer;

for i=1:N

    A = Omega_First_Layer * [1,Test_X(i,:)]';           %Hidden_Nodes_Num * 1%
    Z = 1./(ones(Hidden_Nodes_Num,1) + exp(-A));        %Hidden_Nodes_Num * 1%
    B = Omega_Second_Layer * [1,Z']';                   %1 * 1%
    Y = 1/(1 + exp(-B));                                 %1 * 1%
    
    Correct_Num = Correct_Num + (Y > 0.5 == Test_Y(i));
end

Correct_Rate = Correct_Num / N;

Test_Result.Correct_Num = Correct_Num;
Test_Result.Correct_Rate = Correct_Rate;
end