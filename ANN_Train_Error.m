function Error_Mean = ANN_Train_Error(Data_Num,Hidden_Nodes_Num,Data_X,Data_Y,Omega_First_Layer,Omega_Second_Layer)

%total cost
Error_Sum = 0;

for i=1 : Data_Num 
    
    A = Omega_First_Layer * [1,Data_X(i,:)]';           %Hidden_Nodes_Num * 1%
    Z = 1./(ones(Hidden_Nodes_Num,1) + exp(-A));        %Hidden_Nodes_Num * 1%
    B = Omega_Second_Layer * [1,Z']';                   %1 * 1%
    Y = 1/(1 + exp(-B));                                %1 * 1%
    
    Error_Sum = Error_Sum + (Y - Data_Y(i)) ^2;
end

Error_Mean = Error_Sum / Data_Num ;

end