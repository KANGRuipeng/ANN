function Omega = ANN_Train(Data_X,Data_Y,Iterator_Max,Hidden_Nodes)

[Data_Num,Feature_Num] = size(Data_X);

if Data_Num ~= length(Data_Y)
    error('inconsistent sample size');
end

%Here we set the number of hidden layer nodes ae one-third of data number%
Hidden_Nodes_Num = Hidden_Nodes;
%Here is the define of first layer parameter%
Omega_First_Layer = randn(Hidden_Nodes_Num,Feature_Num + 1)/100;
%The order of parameters are first layer and then second layer%
Omega_Second_Layer = randn(1,Hidden_Nodes_Num + 1)/100;


%Define It counter%
Iterator_Num = 0;

while Iterator_Num < Iterator_Max
    
    %Set zero to start new calculate%
    Delta_Omega_First_Layer = zeros(Hidden_Nodes_Num , Feature_Num + 1);
    Delta_Omega_Second_Layer = zeros(1 , Hidden_Nodes_Num + 1);
    
    %Back proporagation%
    for i = 1:Data_Num
        
        A = Omega_First_Layer * [1,Data_X(i,:)]';         %Hidden_Nodes_Num * 1%
        Z = 1./(ones(Hidden_Nodes_Num,1) + exp(-A));        %Hidden_Nodes_Num * 1%
        B = Omega_Second_Layer * [1,Z']';                   %1 * 1%
        Y = 1/(1 + exp(-B));                                %1 * 1%
        
        Delta_Omega_Second_Layer = Delta_Omega_Second_Layer + (Y -Data_Y(i))*Y*(1-Y)*[1,Z']; 

        Delta_Omega_First_Layer = Delta_Omega_First_Layer  ...
            + (Y*(1-Y)* (Y -Data_Y(i))).* ...                %1*1%
            ((Z.*(ones(Hidden_Nodes_Num,1)-Z)).*(Omega_Second_Layer(1,2:Hidden_Nodes_Num+1))')*[1,Data_X(i,:)];
    
    end
    
    Zeta = 1;
    Error_Mean =  ANN_Train_Error(Data_Num,Hidden_Nodes_Num,Data_X,Data_Y,Omega_First_Layer,Omega_Second_Layer);
    while Zeta > 1e-10    
        
        Omega_First_Layer_B = Omega_First_Layer - Zeta*Delta_Omega_First_Layer;
        Omega_Second_Layer_B = Omega_Second_Layer - Zeta*Delta_Omega_Second_Layer;
        
        Error_New_Mean =  ANN_Train_Error(Data_Num,Hidden_Nodes_Num,Data_X,Data_Y,Omega_First_Layer_B,Omega_Second_Layer_B);
        if Error_New_Mean < Error_Mean
            break;
        end
        Zeta = Zeta * 0.1;     
        
    end
    
    Omega_First_Layer = Omega_First_Layer_B;
    Omega_Second_Layer = Omega_Second_Layer_B;
    
    if abs(Error_New_Mean - Error_Mean)< 1e-10
        break;
    end
    
    Error_Mean = Error_New_Mean;
    
    
    if Zeta <= 1e-10
        disp('cannot descend anymore');
        break;
    end
    
    Iterator_Num = Iterator_Num + 1;
    
end

Omega.FirstLayer = Omega_First_Layer;
Omega.SecondLayer = Omega_Second_Layer;

end