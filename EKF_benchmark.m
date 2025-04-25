% Simulation study: ideal first-order RC model, udds
% If you want to disable improvement 1,2,3, then change their value to 0.
clear
clc
%% Data Settings
Idata = readtable('DATA\100% SOH\I.csv');
Udata = readtable('DATA\100% SOH\U.csv');
SOCdata = readtable('DATA\100% SOH\SOC.csv');
repeat=length(Idata{:, 1});
I = Idata{1, :};  % Extract the i-th row as a table row
U = Udata{1,:};
SOC_true = SOCdata{1,:};
t=0:1:length(I)-1;
%% plot the curve
% f1=figure();
% set(f1, 'Position', [100, 100, 350, 300]);
% hold on
% %plot(t,100*(SOC_true),'r','linewidth',1.5,'DisplayName','Actual SOC')
% plot(t,U,'r','linewidth',1.5,'DisplayName','Actual SOC')
% legend()
% grid on
% xlabel('Time (s)')
% ylabel('SOC (%)')
% %ylim([0 1])

%% Setting some parameters
Qmax=2.2;
SOH_true=1;
R1=0.07;
R2=0.04;
R2C_reciporal=0.01;
dt=1;

sigma_I=1e-3;
sigma_OCV=1e-2;
sigma_V=1e-3;

T=readtable('OCV.csv');
OCVref=table2array(T(:,2:end));
SOHref=table2array(T(:,1));
SOCtable=0:0.01:1;
OCVtable=update_para(SOCtable,1,SOHref,OCVref);

% Values that are tracked
SOH_RMS=0;
SOC_RMS=0;
for iii=1:repeat
    if(mod(iii,100)==0)
        disp(iii)
    end
    % initialize state
    I = Idata{iii, :};  % Extract the i-th row as a table row
    U = Udata{iii,:};
    SOC_true = SOCdata{iii,:};
    Uc=0;
    U_c=0;
    Irms=rms(I);
    %% EKF algorithm
    % set some parameters
    % Values that are tracked
    SOH_est=zeros(1,length(t));
    SOC_est=zeros(1,length(t));
    %Initialization
    OCV0=U(1)-R1*I(1);
    SOC0=interp1(OCVtable, SOCtable, OCV0, 'linear', 'extrap');
    state0=[SOC0*Qmax*3600 0 Qmax*3600];%initial state: remaining capacity, Uc, maximum capacity
    state=state0.';
    initialCovariance=[(0.02*Qmax*3600)^2 0 0;0 1e-8 0;0 0 (0.1*Qmax*3600)^2];
    Variance = initialCovariance;

    %SOH & SOC estimation
    tic;
    for j=1:length(I)
        %EKF
        %predict
        processnoise=[dt^2*sigma_I^2 0 0; 0 (Irms*R2*0.05)^2 0; 0 0 1e-4];
        measurenoise=(sigma_V+sigma_OCV)^2;
        state=stateModel(state,dt,I(j),R2,R2C_reciporal);
        F=get_F(dt,R2C_reciporal);
        Variance=F*Variance*F.'+processnoise;
        %H matrix
        dOCVdSOH=0;
        dOCVdSOC = get_dOCVdSOC_from_fitting(state(1)/state(3),OCVref(end,:));
        H=[dOCVdSOC/state(3) 0 dOCVdSOH/(Qmax*3600)-dOCVdSOC*state(1)/state(3)/state(3)];
        %kalman gain
        K=Variance*H.'/(H*Variance*H.'+measurenoise);
        %update
        m_exp=get_OCV_from_fitting(state(1)/state(3),OCVref(end,:))+state(2)+R1*I(j);
        state0=state;
        state=state+K*(U(j)-m_exp);
        %H matrix update
        Variance=([1,0,0;0,1,0;0,0,1]-K*H)*Variance;  
        %correction
        if(state(3)<0.8*Qmax*3600)
            ds=0.8*Qmax*3600-state(3);
            Variance(3,3)=Variance(3,3)+ds^2;
            state(3)=0.8*Qmax*3600;
        elseif(state(3)>Qmax*3600)
            ds=state(3)-Qmax*3600;
            Variance(3,3)=Variance(3,3)+ds^2;
            state(3)=Qmax*3600;
        end
        if(state(1)/state(3)>1)
            ds=state(1)-state(3);
            Variance(1,1)=Variance(1,1)+ds^2;
            state(1)=state(3);
        elseif(state(1)<0)
            ds=-state(1);
            Variance(1,1)=Variance(1,1)+ds^2;
            state(1)=0;
        end
        %record the values
        SOC_est(j)=state(1)/state(3);
        SOH_est(j)=state(3)/Qmax/3600;
    end
    SOH_RMS=SOH_RMS+rms(SOH_est-SOH_true)/repeat;
    SOC_RMS=SOC_RMS+rms(SOC_est-SOC_true)/repeat;
end

%% Plotting the results
%SOC estimation result
T = tiledlayout(2, 1, 'TileSpacing', 'compact');
nexttile
hold on
plot(t,SOC_true*100,'DisplayName','True SOC')
plot(t,SOC_est*100,'DisplayName','Estimated SOC')
legend('location','east','FontSize',11)
set(gca,'Fontsize',12)
grid on
xlabel('Time (s)')
ylabel('SOC (%)')

%SOH estimation result
nexttile
hold on
plot(t,SOH_true*ones(1,length(t))*100,'DisplayName','True SOH')
plot(t,SOH_est*100,'DisplayName','Estimated SOH')
legend('location','east','FontSize',11)
set(gca,'Fontsize',12)
grid on
xlabel('Time (s)',FontSize=12)
ylabel('SOH (%)',FontSize=12)
ylim([85 105])
exportgraphics(T, 'baseline.png', 'Resolution', 900);

%% display metrics
disp(append("SOC RMS error =",num2str(100*SOC_RMS),"%"))
disp(append("SOH RMS error =",num2str(100*SOH_RMS),"%"))

%% functions
function stateNext = stateModel(state,dt,I,R2,R2C_reciporal)
    A = [1 0 0; 0 exp(-dt*R2C_reciporal) 0; 0 0 1];
    B = [dt; R2-R2*exp(-dt*R2C_reciporal); 0];
    stateNext = A*state+B*I;
end

function F_matrix=get_F(dt,R2C_reciporal)
    F_matrix=[1 0 0; 0 exp(-dt*R2C_reciporal) 0; 0 0 1];
end

function dOCVdSOC = get_dOCVdSOC_from_fitting(SOC,coeficients)
xspace=zeros(length(coeficients),length(SOC));
for i=1:length(coeficients)-1
    xspace(i,:)=(length(coeficients)-i).*SOC.^(length(coeficients)-i-1);
end
dOCVdSOC=coeficients*xspace;
end

function OCV = get_OCV_from_fitting(SOC,coeficients)
xspace=NaN(length(coeficients),length(SOC));
for i=1:length(coeficients)
    xspace(i,:)=SOC.^(length(coeficients)-i);
end
OCV=coeficients*xspace;
end

function paranew=update_para(SOC,SOH,SOHvec,paraTable)
SOH_clipped=SOH;
if SOH_clipped>1
    SOH_clipped=1;
elseif SOH_clipped<0.8
    SOH_clipped=0.8;
end
i=1;
while (SOHvec(i+1)<SOH_clipped || i+1<length(SOHvec))
    i=i+1;
end
p1=(SOH_clipped-SOHvec(i))/(SOHvec(i+1)-SOHvec(i));
p2=1-p1;
j=length(paraTable(1,:));
SOCvec=SOC.^0;
for ii=1:j-1
    SOCvec=[SOC.^ii; SOCvec];
end
para=p1*paraTable(i,:)+p2*paraTable(i+1,:);
paranew=para*SOCvec;
end