% Simulation study: ideal first-order RC model, udds
clear all
clc;
%% Data Settings
load('Current_Profiles.mat')
%% Generate OCV curves
T=readtable('OCV.csv');
OCVtable=table2array(T(:,2:end));
SOHref=table2array(T(:,1));
T=readtable('R1.csv');
R1table=table2array(T(:,2:end));
T=readtable('R2.csv');
R2table=table2array(T(:,2:end));
T=readtable('C.csv');
Ctable=table2array(T(:,2:end));
%% plot the curves
figure('Position', [100, 100, 800, 600]);
SOC=0:0.01:1;
SOH=0.8:0.05:1;
t = tiledlayout(2, 2, 'TileSpacing', 'compact');
nexttile
hold on
for i=1:length(SOH)
    OCV=update_para(SOC,SOH(i),SOHref,OCVtable);
    plot(SOC*100,OCV,'DisplayName',append('SOH = ',num2str(SOH(i)*100),'%'))
end
xlabel('SOC (%)',FontSize=12)
ylabel('OCV (V)',FontSize=12)
legend('location','northwest','FontSize',11)
set(gca,'Fontsize',12)
grid on
ylim([2.9 5])
title("(a)", 'FontWeight', 'normal')
nexttile
hold on
for i=1:length(SOH)
    R1=update_para(SOC,SOH(i),SOHref,R1table);
    plot(SOC*100,R1,'DisplayName',append('SOH = ',num2str(SOH(i)*100),'%'))
end
xlabel('SOC (%)',FontSize=12)
ylabel('R1 (ohm)',FontSize=12)
%legend('location','southeast','FontSize',12)
set(gca,'Fontsize',12)
grid on
title("(b)", 'FontWeight', 'normal')
nexttile
hold on
for i=1:length(SOH)
    R2=update_para(SOC,SOH(i),SOHref,R2table);
    plot(SOC*100,max(R2, 0.01),'DisplayName',append('SOH = ',num2str(SOH(i)*100),'%'))
end
xlabel('SOC (%)',FontSize=12)
ylabel('R2 (ohm)',FontSize=12)
%legend('location','southeast','FontSize',12)
set(gca,'Fontsize',12)
grid on
title("(c)", 'FontWeight', 'normal')
nexttile
hold on
for i=1:length(SOH)
    C=update_para(SOC,SOH(i),SOHref,Ctable);
    plot(SOC*100,max(C,1000),'DisplayName',append('SOH = ',num2str(SOH(i)*100),'%'))
end
xlabel('SOC (%)',FontSize=12)
ylabel('C (F)',FontSize=12)
%legend('location','southeast','FontSize',12)
set(gca,'Fontsize',12)
grid on
title("(d)", 'FontWeight', 'normal')
exportgraphics(t, 'data_generation.png', 'Resolution', 900);
%% DATA Generation
Qmax=2.2;
SOH_true=1;
sigma_V=1e-3;
rowindex=0;
columnnum=300;
SOCdata=zeros(20000,columnnum);
Idata=zeros(20000,columnnum);
Udata=zeros(20000,columnnum);
%dicharge
for SOC0=0.2:0.01:0.95
    for k=1:3
        for i=1:8
            I=-Qmax*Current_Profiles{1,i}.'/2*2^k;
            rownum=ceil(length(I)/columnnum);
            I=[I,I];
            I=I(1:rownum*columnnum);
            I=reshape(I,columnnum,rownum).';
            for j=1:rownum
                rowindex=rowindex+1;
                Itemp=I(j,:);
                [Utemp,SOC_true] = generate_data(SOH_true, SOC0, sigma_V, Qmax, Itemp, SOHref, OCVtable, R1table, R2table, Ctable);
                Idata(rowindex,:)=Itemp;
                Udata(rowindex,:)=Utemp;
                SOCdata(rowindex,:)=SOC_true;
            end
        end
    end
end
%charging
for SOC0=0:0.01:0.6
    for k=1:3
        for i=10:10:100
            I=zeros(1,columnnum);
            pulsenum=ceil(columnnum/100);
            for j=1:pulsenum
                I(j*100-i+1:j*100)=ones(1,i);
            end
            Itemp=Qmax*I/2*2^k;
            rowindex=rowindex+1;
            [Utemp,SOC_true] = generate_data(SOH_true, SOC0, sigma_V, Qmax, Itemp, SOHref, OCVtable, R1table, R2table, Ctable);
            Idata(rowindex,:)=Itemp;
            Udata(rowindex,:)=Utemp;
            SOCdata(rowindex,:)=SOC_true;
        end
    end
end

Idata=Idata(1:rowindex,:);
Udata=Udata(1:rowindex,:);
SOCdata=SOCdata(1:rowindex,:);
%Generate dataset
writematrix(Idata, 'I.csv');
writematrix(Udata, 'U.csv');
writematrix(SOCdata, 'SOC.csv');


%% functions
function [U,SOC_true] = generate_data(SOH_true, SOC0, sigma_V, Qmax, I, SOHref, OCVtable, R1table, R2table, Ctable)
% EKF "true" data generation: UDDS
rng('default');
rng(100)
%setting some parameters

Cmin=1000;
R2min=0.01;

%Values that are tracked
dt=1;
U=zeros(1,length(I));
SOC_true=zeros(1,length(I)+1);
% initialize state

SOC_true(1)=SOC0;
U_c=0;
%start simulation
for i=1:length(I)
    OCV=update_para(SOC_true(i),SOH_true,SOHref,OCVtable);
    R1=update_para(SOC_true(i),SOH_true,SOHref,R1table);
    R2=update_para(SOC_true(i),SOH_true,SOHref,R2table);
    C=update_para(SOC_true(i),SOH_true,SOHref,Ctable);
    if(R2<R2min)
        R2=R2min;
    end
    if(C<Cmin)
        C=Cmin;
    end
    R2C_reciporal=1/R2/C;

    SOC_true(i+1)=SOC_true(i)+I(i)*dt/3600/(SOH_true*Qmax);
    U_c=U_c*exp(-dt*R2C_reciporal)+(R2-R2*exp(-dt*R2C_reciporal))*I(i);
    U(i)=OCV+U_c+I(i)*R1+sigma_V*randn;
end
SOC_true=SOC_true(2:end);
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


