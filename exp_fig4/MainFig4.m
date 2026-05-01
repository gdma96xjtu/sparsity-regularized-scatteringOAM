% --- process complex number --- %
clc;clear all;close all;
%%  rearrange in terms of the sum number of "1"
tobit=8; % total bite
coestate=dec2bin(0:1:2^tobit-1);% dec2bin convert decimal to binary
con=zeros(2^tobit,tobit); % coefficient
for i = 1 : size(con,1)
    for j=1:size(con,2)
        con(i,j)=str2num(coestate(i,j));
    end
end

% --- find equal to [1:8]
for i=1:2^tobit
    temp=con(i,:);
    index(i,:)=sum(temp(find(temp==1))); % the sum number of temp equal to 1
end
clear temp

% --- find row index of con equal to [1:8]
k=1;
for i=1:max(index)
    temp=find(index==i);  % fine position of index == i
    rownum(k:k+length(temp)-1,:) = temp;  % the index
    numdot(i) = length (temp); % the number of temp 
    k=k+length(temp);
end

% --- rearrange con by row index (i.e. by the number of con equal to 1)
rearrcon=zeros(size(con));
for i = 1:2^tobit-1 % the end of recon is zeros
    rearrcon(i,:) = con(rownum(i),:); % two index for con; inside is index for rownum
end
%%  load recon data
load('coe.mat')
load('xrecondenoise.mat') % recon
load('fidelityt_denoise.mat')
load('ampsprdenoise.mat')
load('phasprdenoise.mat')
s1=size(xrecondenoise,2);
%% estimate
% --- ssim
realsim=zeros(1,s1);
imagsim=zeros(1,s1);
% --- psnr
realpsnr=zeros(1,s1);
imagpsnr=zeros(1,s1);
for i = 1: s1
    temp1=coe(:,i)/norm(coe(:,i));
    temp2=xrecondenoise(:,i)/norm(xrecondenoise(:,i));
    % --- ssim
    realsim(i) = ssim(real(temp1),real(temp2));% real similarity
    imagsim(i) = ssim(imag(temp1),imag(temp2));% imag similarity
    
    % --- psnr
    realpsnr(i)= psnr(real(temp1),real(temp2));% real psnr
    imagpsnr(i)= psnr(imag(temp1),imag(temp2));% imag psnr
end

%%  calculate crosstalk of l=[-4:3] (ref: Lei Gong Light supplymentary material)
% --- find l = [-4 :3 ]
p0=0;l0=-4:1:3;
lbegin=61; % lbegin is position of l=-4 and p=0
t0=7; % t0 is [number of (super state)]-1
originalstate=zeros(length(p0)*length(l0),s1); 
reconstate=zeros(length(p0)*length(l0),s1);
for i = 1:s1
    % --- original state
    originalstate(1:1+t0,i)=coe(lbegin:1:lbegin+t0,i); %
    
    % --- recon state
    reconstate(1:1+t0,i)=xrecondenoise(lbegin:1:lbegin+t0,i);
    
end
% --- choose the number of 1 is equal to 1 (also means the single state)
for i = 1 : length(p0)*length(l0)
    % --- coefficient
    coesingle(:,i)=originalstate(:,rownum(i)); % rownum is the index
    reconsingle(:,i)=reconstate(:,rownum(i));
    
    % --- fidelity 
    fidelity(i) = fidelity_denoise(rownum(i));
    
    % --- ssim
    realssim(i)=realsim(rownum(i));
    imagssim(i)=imagsim(rownum(i));
    
    % --- psnr
    realpsnrr(i)=realpsnr(rownum(i));
    imagpsnrr(i)=imagpsnr(rownum(i));
    
end
% calculate (equ. 9) of ref 
I=zeros(length(p0)*length(l0),length(p0)*length(l0),2); % I(:,:,1) for theory value ;I(:,:,2) for exp value
for i = 1 : length(p0)*length(l0)
    
    for j = 1 : length(p0)*length(l0)
        I(i,j,1) = abs(coesingle(i,j)).^2/sum(abs(coesingle(i,:)).^2); % theory
        I(i,j,2) = abs(reconsingle(i,j)).^2/sum(abs(reconsingle(i,:)).^2); % exp
    end
end
% --- crosstalk 
crst=roundn(10*log10(I),-4);
the=-10;
cros=crst;
cros(crst>the)=[];
maxcros=max(cros);

% xlswrite('maxcrosstalk.xlsx',maxcros)
% xlswrite('orthognality.xlsx',I(:,:,2))
% xlswrite('crosstalk.xlsx',crst(:,:,2))
%% consider super state p=0,l=-4:4;....-20:20
len=4:1:20;
p=0;l=-64;
p0=0;
for i = 1 :length(len)
    l0=-len(i):1:len(i);
    lbegin=-(l-l0(1))+1; % lbegin is position of l0(1) and p=0
    t0=length(l0)-1; % t0 is [number of (super state)]-1
    
    % --- fidelity 
    num(i)=length(l0); % the number of superposition state 
    superfi(i) = fidelity_denoise (i+2*2^tobit);
    
    ampdenoise=zeros(length(p0)*length(l0),2); % amp(:,1) is sim amp(:,2) is exp
    phadenoise=zeros(length(p0)*length(l0),2);
    % --- real
    ampdenoise(1:1+t0,1)=real(coe(lbegin:1:lbegin+t0,i+2*2^tobit)); % +2*2^tobit means 513 ...
    ampdenoise(1:1+t0,2)=real(xrecondenoise(lbegin:1:lbegin+t0,i+2*2^tobit));
    % --- imag
    phadenoise(1:1+t0,1)=imag(coe(lbegin:1:lbegin+t0,i+2*2^tobit));
    phadenoise(1:1+t0,2)=imag(xrecondenoise(lbegin:1:lbegin+t0,i+2*2^tobit));
    
    % ------ sprectrum
    ampsprdenoise=zeros(length(p0)*length(l0),2); % ampspr(:,1) is sim amp(:,2) is exp
    phasprdenoise=zeros(length(p0)*length(l0),2);
    for j = 1:2
        ampsprdenoise(:,j)= abs((ampdenoise(:,j))).^2./sum(abs(ampdenoise(:,j)).^2); % real sprectrum
        phasprdenoise(:,j)= abs((phadenoise(:,j))).^2./sum(abs(phadenoise(:,j)).^2); % imag sprectrum
    end
    
    % --- padarray
    temp1 = padarray(ampsprdenoise,[(2*len(end)+1-length(l0))/2 0],'both'); %
    temp2 = padarray(phasprdenoise,[(2*len(end)+1-length(l0))/2 0],'both');
    
    % --- introduce cell
    realspr{i}=temp1;
    imagspr{i}=temp2;
end
%%  after discussion, we need add the superposition state [-1:1],... [-3:3]
sup1='00011100';
sup2='00111110';
sup3='01111111';
dec(1)=bin2dec(sup1);
dec(2)=bin2dec(sup2);
dec(3)=bin2dec(sup3);
supfi=fidelity_denoise(dec); % the fidelity of [-1:1],[-3:3]
sup=[supfi,superfi];
%% figure
close all 
fontsize=10;
marksize=4.5;
a=0.1:0.1:1;
% for i = 1:length(a)
figure ; %  plot fidelity 
% plot(num,superfi,'ro','linewidth',1,'MarkerSize',marksize)
bar(sup,'g');

% b = bar(superfi,'FaceColor','flat','EdgeColor','none');


alpha(a(4))
% xlim([5 45])
set(gca,'XTickLabel',[],'YTickLabel',[]);
ylim([0.8 1])
set(gca,'FontName','Times New Roman','FontSize',fontsize) % 设置坐标轴刻度字体名称，大小
axis normal;
set(gca, 'LooseInset', [0,0,0,0]);
% set(gca,'Position',[0.1 0.1 0.8 0.4])
set (gcf,'position',[ 101         320        1678         426])
box off
% end
% saveas(gca,strcat('superfidelity','.png'));

% --- 
seq=[4 10 16 20]; % sequence of superposition state
clear temp temp1 temp2 temp3
for i = 1 : length(seq)
    figure; % multi -4:4; ...... -20:20
    temp=cell2mat(realspr(seq(i)-3));
    % --- get rid off 0 (result from paddary)
    temp1=temp(:,1);
    temp1(find(temp1==0))=[];
    
    temp2=temp(:,2);
    temp2(find(temp2==0))=[];
    
    temp3(:,1)=temp1;
    temp3(:,2)=temp2;
    
    bar(temp3);
%     alpha(a(6))
    
%     lim=-seq(i):3:seq(i);
    
    set(gca,'FontName','Times New Roman','FontSize',fontsize) % 设置坐标轴刻度字体名称，大小
    axis normal;
    set(gca, 'LooseInset', [0,0,0,0]);
    box off
    % saveas(gca,strcat('superfig',num2str(seq(i)),'.png'));
    set(gca,'XTickLabel',[],'YTickLabel',[]);
    % saveas(gca,strcat('super',num2str(seq(i)),'.png'));
    clear temp temp1 temp2 temp3
end