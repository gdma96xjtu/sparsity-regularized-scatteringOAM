% ---- The code verfies the performance of TV-based PR for the measurement
%                              of  LG spectrum                           %
%
% by gdma
clc;clear all;clear

%% sparsity
clear
tic
seed=2;
rng(seed)
fe=zeros(4,100,4);
num=zeros(4,100,4);
input=1024;
xreconfinal=zeros(input,4,4); 
gama=2:5;
SNR=3;
m=32;
for ii=1:4
    coe=zeros(1024,1);
    coe(1:10^(ii-1),1)=1i*(rand(10^(ii-1),1)*2-1)+(rand(10^(ii-1),1)*2-1);
    for gamai=1:length(gama)
        gamai
        n=m*gama(gamai); % output dimension
        TM = raylrnd(sqrt(1/2/n^2),n^2,m^2).*exp(1i * random('unif',-pi,+pi-eps('double'),n^2,m^2)); % transmission matrix generation.
        for SNRi=1:length(SNR)
            TM = awgn(TM,SNR(SNRi),'measured','linear'); % add noise upon preset SNR.
            TM_out=TM;        
            Y=TM*coe; % output field
            Y= awgn(Y,SNR(SNRi),'measured','linear'); % add noise upon preset SNR.
            Y=abs(Y);% output amplitude
            [U,S,V]=svd(TM,'econ'); % economic svd for inverse TM calculation.
            sv = diag(S);
            TMinv = V*(repmat(1./sv,[1,n^2]).*U'); % inverse TM calculation.%% 
            
            a=coe;
            At=TMinv;
            denoise=1;
                
            Xretv0=At*(Y.*exp(2*pi*ones(size(Y))));%initialization
            Xcorr0 = abs(a'*Xretv0/norm(a)/norm(Xretv0))^2;
            
            sia1=size(a,1);
            sia2=size(Y,2);
            for i=1:sia2
                Xcorr0(i) = a(:,i)'*Xretv0(:,i)/norm(a(:,i))/norm(Xretv0(:,i)); % calculate correlation
            end
            xrecon=zeros(sia1,sia2);
            if gamai==1 || gamai==2
                iterMax = 300; % max iteration number to prevent infinite loop.
            else
                iterMax = 100;
            end
    
            iterTor = 10^-7; % convergence criteria.
            Xcorr_iter = zeros(iterMax,sia2); % archives correlation changes during iteration.
            TMinv=At;
                   % disp('iteration start!'); tic;
            for i=1:sia2
                Xiter = Xretv0(:,i); % set retrieved field for the initial guess.
                Xcorr_iter(1,i) = Xcorr0(i);
                for itr = 1:iterMax
                    Yiter = Y(:,i).^2.*exp(1i*angle(TM_out * Xiter)); % transform incident field into corresponding diffused field space.
                    Xiter = TMinv*Yiter; % return to the incident field space.
                    Yiter = Y(:,i).*exp(1i*angle(TM_out * Xiter)); % transform incident field into corresponding diffused field space.
                    Xiter = TMinv*Yiter; % return to the incident field space.
                    
                    % ------ above is phase retrieval below is  TV-regularization
                    if denoise == 1
                        % --- find optimization parameter regParam
                        epara = -10:1:10; % exp index; '1' is precion
                        apara = 0.1:0.1:9.9; % '0.1' is presion
                        for p = 1:length(epara) % consider exp first
                            regParam = 10^(epara(p));
                            re=real(Xiter);
                            im=imag(Xiter);
                            re=prox_tv1d(re, regParam);  % contain FISTA update
                            im=prox_tv1d(im, regParam);
                            tempeXiter=re+1i*im;
                            tempe(p) = abs(a(:,i)'*tempeXiter/norm(tempeXiter)/norm(a(:,i)));
                        end
                        indexe = find (tempe == max(tempe) ); % optimization epara
                        
                        % --- find aparam
                        for q = 1:length(apara) % consider exp first
                            regParam = apara(q)*10^(epara(indexe(1)));
                            re=real(Xiter);
                            im=imag(Xiter);
                            re=prox_tv1d(re, regParam);  % contain FISTA update
                            im=prox_tv1d(im, regParam);
                            tempaXiter=re+1i*im;
                            tempa(q) = abs(a(:,i)'*tempaXiter/norm(tempaXiter)/norm(a(:,i)));
                        end
                        indexa = find (tempa == max(tempa) ); % optimization apara
                        
                        % --- find out
                        regParam = apara(indexa(1))*10^(epara(indexe(1))); %(for single p=0;l=10,SNR=1);    % regularization parameter for  prox_tv  but !!! 0.5e-2 is optimization for single
                        re=real(Xiter);
                        im=imag(Xiter);
                        re=prox_tv1d(re, regParam);  % contain FISTA update
                        im=prox_tv1d(im, regParam);
                        Xiter=re+1i*im;
                        Xcorr_iter(itr+1,i) = a(:,i)'*Xiter/norm(Xiter)/norm(a(:,i)); % correlation calculation.   norm(X)=1
                    end
                    % --- calculate fidelity
                    xrecon = Xiter*exp(-1i*angle(Xcorr_iter(itr,i)));
                    fe(ii,itr,gamai) = abs(coe'*xrecon/norm(coe)/norm(xrecon))^2; % fedelity
                    
                    if abs(abs(Xcorr_iter(itr+1,i)) - abs(Xcorr_iter(itr,i))) < iterTor % iteration ends when the criterion is satisfied.
                        break;
                    end
                end
                num(ii,itr,gamai)=itr;
                xreconfinal(:,itr,gamai)=xrecon;
            end
         end
    end
end
toc
% filename =strcat('numtv1-1000dsnr.mat');
% save(filename,'num')
% filename =strcat('fidelitytv1-1000dsnr.mat');
% save(filename,'fe')
% filename =strcat('xreconfinaltv1-1000dsnr.mat');
% save(filename,'xreconfinal')
 

%  sparsity plot_fig

Linewidth=2;
fontsize=20;
ymin=-0.01;
ymax=1.01;

clear temp1 temp2
for i = 1:4
    figure(Position=[100,100,500,500])
    temp1 =squeeze(fe(:,:,i));
    for j = 1:size(temp1,1)
        temp2=(temp1(j,:));
        temp2(temp2==0)=nan;
        plot(temp2,'o','LineWidth',Linewidth,'LineStyle','-');
        hold on
    end
    set(gca,'FontName','Times New Roman','FontSize',fontsize) % 设置坐标轴刻度字体名称，大小
    title(['\eta=',num2str(gama(i)^2)]);
    axis normal;
    box off
    set(gca, 'LooseInset', [0,0,0,0]);
    
    set(gca,'ylim',[ymin,ymax]);
    if i<3
        set(gca,'xlim',[0,100]);
    else
        set(gca,'xlim',[0,30]);
    end
    legend('n=1','n=10','n=100','n=1000','FontSize',fontsize','location','east','interpreter','latex', 'Box', 'off')
    saveas(gca,strcat('gammasfigcoe',num2str(gama(i)),'.svg'));
    % saveas(gca,strcat('gamma',num2str(SNR(i)),'.png'));
end

%%  GS without regulation
clear
tic
coe=zeros(1024,1);
coe(2,1)=1;
input=1024;
seed=4;
rng(seed)
m=32;% input dimension
gama=2:7;
SNR=[1,2,3,5,10,100];  % linear signal-to-noise ratio (SNR) to mimic practical noisy situations.
fe=zeros(length(gama),length(SNR),100);
num=zeros(length(gama),length(SNR),100);
xreconfinal=zeros(length(gama),length(SNR),input);
for gamai=1:length(gama)
    n=m*gama(gamai); % output dimension
    TM = raylrnd(sqrt(1/2/n^2),n^2,m^2).*exp(1i * random('unif',-pi,+pi-eps('double'),n^2,m^2)); % transmission matrix generation.
    for SNRi=1:length(SNR)
        gama(gamai)
        SNR(SNRi)
        TM = awgn(TM,SNR(SNRi),'measured','linear'); % add noise upon preset SNR.
        TM_out=TM;        
        Y=TM*coe; % output field
        Y= awgn(Y,SNR(SNRi),'measured','linear'); % add noise upon preset SNR.
        Y=abs(Y);% output amplitude
        
        [U,S,V]=svd(TM,'econ'); % economic svd for inverse TM calculation.
        sv = diag(S);
        TMinv = V*(repmat(1./sv,[1,n^2]).*U'); % inverse TM calculation.%% 
        a=coe;
        At=TMinv;
        Xretv0=At*(Y.*exp(2*pi*ones(size(Y))));%initialization
        Xcorr0 = abs(a'*Xretv0/norm(a)/norm(Xretv0))^2;
        sia1=size(a,1);
        sia2=size(Y,2);
        for i=1:sia2
            Xcorr0(i) = a(:,i)'*Xretv0(:,i)/norm(a(:,i))/norm(Xretv0(:,i)); % calculate correlation
        end
        xrecon=zeros(sia1,sia2);
        iterTor = 10^-7; % convergence criteria.
        TMinv=At;
        iterMax = 1000;
        Xcorr_iter = zeros(iterMax,sia2); % archives correlation changes during iteration.
        % disp('iteration start!'); tic;
        for i=1:sia2
            Xiter = Xretv0(:,i); % set retrieved field for the initial guess.
            Xcorr_iter(1,i) = Xcorr0(i);
            for itr = 1:iterMax
                Yiter = Y(:,i).*exp(1i*angle(TM_out * Xiter)); % transform incident field into corresponding diffused field space.
                Xiter = TMinv*Yiter; % return to the incident field space.
                Xcorr_iter(itr+1,i) = a(:,i)'*Xiter/norm(Xiter)/norm(a(:,i)); % correlation calculation.   norm(X)=1
                xrecon = Xiter*exp(-1i*angle(Xcorr_iter(itr,i)));
                fe(gamai,SNRi,itr) = abs(coe'*xrecon/norm(coe)/norm(xrecon))^2; % fedelity
                if abs(abs(Xcorr_iter(itr+1,i)) - abs(Xcorr_iter(itr,i))) < iterTor % iteration ends when the criterion is satisfied.
                    break;
                end
            end
            num(gamai,SNRi,i)=itr;
            xreconfinal(gamai,SNRi,:)=xrecon;
        end
    end
end
toc
% filename =strcat('numGS.mat');
% save(filename,'num')
% filename =strcat('fidelityGS.mat');
% save(filename,'fe')
% filename =strcat('xreconfinalGS.mat');
% save(filename,'xreconfinal')


%  GS without regulation plot_fig

Linewidth=2;
fontsize=20;
ymin=-0.01;
ymax=1.01;

clear temp1 temp2
for i = 1:length(SNR)
    figure(Position=[100,100,500,500])
    temp1 =squeeze(fe(:,i,:));
    ax = gca;
    defaultColors = ax.ColorOrder;
    for j = length(gama):-1:1
        temp2=(temp1(j,:));
        temp2(temp2==0)=[];
        colorIdx = mod(j-1, size(defaultColors,1)) + 1;
        baseColor = defaultColors(colorIdx, :);
        lineColor = [baseColor, 1];
        if i==1&&j==1
            plot(temp2,'o','MarkerSize',4,'LineWidth',Linewidth,'LineStyle','-','color',lineColor);
        else
            plot(temp2,'o','LineWidth',Linewidth,'LineStyle','-','color',lineColor);
        end
        hold on
    end
    if i==length(SNR)
        legend(['$\eta$=',num2str(gama(1).^2)],['$\eta$=',num2str(gama(2).^2)],...
            ['$\eta$=',num2str(gama(3).^2)],['$\eta$=',num2str(gama(4).^2)],...
            ['$\eta$=',num2str(gama(5).^2)],['$\eta$=',num2str(gama(6).^2)],...
            'FontSize',fontsize','location','best','interpreter','latex', 'Box', 'off')
    end
    set(gca,'FontName','Times New Roman','FontSize',fontsize) % 设置坐标轴刻度字体名称，大小
    title(['SNR=',num2str(SNR(i))]);
    axis normal;
    box off
    xmaxGS=300;
    xlim([0,xmaxGS])
    set(gca, 'LooseInset', [0,0,0,0]);
    set(gca,'ylim',[ymin,ymax]);
    % saveas(gca,strcat('gammasfigGS',num2str(SNR(i)),'.svg'));
    % saveas(gca,strcat('gamma',num2str(SNR(i)),'.png'));
end

%%  GS2-1 without regulation
clear
tic
coe=zeros(1024,1);
coe(2,1)=1;
input=1024;
seed=4;
rng(seed)
m=32;% input dimension
gama=2:7;
SNR=[1,2,3,5,10,100];  % linear signal-to-noise ratio (SNR) to mimic practical noisy situations.
fe=zeros(length(gama),length(SNR),100);
num=zeros(length(gama),length(SNR),100);
xreconfinal=zeros(length(gama),length(SNR),input);
for gamai=1:length(gama)
    n=m*gama(gamai); % output dimension
    TM = raylrnd(sqrt(1/2/n^2),n^2,m^2).*exp(1i * random('unif',-pi,+pi-eps('double'),n^2,m^2)); % transmission matrix generation.
    for SNRi=1:length(SNR)
        gama(gamai)
        SNR(SNRi)
        TM = awgn(TM,SNR(SNRi),'measured','linear'); % add noise upon preset SNR.
        TM_out=TM;        
        Y=TM*coe; % output field
        Y= awgn(Y,SNR(SNRi),'measured','linear'); % add noise upon preset SNR.
        Y=abs(Y);% output amplitude
        
        [U,S,V]=svd(TM,'econ'); % economic svd for inverse TM calculation.
        sv = diag(S);
        TMinv = V*(repmat(1./sv,[1,n^2]).*U'); % inverse TM calculation.%% 
        a=coe;
        At=TMinv;
        Xretv0=At*(Y.*exp(2*pi*ones(size(Y))));%initialization
        Xcorr0 = abs(a'*Xretv0/norm(a)/norm(Xretv0))^2;
        sia1=size(a,1);
        sia2=size(Y,2);
        for i=1:sia2
            Xcorr0(i) = a(:,i)'*Xretv0(:,i)/norm(a(:,i))/norm(Xretv0(:,i)); % calculate correlation
        end
        xrecon=zeros(sia1,sia2);
        iterTor = 10^-7; % convergence criteria.
        TMinv=At;
        iterMax = 300;
        Xcorr_iter = zeros(iterMax,sia2); % archives correlation changes during iteration.
        % disp('iteration start!'); tic
        for i=1:sia2
            Xiter = Xretv0(:,i); % set retrieved field for the initial guess.
            Xcorr_iter(1,i) = Xcorr0(i);
            for itr = 1:iterMax
                Yiter = Y(:,i).^2.*exp(1i*angle(TM_out * Xiter)); % transform incident field into corresponding diffused field space.
                Xiter = TMinv*Yiter; % return to the incident field space.
                Yiter = Y(:,i).*exp(1i*angle(TM_out * Xiter)); % transform incident field into corresponding diffused field space.
                Xiter = TMinv*Yiter; % return to the incident field space.
                Xcorr_iter(itr+1,i) = a(:,i)'*Xiter/norm(Xiter)/norm(a(:,i)); % correlation calculation.   norm(X)=1
                xrecon = Xiter*exp(-1i*angle(Xcorr_iter(itr,i)));
                fe(gamai,SNRi,itr) = abs(coe'*xrecon/norm(coe)/norm(xrecon))^2; % fedelity
                
                if abs(abs(Xcorr_iter(itr+1,i)) - abs(Xcorr_iter(itr,i))) < iterTor % iteration ends when the criterion is satisfied.
                    break;
                end
            end
            num(gamai,SNRi,i)=itr;
            xreconfinal(gamai,SNRi,:)=xrecon;
        end
    end
end
toc

% filename =strcat('numGS21.mat');
% save(filename,'num')
% filename =strcat('fidelityGS21.mat');
% save(filename,'fe')
% filename =strcat('xreconfinalGS21.mat');
% save(filename,'xreconfinal')

%  GS2-1 without regulation plot_fig
Linewidth=2;
fontsize=20;
ymin=-0.01;
ymax=1.01; 

clear temp1 temp2
for i = 1:length(SNR)
    figure(Position=[100,100,500,500])
    temp1 =squeeze(fe(:,i,:));
    ax = gca;
    defaultColors = ax.ColorOrder;
    for j = length(gama):-1:1
        temp2=(temp1(j,:));
        temp2(temp2==0)=[];
        colorIdx = mod(j-1, size(defaultColors,1)) + 1;
        baseColor = defaultColors(colorIdx, :);
        lineColor = [baseColor, 1];
        if i==1&&j==1
            plot(temp2,'o','MarkerSize',4,'LineWidth',Linewidth,'LineStyle','-','color',lineColor);
        else
            plot(temp2,'o','LineWidth',Linewidth,'LineStyle','-','color',lineColor);
        end
        hold on
    end
    if i==length(SNR)
        legend(['$\eta$=',num2str(gama(1).^2)],['$\eta$=',num2str(gama(2).^2)],...
            ['$\eta$=',num2str(gama(3).^2)],['$\eta$=',num2str(gama(4).^2)],...
            ['$\eta$=',num2str(gama(5).^2)],['$\eta$=',num2str(gama(6).^2)],...
            'FontSize',fontsize','location','best','interpreter','latex', 'Box', 'off')
    end
    set(gca,'FontName','Times New Roman','FontSize',fontsize) % 设置坐标轴刻度字体名称，大小
    title(['SNR=',num2str(SNR(i))]);
    axis normal;
    box off
    % xmaxtv=[100 80 30 15 15 15];
    xmaxGS21=150;
    xlim([0,xmaxGS21])
    set(gca, 'LooseInset', [0,0,0,0]);
    set(gca,'ylim',[ymin,ymax]);
    % saveas(gca,strcat('gammasfigGS',num2str(SNR(i)),'.svg'));
    % saveas(gca,strcat('gamma',num2str(SNR(i)),'.png'));
end


%% GS2-1 with regulation
clear
tic
coe=zeros(1024,1);
coe(2,1)=1;
input=1024;
seed=4;
rng(seed)
m=32;% input dimension
gama=2:7;
SNR=[1,2,3,5,10,100];  % linear signal-to-noise ratio (SNR) to mimic practical noisy situations.
fe=zeros(length(gama),length(SNR),100);
num=zeros(length(gama),length(SNR),100);
xreconfinal=zeros(length(gama),length(SNR),input);
for gamai=1:length(gama)
    n=m*gama(gamai); % output dimension
    TM = raylrnd(sqrt(1/2/n^2),n^2,m^2).*exp(1i * random('unif',-pi,+pi-eps('double'),n^2,m^2)); % transmission matrix generation.
    for SNRi=1:length(SNR)
        gama(gamai)
        SNR(SNRi)
        TM = awgn(TM,SNR(SNRi),'measured','linear'); % add noise upon preset SNR.
        TM_out=TM;        
        Y=TM*coe; % output field
        Y= awgn(Y,SNR(SNRi),'measured','linear'); % add noise upon preset SNR.
        Y=abs(Y);% output amplitude
        
        [U,S,V]=svd(TM,'econ'); % economic svd for inverse TM calculation.
        sv = diag(S);
        TMinv = V*(repmat(1./sv,[1,n^2]).*U'); % inverse TM calculation.%% 
        a=coe;
        At=TMinv;
        Xretv0=At*(Y.*exp(2*pi*ones(size(Y))));%initialization
        Xcorr0 = abs(a'*Xretv0/norm(a)/norm(Xretv0))^2;
        sia1=size(a,1);
        sia2=size(Y,2);
        for i=1:sia2
            Xcorr0(i) = a(:,i)'*Xretv0(:,i)/norm(a(:,i))/norm(Xretv0(:,i)); % calculate correlation
        end
        xrecon=zeros(sia1,sia2);
        iterTor = 10^-7; % convergence criteria.
        iterMax = 150;
        Xcorr_iter = zeros(iterMax,sia2); % archives correlation changes during iteration.
        % disp('iteration start!'); tic;
        for i=1:sia2
            Xiter = Xretv0(:,i); % set retrieved field for the initial guess.
            Xcorr_iter(1,i) = Xcorr0(i);
            for itr = 1:iterMax
                Yiter = Y(:,i).^2.*exp(1i*angle(TM_out * Xiter)); % transform incident field into corresponding diffused field space.
                Xiter = TMinv*Yiter; % return to the incident field space.
                Yiter = Y(:,i).*exp(1i*angle(TM_out * Xiter)); % transform incident field into corresponding diffused field space.
                Xiter = TMinv*Yiter; % return to the incident field space.
                % ------ above is phase retrieval below is  TV-regularization
                    % --- find optimization parameter regParam
                epara = -10:1:10; % exp index; '1' is precion
                apara = 0.1:0.1:9.9; % '0.1' is presion
                for p = 1:length(epara) % consider exp first
                    regParam = 10^(epara(p));
                    re=real(Xiter);
                    im=imag(Xiter);
                    re=prox_tv1d(re, regParam);  % contain FISTA update
                    im=prox_tv1d(im, regParam);
                    tempeXiter=re+1i*im;
                    tempe(p) = abs(a(:,i)'*tempeXiter/norm(tempeXiter)/norm(a(:,i)));
                end
                indexe = find (tempe == max(tempe) ); % optimization epara
                % --- find aparam
                for q = 1:length(apara) % consider exp first
                    regParam = apara(q)*10^(epara(indexe(1)));
                    re=real(Xiter);
                    im=imag(Xiter);
                    re=prox_tv1d(re, regParam);  % contain FISTA update
                    im=prox_tv1d(im, regParam);
                    tempaXiter=re+1i*im;
                    tempa(q) = abs(a(:,i)'*tempaXiter/norm(tempaXiter)/norm(a(:,i)));
                end
                indexa = find (tempa == max(tempa) ); % optimization apara
                
                % --- find out
                regParam = apara(indexa(1))*10^(epara(indexe(1))); %(for single p=0;l=10,SNR=1);    % regularization parameter for  prox_tv  but !!! 0.5e-2 is optimization for single
                re=real(Xiter);
                im=imag(Xiter);
                re=prox_tv1d(re, regParam);  % contain FISTA update
                im=prox_tv1d(im, regParam);
                Xiter=re+1i*im;
                Xcorr_iter(itr+1,i) = a(:,i)'*Xiter/norm(Xiter)/norm(a(:,i)); % correlation calculation.   norm(X)=1
                % --- calculate fidelity
                xrecon = Xiter*exp(-1i*angle(Xcorr_iter(itr,i)));
                fe(gamai,SNRi,itr) = abs(coe'*xrecon/norm(coe)/norm(xrecon))^2; % fedelity
                if abs(abs(Xcorr_iter(itr+1,i)) - abs(Xcorr_iter(itr,i))) < iterTor % iteration ends when the criterion is satisfied.
                    break;
                end
            end
            num(gamai,SNRi,i)=itr;
            xreconfinal(gamai,SNRi,:)=xrecon;
        end
    end
end
toc

% filename =strcat('numtv.mat');
% save(filename,'num')
% filename =strcat('fidelitytv.mat');
% save(filename,'fe')
% filename =strcat('xreconfinaltv.mat');
% save(filename,'xreconfinal')

% GS2-1 with regulation plot_fig
Linewidth=2;
fontsize=20;
ymin=-0.01;
ymax=1.01; 

clear temp1 temp2
for i = 1:length(SNR)
    figure(Position=[100,100,500,500])
    temp1 =squeeze(fe(:,i,:));
    ax = gca;
    defaultColors = ax.ColorOrder;
    for j = length(gama):-1:1
        temp2=(temp1(j,:));
        temp2(temp2==0)=[];
        colorIdx = mod(j-1, size(defaultColors,1)) + 1;
        baseColor = defaultColors(colorIdx, :);
        lineColor = [baseColor, 1];
        if i==1&&j==1
            plot(temp2,'o','MarkerSize',4,'LineWidth',Linewidth,'LineStyle','-','color',lineColor);
        else
            plot(temp2,'o','LineWidth',Linewidth,'LineStyle','-','color',lineColor);
        end
        hold on
    end
    if i==length(SNR)
        legend(['$\eta$=',num2str(gama(1).^2)],['$\eta$=',num2str(gama(2).^2)],...
            ['$\eta$=',num2str(gama(3).^2)],['$\eta$=',num2str(gama(4).^2)],...
            ['$\eta$=',num2str(gama(5).^2)],['$\eta$=',num2str(gama(6).^2)],...
            'FontSize',fontsize','location','best','interpreter','latex', 'Box', 'off')
    end
    set(gca,'FontName','Times New Roman','FontSize',fontsize) % 设置坐标轴刻度字体名称，大小
    title(['SNR=',num2str(SNR(i))]);
    axis normal;
    box off
    xmaxtv=[100 80 30 15 15 15];
    xlim([0,xmaxtv(i)])
    set(gca, 'LooseInset', [0,0,0,0]);
    set(gca,'ylim',[ymin,ymax]);
    % saveas(gca,strcat('gammasfigGS',num2str(SNR(i)),'.svg'));
    % saveas(gca,strcat('gamma',num2str(SNR(i)),'.png'));
end