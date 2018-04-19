clear all;
clc;

%% Parameterek
Nrho=128;
Nphi=128;
Nbprj=128;
example=4;
switch example
    case 1 % agyi CT
        brain=double(imread('brain2.png'))/255;
        x=(1:size(brain,2))-size(brain,2)/2;
        y=(1:size(brain,1))-size(brain,1)/2;
        dt=1;
        m=@(x,y) reshape(diag(brain(round(y)+size(brain,1)/2,round(x)+size(brain,2)/2)),size(x));
    case 2 % hullam
        x=-4:0.01:4;
        y=-4:0.01:4;
        dt=0.005;
        m=@(x,y) sin(2*pi*x)+cos(2*pi*y);
    case 3 % korlap
        x=-1:0.01:1;
        y=-1:0.01:1;
        dt=0.005;
        m=@(x,y) (x+0.4).^2+y.^2<=0.25^2;
    case 4 % X
        x=-3:0.01:3;
        y=-3:0.01:3;
        dt=0.005;
        m=@(x,y) abs(x-y)<0.1 | abs(x+y)<0.1;
    otherwise
        return;
end

%% Radon-transzformalt eloallitasa a definicio szerint, manualisan
rhomax=max([sqrt(x(1)^2+y(1)^2) sqrt(x(1)^2+y(end)^2) sqrt(x(end)^2+y(1)^2) sqrt(x(end)^2+y(end)^2)]);
rho=linspace(-rhomax,rhomax,Nrho);
phi=linspace(0,pi,Nphi);
t=min(-x(end)-rhomax,y(1)-rhomax):dt:max(-x(1)+rhomax,y(end)+rhomax);
Rm=zeros(Nphi,Nrho);
for ir=1:Nrho
    for ip=1:Nphi
        if phi(ip)>1/4*pi && phi(ip)<3/4*pi % ekkor a szinusz a nagyobb abszolut erteku (pl. vizszintes vonal)
            xt=-t*sin(phi(ip));
            yt=rho(ir)/sin(phi(ip))+t*cos(phi(ip));
        else % ekkor meg a koszinusz (pl. fuggoleges vonal)
            xt=rho(ir)/cos(phi(ip))-t*sin(phi(ip));
            yt=t*cos(phi(ip));
        end
        idx=(xt>=x(1) & xt<=x(end) & yt>=y(1) & yt<=y(end)); % csak az ertelmezesi tartomanyon integralunk
        if numel(find(idx))>1; Rm(ip,ir)=trapz(t(idx),m(xt(idx),yt(idx))); end % integral az (xt,yt) vonal menten
    end
end

%% Rekonstrukcio 1 - backprojection
[x_,y_]=meshgrid((1:Nbprj)-Nbprj/2,(1:Nbprj)-Nbprj/2);
m_bprj=zeros(Nbprj,Nbprj);
for ip=1:Nphi
    b=interp1((1:Nrho)-Nrho/2, Rm(ip,:), x_*cos(phi(ip))+y_*sin(phi(ip))); % a phi(ip) szoghoz tartozo laminaris kep
    %figure(1234); imagesc(b); pause(0.01);
    m_bprj=m_bprj+b*(phi(2)-phi(1));
end

%% Rekonstrukcio 2 - filtered backprojection
[x_,y_]=meshgrid((1:Nbprj)-Nbprj/2,(1:Nbprj)-Nbprj/2);
H=abs(linspace(-1,1,Nrho)); % a szuro atvitele
m_bprj_filt=zeros(Nbprj,Nbprj);
for ip=1:Nphi
    b=interp1((1:Nrho)-Nrho/2, real(ifft(ifftshift(fftshift(fft(Rm(ip,:))).*H))), x_*cos(phi(ip))+y_*sin(phi(ip))); % a phi(ip) szoghoz tartozo laminaris kep
    %figure(1234); imagesc(b); pause(0.01);
    m_bprj_filt=m_bprj_filt+b*(phi(2)-phi(1));
end

%% Abrazolas
[x_,y_]=meshgrid(x,y);
m_=m(x_,y_);
%figure(123); imagesc(radon(m_)'); set(gca,'YDir','normal');
figure(1);
subplot(221);
imagesc(x,y,m_);
colormap(gray);
set(gca,'YDir','normal');
xlabel('x');
ylabel('y');
title('Eredeti kép');
subplot(222);
imagesc(rho,phi*180/pi,Rm);
colormap(gray);
set(gca,'YDir','normal');
xlabel('\rho');
ylabel('\phi [\circ]');
title('Radon-transzformált');
subplot(223);
imagesc(m_bprj);
colormap(gray);
set(gca,'YDir','normal','XTickLabel',[],'YTickLabel',[]);
title('Egyszerû visszavetítés');
subplot(224);
imagesc(m_bprj_filt);
colormap(gray);
set(gca,'YDir','normal','XTickLabel',[],'YTickLabel',[]);
title('Szûrt visszavetítés');