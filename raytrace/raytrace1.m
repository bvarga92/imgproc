clear all;
clc;

%% parameterek
w=300;                       % kep szelessege
h=300;                       % kep magassaga
C=[0 0 1];                   % gomb kozeppontja
r=0.7;                       % gomb sugara
clrSphere=[0.8 0 0];         % gomb szine
shininess=50;                % csillogas
L=[-6 -6 -15];               % pontfeny pozicioja
clrLight=[1 1 1];            % pontfeny szine
clrAmbient=[0.05 0.05 0.05]; % ambiens feny szine
O=[0 0 -3];                  % kamera pozicioja
Q=[0 0 1];                   % kamera iranya

%% szamitas
img=zeros(h,w,3);
x=linspace(-1,1,w);
y=linspace(-h/w,h/w,h);
for ix=1:length(x)
    for iy=1:length(y)
        % metszespont kiszamitasa: ||O+t*D-C||^2=r^2 egyenlet megoldasa
        D=[x(ix) y(iy) Q(3)]-O; % a sugar iranyvektora
        a=D*D';
        b=2*D*(O-C)';
        c=(O-C)*(O-C)'-r^2;
        d=b^2-4*a*c; % a masodfoku egyenlet diszkriminansa
        if d<=0; continue; end % nincs metszespont --> fekete pixel
        t=[(-b-sqrt(d))/(2*a) (-b+sqrt(d))/(2*a)];
        t=min(t(t>=0));
        if isempty(t); continue; end % metszespont a kamera masik oldalan --> fekete pixel
        M=O+t*D; % ez a metszespont
        % arnyalas (Blinn-Phong): ambiens + diffuz + spekularis
        N=(M-C)/norm(M-C); % normalvektor (gomb kozeppont --> metszespont)
        ML=(L-M)/norm(L-M); % metszespont --> pontfeny
        MO=(O-M)/norm(O-M); % metszespont --> kamera
        img(h-iy+1,ix,:)=min(clrAmbient+max(N*ML',0)*clrSphere+clrLight.*max(N*(ML+MO)'/norm(ML+MO),0).^shininess,1);
    end
end

%% abrazolas
figure(1);
imshow(img);