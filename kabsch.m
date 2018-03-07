clear all;
clc;

%% a pontok
N=1000; % pontok szama
P=rand(N,3); % kiindulo ponthalmaz
phiX=rand*2*pi;
phiY=rand*2*pi;
phiZ=rand*2*pi;
M=[1 0 0 ; 0 cos(phiX) -sin(phiX) ; 0 sin(phiX) cos(phiX)]*...
  [cos(phiY) 0 sin(phiY) ; 0 1 0 ; -sin(phiY) 0 cos(phiY)]*...
  [cos(phiZ) -sin(phiZ) 0 ; sin(phiZ) cos(phiZ) 0 ; 0 0 1] % veletlen forgatomatrix
Q=P*M; % elforgatott ponthalmaz
figure(1);
plot3(P(:,1),P(:,2),P(:,3),'b.',Q(:,1),Q(:,2),Q(:,3),'r.');
axis square;

%% eltolas az origoba
Pc=P-repmat(mean(P),N,1);
Qc=Q-repmat(mean(Q),N,1);

%% keresztkorrelacios matrix szingularis ertekek szerinti felbontasa
C=Pc'*Qc;
[U,S,V]=svd(C);
d=det(U*V');
R=U*[1 0 0 ; 0 1 0 ; 0 0 d]*V'

%% forgatas es kompenzalas az eltolassal
Pr=Pc*R;
Pr=Pr+repmat(mean(Q),N,1);
pause(1);
figure(1);
hold on;
plot3(Pr(:,1),Pr(:,2),Pr(:,3),'g.');
hold off;

%% animacio
F=50;
for ii=1:F
    Pr=Pc*real(R^(1/F))^ii+ii/F*repmat(mean(Q),N,1)+(F-ii)/F*repmat(mean(P),N,1);
    figure(2);
    plot3(P(:,1),P(:,2),P(:,3),'b.',Q(:,1),Q(:,2),Q(:,3),'r.',Pr(:,1),Pr(:,2),Pr(:,3),'g.');
    axis square;
    xlim([min([P(:,1);Q(:,1)]) max([P(:,1);Q(:,1)])]);
    ylim([min([P(:,2);Q(:,2)]) max([P(:,2);Q(:,2)])]);
    zlim([min([P(:,3);Q(:,3)]) max([P(:,3);Q(:,3)])]);
    pause(0.05);
end