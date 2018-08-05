clear all;
clc;

%% parameterek
w=300;                       % kep szelessege
h=300;                       % kep magassaga
maxReflections=5;            % maximalis visszaverodesi melyseg
C=[0.5 0.2 3.5];             % gomb kozeppontja
r=0.7;                       % gomb sugara
clrSphere=[0.79 0.51 0.22];  % gomb szine
reflSphere=0.4;              % gomb visszaverodesi tenyezoje
shininessSphere=50;          % gomb csillogasa
Pp=[0 -0.5 0];               % sik egy pontja
Np=[0 1 0];                  % sik normalvektora
clrPlane=@(M) ...            % sik szine (helyfuggo)
    (mod(floor(M(1)*1.5),2)==mod(floor(M(3)*1.5),2))*[0.3 1 1]+[0.7 0 0];
reflPlane=0.3;               % sik visszaverodesi tenyezoje
shininessPlane=50;           % sik csillogasa
L=[5 6 -10];                 % pontfeny pozicioja
clrLight=[1 1 1];            % pontfeny szine
clrAmbient=[0.05 0.05 0.05]; % ambiens feny szine
O=[0 0.5 -3];                % kamera pozicioja
Q=[0 0 1];                   % kamera iranya

%% metszespontokat kiszamito fuggvenyek
checkSphere=@(t) t((t>=0)&(isreal(t)));
intsctSphere=@(Di,Oi) min(checkSphere(roots([1 2*Di*(Oi-C)' (Oi-C)*(Oi-C)'-r^2])));
checkPlane=@(t) t((~isinf(t))&(t>=0));
intsctPlane=@(Di,Oi) checkPlane((Pp-Oi)*Np'/(Di*Np'));

%% szamitas
img=zeros(h,w,3);
x=linspace(-1,1,w);
y=linspace(-h/w,h/w,h);
for ix=1:length(x)
    for iy=1:length(y)
        D=[x(ix) y(iy) Q(3)]-O;
        D=D/norm(D);
        Di=D;
        Oi=O;
        intensity=1;
        pixColor=[0 0 0];
        for ir=1:maxReflections
            % megkeressuk a legkozelebbi metszespontot
            tSphere=intsctSphere(Di,Oi);
            tPlane=intsctPlane(Di,Oi);
            if isempty(tSphere) && isempty(tPlane); break; end;
            if isempty(tSphere) || any(tPlane<tSphere)
                object='p';
                M=Oi+tPlane*Di;
                N=Np;
                clr=clrPlane(M);
                refl=reflPlane;
                shininess=shininessPlane;
            else
                object='s';
                M=Oi+tSphere*Di;
                N=(M-C)/norm(M-C);
                clr=clrSphere;
                refl=reflSphere;
                shininess=shininessSphere;
            end
            ML=(L-M)/norm(L-M);
            MO=(Oi-M)/norm(Oi-M);
            % ha arnyekban van, feketen hagyjuk
            if object=='p'
                if ~isempty(intsctSphere(ML,M+N*0.0001)); break; end
            else
                if ~isempty(intsctPlane(ML,M+N*0.0001)); break; end
            end            
            % arnyalas (Blinn-Phong): ambiens + diffuz + spekularis
            pixColor=pixColor+intensity*(clrAmbient+max(N*ML',0)*clr+clrLight.*max(N*(ML+MO)'/norm(ML+MO),0).^shininess);
            % a sugar tovabbi utjanak kiszamitasa
            Oi=M+N*0.0001;
            Di=Di-2*(Di*N').*N;
            Di=Di/norm(Di);
            intensity=intensity*refl;
        end
        img(h-iy+1,ix,:)=min(max(pixColor,0),1);
    end
end

%% abrazolas
figure(1);
imshow(img);