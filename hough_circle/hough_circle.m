clear all;
clc;

%% PARAMETEREK
inputFile='input.png'; % 1 bpp szinmelyseg, 0 fekete, 1 feher
rMin=10;
rMax=100;
rStep=1;

%% MATRIX OSSZEALLITASA
img=imread(inputFile);
H=zeros(size(img,2)+2*rMax,size(img,1)+2*rMax,length(rMin:rStep:rMax));
for y=0:size(img,1)-1
    for x=0:size(img,2)-1
        if img(size(img,1)-y,x+1)==0
            idxR=0;
            for r=rMin:rStep:rMax
                idxR=idxR+1;
                for theta=0:2:360
                    cX=round(x-r*cos(theta*180/pi))+rMax+1;
                    cY=round(y-r*sin(theta*180/pi))+rMax+1;
                    H(cX,cY,idxR)=H(cX,cY,idxR)+1;
                end
            end
        end
    end
end

%% KOR DETEKTALASA
[maxVote idx]=max(H(:));
r=round(idx/size(H,1)/size(H,2))*rStep+rMin;
cX=mod(idx,size(H,1))-rMax;
cY=mod(floor(idx/size(H,1)),size(H,2))-rMax;
fprintf('Kozeppont [%.1f  %.1f], sugar %.1f.\n',cX,cY,r);
figure(1);
imshow(img,[0 1]);
hold on;
plot(cX+r*cos(0:pi/100:2*pi),size(img,1)-cY-r*sin(0:pi/100:2*pi),'r','LineWidth',1);
hold off;
