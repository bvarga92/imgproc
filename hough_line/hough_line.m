clear all;
clc;

%% PARAMETEREK
inputFile='input.png'; % 1 bpp szinmelyseg, 0 fekete, 1 feher
thetaMin=-90;
thetaMax=90;
thetaStep=1;
votingPoints=20;

%% MATRIX OSSZEALLITASA
img=imread(inputFile);
figure(1);
subplot(121);
imshow(img,[0 1]);
title('Eredeti');
H=zeros(length(thetaMin:thetaStep:thetaMax),2*sum(size(img)));
for y=0:size(img,1)-1
    for x=0:size(img,2)-1
        if img(size(img,1)-y,x+1)==0
            idxA=0;
            for theta=thetaMin:thetaStep:thetaMax
                idxA=idxA+1;
                idxR=round(x*cos(theta/180*pi)+y*sin(theta/180*pi))+sum(size(img));
                H(idxA,idxR)=H(idxA,idxR)+1;
            end
        end
    end
end

%% ELFORGATAS
[Hord,ord]=sort(H(:),'descend');
theta=-mean(mod(ord(1:votingPoints)-1,size(H,1))*thetaStep+thetaMin);
fprintf('Forgatas %.2f fokkal.\n',theta);
imgOut=imrotate(img,theta,'crop');
mask=~imrotate(true(size(img)),theta,'crop');
imgOut(mask)=1;
imwrite(imgOut,'output.png','BitDepth',1);
figure(1);
subplot(122);
imshow(imgOut,[0 1]);
title('Elforgatott');