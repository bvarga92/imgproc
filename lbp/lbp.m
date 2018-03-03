clear all;
clc;

%% PARAMETEREK
inputFile='brick1.jpg';
% inputFile='brick2.jpg';
rotationInvariant=true;

%% LOCAL BINARY PATTERN
img=imread(inputFile);
LBP=zeros(size(img),'uint8');
for row=2:size(img,1)-1
    for col=2:size(img,2)-1
        LBP(row,col)=LBP(row,col)+(img(row  ,col+1)>img(row,col))*2^0;
        LBP(row,col)=LBP(row,col)+(img(row+1,col+1)>img(row,col))*2^1;
        LBP(row,col)=LBP(row,col)+(img(row+1,col  )>img(row,col))*2^2;
        LBP(row,col)=LBP(row,col)+(img(row+1,col-1)>img(row,col))*2^3;
        LBP(row,col)=LBP(row,col)+(img(row  ,col-1)>img(row,col))*2^4;
        LBP(row,col)=LBP(row,col)+(img(row-1,col-1)>img(row,col))*2^5;
        LBP(row,col)=LBP(row,col)+(img(row-1,col  )>img(row,col))*2^6;
        LBP(row,col)=LBP(row,col)+(img(row-1,col+1)>img(row,col))*2^7;
        if rotationInvariant
            minVal=LBP(row,col);
            for ii=1:7
                LBP(row,col)=bitshift(LBP(row,col),-1,'uint8')+mod(LBP(row,col),2)*2^7;
                if LBP(row,col)<minVal
                   minVal=LBP(row,col);
                end
            end
            LBP(row,col)=minVal;
        end
    end
end

%% ABRAZOLAS
figure();
subplot(221);
imshow(img);
title('Eredeti kep');
subplot(223);
imshow(LBP);
title('LBP minta');
subplot(122);
hist(double(LBP(:)),256);
xlim([-4 260]);
title('LBP hisztogram');