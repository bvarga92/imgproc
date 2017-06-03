%A JPEG szabvany keptomoritesi eljarasanak demonstralasa.
clear all;
close all;
clc;

%kiindulo kep (szurkearnyalatos!)
x=imread('input.bmp');
N=size(x,1);
M=size(x,2);
if mod(N,8)~=0 || mod(M,8)~=0
   error('A kep mindket iranyu meretenek 8-cal oszthatonak kell lennie!'); 
end

%kvantalasi matrix
q=[16 11 10 16  24  40  51 61;
   12 12 14 19  26  58  60 55;
   14 13 16 24  40  57  69 56;
   14 17 22 29  51  87  80 62;
   18 22 37 56  68 109 103 77;
   24 36 55 64  81 104 113 92;
   49 64 78 87 103 121 120 101;
   72 92 95 98 112 100 103 99];

%8x8-as blokkokban diszkret koszinusz transzformaciot hajtunk vegre,
%majd kvantalunk. A kvantalt DCT-egyutthatokat 8x8-as blokkonkent
%egy-egy vektorba rendezzuk. A blokkok veget jelzo ertek (EOB) 127,
%azaz a 8 biten abrazolhato legnagyobb elojeles szam. Az egyes blokkok
%vektorait egy nagy vektorra fuzzuk ossze, ez lesz a kimenet.
temp=zeros(8,8);
out=[];
for ii=1:8:N-7
    for jj=1:8:M-7
        temp=round(dct2(double(x(ii:ii+7,jj:jj+7)))./q);
        zigzag=toZigzag(temp)';
        for k=64:-1:1
            if zigzag(k)==0
                zigzag(k)=[];
            else
                break;
            end                
        end
        zigzag(zigzag==127)=126;
        out=[out zigzag 127];
    end
end
disp(sprintf('%.0f%%-os meretcsokkenes!',100*(1-length(out)/(N*M))));

%visszaallitas
eobs=[0 find(out==127)];
x_restored=zeros(N,M);
for ii=1:8:N-7
    for jj=1:8:M-7
        zigzag=out((eobs((ii-1)*M/64+(jj-1)/8+1)+1):(eobs((ii-1)*M/64+(jj-1)/8+2)-1));
        temp=invZigzag([zigzag zeros(1,64-length(zigzag))]);
        x_restored(ii:ii+7,jj:jj+7)=idct2(temp.*q);
    end
end
x_restored=uint8(x_restored);

%abrazolas
figure('units','normalized','outerposition',[0 0 1 1]);
subplot(121);
imshow(x);
title('Eredeti');
subplot(122);
imshow(x_restored);
title('Kodolt-visszaallitott');
imwrite(x_restored,'output.bmp');
