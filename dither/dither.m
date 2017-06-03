clear all;
clc;

I=imread('input.jpg'); %a bemeneti kep (szurkearnyalatos, 8 bpp)
B=1; %bitszam (1...8)

%% KVANTALAS DITHER NELKUL
Io=uint8(double(bitshift(I,B-8,8))/(2^B-1)*255);
imwrite(Io,sprintf('output_%db_1_none.jpg',B));

%% DITHER VELETLEN ZAJJAL
d=rand(size(I))*255/(2^B-1);
Io=zeros(size(I),'uint8');
for ii=1:(2^B-1)
    lo=ii*255/(2^B-1);
    hi=(ii+1)*255/(2^B-1);
    Io(((double(I)+d)>lo)&((double(I)+d)<=hi))=uint8(lo);
end
imwrite(Io,sprintf('output_%db_2_rand.jpg',B));

%% DITHER MATRIXSZAL (2x2)
D=[  0  170;
   255   85];
D=D/(2^B-1);
d=repmat(D,floor(size(I,1)/2),floor(size(I,2)/2));
rrem=mod(size(I,1),2);
crem=mod(size(I,2),2);
d=[d ; d(1:rrem,:)];
d=[d d(:,1:crem)];
Io=zeros(size(I),'uint8');
for ii=1:(2^B-1)
    lo=ii*255/(2^B-1);
    hi=(ii+1)*255/(2^B-1);
    Io(((double(I)+d)>lo)&((double(I)+d)<=hi))=uint8(lo);
end
imwrite(Io,sprintf('output_%db_3_mtx2.jpg',B));

%% DITHER MATRIXSZAL (4x4)
D=[  0  238   51  221;
   187   85  136  102;
    34  204   17  255;
   153  119  170  238];
D=D/(2^B-1);
d=repmat(D,floor(size(I,1)/4),floor(size(I,2)/4));
rrem=mod(size(I,1),4);
crem=mod(size(I,2),4);
d=[d ; d(1:rrem,:)];
d=[d d(:,1:crem)];
Io=zeros(size(I),'uint8');
for ii=1:(2^B-1)
    lo=ii*255/(2^B-1);
    hi=(ii+1)*255/(2^B-1);
    Io(((double(I)+d)>lo)&((double(I)+d)<=hi))=uint8(lo);
end
imwrite(Io,sprintf('output_%db_4_mtx4.jpg',B));

%% FLOYD-STEINBERG-ELJARAS (HIBATERJESZTESES DITHER)
Io=I;
for r=1:size(I,1)
    for c=1:size(I,2)
        q=uint8(double(bitshift(Io(r,c),B-8,8))/(2^B-1)*255);
        e=double(Io(r,c))-double(q);
        Io(r,c)=q;
        if size(I,2)>c
            Io(r,c+1)=uint8(double(Io(r,c+1))+3/8*e);
        end
        if size(I,1)>r
            Io(r+1,c)=uint8(double(Io(r+1,c))+3/8*e);
        end
        if (size(I,2)>c)&&(size(I,1)>r)
            Io(r+1,c+1)=uint8(double(Io(r+1,c+1))+1/4*e);
        end
    end
end
imwrite(Io,sprintf('output_%db_5_fs.jpg',B));
