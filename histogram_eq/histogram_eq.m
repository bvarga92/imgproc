clear all;
clc;

I=imread('img1.jpg'); %a bemeneti kep (szurkearnyalatos, 8 bpp)

h=hist(double(I(:)),256);
N=numel(I);
Io=zeros(size(I),'uint8');
for ii=1:256
    Io(I==ii-1)=uint8(255/N*sum(h(1:ii)));
end
h_eq=hist(double(Io(:)),256);

subplot(221);
imshow(I);
subplot(222);
bar(h);
xlim([0 255]);
subplot(223);
imshow(Io);
subplot(224);
bar(h_eq);
xlim([0 255]);

imwrite(Io,'img1_eq.jpg');
