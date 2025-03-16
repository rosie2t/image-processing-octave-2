%Exercise 1
%Read the image
%Sobel Filter
Im=imread('lighthouse.jpg');
PGR=[-1 0 1;-2 0 2;-1 0 1]; %Sobel filter
PGC=[1 2 1;0 0 0;-1 -2 -1]; %Sobel filter
GR=imfilter(Im,PGR);
GC=imfilter(Im,PGC);
Final_edges=round(sqrt((double(GR).^2+double(GC).^2)));
Gimage=uint8(Final_edges);
figure (1)
imshow(Gimage)
title('Edge detection response after filtering')
Final_edges=im2bw(Gimage, "Otsu");
figure(2)
imshow(Final_edges)
title('Final/binary edges detected')

%Read the image
%Roberts Filter
Im=imread('lighthouse.jpg');
PGR=[0 0 -1;0 1 0;0 0 0]; %Roberts filter
PGC=[-1 0 0;0 1 0;0 0 0]; %Roberts filter
GR=imfilter(Im,PGR);
GC=imfilter(Im,PGC);
Final_edges=round(sqrt((double(GR).^2+double(GC).^2)));
Gimage=uint8(Final_edges);
figure (1)
imshow(Gimage)
title('Edge detection response after filtering')
Final_edges=im2bw(Gimage, "Otsu");
figure(2)
imshow(Final_edges)
title('Final/binary edges detected')

%Read the image
%Prewitt Filter
Im=imread('lighthouse.jpg');
PGR=[-1 0 1;-1 0 1;-1 0 0]; %Prewitt filter
PGC=[1 1 1;0 0 0;-1 -1 -1]; %Prewitt filter
GR=imfilter(Im,PGR);
GC=imfilter(Im,PGC);
Final_edges=round(sqrt((double(GR).^2+double(GC).^2)));
Gimage=uint8(Final_edges);
figure (1)
imshow(Gimage)
title('Edge detection response after filtering')
Final_edges=im2bw(Gimage, "Otsu");
figure(2)
imshow(Final_edges)
title('Final/binary edges detected')

%Read the image
%Kirsch T=30
Im=imread('lighthouse.jpg');
figure
imshow(Im)
title('Initial image')
A=15;
mask1=[5 5 5;-3 0 -3;-3 -3 -3]/A;
mask2=[-3 5 5;-3 0 5;-3 -3 -3]/A;
mask3=[-3 -3 5;-3 0 5;-3 -3 5]/A;
mask4=[-3 -3 -3;-3 0 5;-3 5 5]/A;
mask5=[-3 -3 -3;-3 0 -3;5 5 5]/A;
mask6=[-3 -3 -3;5 0 -3;5 5 -3]/A;
mask7=[5 -3 -3;5 0 -3;5 -3 -3]/A;
mask8=[5 5 -3;5 0 -3;-3 -3 -3]/A;
B=imfilter(Im,mask1);
B=max(B,imfilter(Im,mask2,'replicate'));
B=max(B,imfilter(Im,mask3,'replicate'));
B=max(B,imfilter(Im,mask4,'replicate'));
B=max(B,imfilter(Im,mask5,'replicate'));
B=max(B,imfilter(Im,mask6,'replicate'));
B=max(B,imfilter(Im,mask7,'replicate'));
B=max(B,imfilter(Im,mask8,'replicate'));
figure
imshow(B)
title('First Kirsch edges of the image')
T=30;
BW = B > T;
figure
imshow(BW)
title('Final Kirsch edges of the image')

%Read the image
%Log Filter
Im=imread('lighthouse.jpg');
sigma=1.4;
Threshold=4;
Log_filter=LaGa(sigma);
BW1=imfilter(Im,Log_filter);
figure (1)
imshow(Im)
BW1(BW1<Threshold) = 0;
3
BW1(BW1>=Threshold) = 255;
figure (2)
imshow(BW1)
title('Final Log edges of the image')

%Read the image
%Canny Method
Im=imread('lighthouse.jpg');
figure (1)
imshow(Im)
title('Original image');
CannyDetectedEdges=edge(Im, "Canny");
figure (2)
imshow(CannyDetectedEdges)
title('Edges detected using Canny filter');


%Exercise 3
%Q10
clear all; close all; clc;
%Read the image and convert it to double precision
Im = imread('bridge.jpg');
Im = double(Im);
[mI, nI] = size(Im);
%Original Image
figure(1)
imshow(uint8(Im))
%Choose demo block
m = 7; n = 10; % (row, col)
A = Im(1+8*m:8+8*m, 1+8*n:8+8*n);
disp('Selected block values')
disp(A)
%Subtraction of the value 128
Is = Im - 128;
A = Is(1+8*m:8+8*m, 1+8*n:8+8*n);
disp('Subtraction of value 128')
disp(A)
%Estimation of 2D-DCT for 8x8 blocks of the image
fun = @dct2;
J = blockproc(Is, [8 8], fun);
figure(2)
imagesc(J);
colorbar();
title('2D-DCT')
%Selected block process
B = J(1+8*m:8+8*m, 1+8*n:8+8*n);
disp('2D-DCT of selected block')
disp(B)
figure(3)
imagesc(B);
colorbar();
title('2D-DCT of selected block')
%Quantization table Q10
Q10 = [
    80 60 50 80 120 200 255 255;
    55 60 70 95 130 255 255 255;
    70 65 80 120 200 255 255 255;
    70 85 110 145 255 255 255 255;
    90 110 185 255 255 255 255 255;
    120 175 255 255 255 255 255 255;
    245 255 255 255 255 255 255 255;
    255 255 255 255 255 255 255 255];
%Use of quantization table Q10
Q = Q10;
%Quantization of 2D-DCT coefficients
Jquant=round(blockproc(J,[8 8],'divq',Q));
C = Jquant(1+8*m:8+8*m, 1+8*n:8+8*n);
disp('Quantized 2D-DCT coefficients of selected block')
disp(C)
%Inverse quantization of 2D-DCT coefficients
Jdquant=blockproc(Jquant,[8 8],'multip',Q);
disp('Inverse quantization of 2D-DCT coefficients for selected block')
disp(Jdquant(1+8*m:8+8*m, 1+8*n:8+8*n))
%Application of 2D-IDCT on quantized 8x8 blocks
Iinv = blockproc(Jdquant, [8 8], @idct2);
disp('Application of 2D-IDCT for selected block')
disp(Iinv(1+8*m:8+8*m, 1+8*n:8+8*n))
%Addition of value 128 and rounding
Iinv = Iinv + 128;
Im_reconstructed = round(Iinv);
disp('Final reconstructed block adding value 128')
disp(round(Iinv(1+8*m:8+8*m, 1+8*n:8+8*n)))
%Huffman encoding
[encoded, dict] = huffman_encode(Jquant);
%Huffman decoding
decoded = huffman_decode(encoded, dict, size(Jquant));
%Restoration of quantization of 2D-DCT coefficients
decoded = blockproc(decoded, [8 8], Jdquant);
Im_reconstructed = blockproc(decoded, [8 8], @idct2) + 128;

%Final encoded image
figure(6)
imshow(uint8(Im_reconstructed))
title('Final encoded-compressed image')

function [encoded, dict] = huffman_encode(Im)
  symbols = unique(Im(:));
  freq = histc(Im(:), symbols);
  dict = huffmandict(symbols, freq / numel(Im));
  encoded = huffmanenco(Im(:), dict);
end

function decoded = huffman_decode(encoded, dict, Im_size)
  decoded = huffmandeco(encoded, dict);
  decoded = reshape(decoded, Im_size);
end


Q10 = [
    80 60 50 80 120 200 255 255;
    55 60 70 95 130 255 255 255;
    70 65 80 120 200 255 255 255;
    70 85 110 145 255 255 255 255;
    90 110 185 255 255 255 255 255;
    120 175 255 255 255 255 255 255;
    245 255 255 255 255 255 255 255;
    255 255 255 255 255 255 255 255];

Q50 = [
    16 11 10 16 24 40 51 61;
    12 12 14 19 26 58 60 55;
    14 13 16 24 40 57 69 56;
    14 17 22 29 51 87 80 62;
    18 22 37 56 68 109 103 77;
    24 35 55 64 81 104 113 92;
    49 64 78 87 103 121 120 101;
    72 92 95 98 112 100 103 99];

%Exercise 4
clear all;close all;clc;
I=imread('Vader 18.jpg');
I=rgb2gray(I);
figure (1), imshow(I)
title('Original image')
[M N]=size(I); Inew=zeros(M+2,N+2);
C=zeros(M,N); Maska=zeros(3,3);LBP=Inew;
Inew(2:M+1,2:N+1)=double(I(1:M,1:N));
for i=2:M+1
 for j=2:N+1
 T=Inew(i,j);
 Maska=Inew(i-1:i+1,j-1:j+1);
 Maska(2,2)=0;
 C2=sum(sum(Maska));
 B=find(Maska>T);
 empty_B=sum(sum(B));
 if empty_B>0
 C1=sum(sum(Maska(B)));
 Maska(1:3,1:3)=0;
 Maska(B)=1;
 number_1=sum(sum(Maska));
 else
 C1=0;
 Maska(1:3,1:3)=0;
 number_1=0;
 end
 LBP(i,j)=Maska(1,1)*2^7+Maska(1,2)*2^6+Maska(1,3)*2^5 ...
 +Maska(2,3)*2^4+Maska(3,3)*2^3+Maska(3,2)*2^2 ...
 +Maska(3,1)*2^1+Maska(2,1)*2^0;
 %Contrast
 C2=C2-C1;
 if number_1>0
 if 8-number_1>0
 C(i,j)=(C1/number_1)-(C2/(8-number_1));
 else
 C(i,j)=C1/number_1;
 end
 else
 C(i,j)=-(C2/(8-number_1));
 end
 end
end
[MC NC]=size(C);
HC=zeros(511,1);
x=zeros(511,1);
for i=-255:255
 x(256+i)=i;
end
I(1:M,1:N)=uint8(LBP(2:M+1,2:N+1));
figure (2), imshow(I)
title('LBP image')
figure (3), imhist(I)
title('LBP histogram')

img = imread('Vader 18.jpg');
gray_img = rgb2gray(img);
hist_brightness = hist(double(gray_img(:)), 256);
norm_hist_brightness = hist_brightness / sum(hist_brightness);
figure (1), imhist(img)
title('Brightness histogram')

clear all;close all;clc;
I=imread('Vader 18.jpg');
I=rgb2gray(I);
figure (1), imshow(I)
title('Original image')
[M N]=size(I); Inew=zeros(M+2,N+2);
C=zeros(M,N); Maska=zeros(3,3);LBP=Inew;
Inew(2:M+1,2:N+1)=double(I(1:M,1:N));
for i=2:M+1
 for j=2:N+1
 T=Inew(i,j);
 Maska=Inew(i-1:i+1,j-1:j+1);
 Maska(2,2)=0;
 C2=sum(sum(Maska));
 B=find(Maska>T);
 empty_B=sum(sum(B));
 if empty_B>0
 C1=sum(sum(Maska(B)));
 Maska(1:3,1:3)=0;
 Maska(B)=1;
 number_1=sum(sum(Maska));
 else
 C1=0;
 Maska(1:3,1:3)=0;
 number_1=0;
 end
 LBP(i,j)=Maska(1,1)*2^7+Maska(1,2)*2^6+Maska(1,3)*2^5 ...
 +Maska(2,3)*2^4+Maska(3,3)*2^3+Maska(3,2)*2^2 ...
 +Maska(3,1)*2^1+Maska(2,1)*2^0;
 %Contrast
 C2=C2-C1;
 if number_1>0
 if 8-number_1>0
 C(i,j)=(C1/number_1)-(C2/(8-number_1));
 else
 C(i,j)=C1/number_1;
 end
 else
 C(i,j)=-(C2/(8-number_1));
 end
 end
end
[MC NC]=size(C);
HC=zeros(511,1);
x=zeros(511,1);
for i=-255:255
 x(256+i)=i;
end
I(1:M,1:N)=uint8(LBP(2:M+1,2:N+1));
figure (2), imshow(I)
title('LBP image')
figure (3), imhist(I)
title('LBP histogram')
hist_brightness = hist(double(I(:)), 256);
norm_hist_brightness = hist_brightness / sum(hist_brightness);
figure (1), imhist(I)
title('Brightness histogram')
function dist_L1 = compute_L1(f1, f2)
    dist_L1 = sum(abs(f1 - f2));
end

function dist_L2 = compute_L2(f1, f2)
    dist_L2 = sqrt(sum((f1 - f2).^2));
end
%Calculation of L1 and L2 between two identical vectors (for example)
dist_L1_brightness = compute_L1(norm_hist_brightness, norm_hist_brightness);
disp(['L1 distance for brightness histograms: ', num2str(dist_L1_brightness)]);

dist_L2_brightness = compute_L2(norm_hist_brightness, norm_hist_brightness);
disp(['L2 distance for brightness histograms: ', num2str(dist_L2_brightness)]);

dist_L1_lbp = compute_L1(LBP, LBP);
disp(['L1 distance for LBP histograms: ', num2str(dist_L1_lbp)]);

dist_L2_lbp = compute_L2(LBP, LBP);
disp(['L2 distance for LBP histograms: ', num2str(dist_L2_lbp)]);












