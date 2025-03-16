function h = LaGa(sigma)
sigma=1.5;
filtersize=2*ceil(3*sigma)+1;
siz   = (filtersize-1)/2;
std2   = sigma^2;   
[x,y] = meshgrid(-siz:siz,-siz:siz);
arg   = -(x.*x + y.*y)/(2*std2);
h  = exp(arg);
h(h<eps*max(h(:))) = 0;
sumh = sum(h(:));
if sumh ~= 0
    h  = h/sumh;
end
% Ipologismos Laplacian    
h1=h.*(x.*x+y.*y- std2)/(std2^2);
% Athroisma sintelestwn iso me miden
h=h1 - sum(h1(:))/(filtersize^2); 
end

