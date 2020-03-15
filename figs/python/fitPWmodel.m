function B=fitPWmodel(modelflag,p,wp,N)
% INPUT
% modelflag: 1 for Gaussian, 2 for Student's-t, 3 for Lattimore et al.
%(1992)
% p: data for 'real' cdf
% wp: w(p) - data for decision weights cdf
% N: number of bootstrapping samples
%
% OUTPUT
% B is a 2x2 matrix. First row is the fitted parameter values ([mu sigma] for
% Gaussian; [delta nu] for Student's-t; [delta gamma] for Lattimore et al.
% (1992). Second row is the standard error for the same parameters in the
% same order


% Define Nelder-Mead parameters
opts = optimset('MaxFunEvals',50000, 'MaxIter',10000);

% Choose model
switch(modelflag)
    case 1
        [bootstat,bootsam] = bootstrp(N,@Gaumodel,p,wp,opts);
    case 2
        [bootstat,bootsam] = bootstrp(N,@StudTmodel,p,wp,opts);
    case 3
        [bootstat,bootsam] = bootstrp(N,@Lattimoremodel,p,wp,opts);
end

% Results
B=zeros(2,2);
B(1,:)=mean(bootstat);
B(2,:)=std(bootstat);

end

function B=Gaumodel(x,yy,opts)
y = @(b,x) normcdf(x,b(1),b(2));
OLS = @(b) sum((y(b,norminv(x)) - yy).^2);
B = fminsearch(OLS, rand(2,1), opts);
end

function B=StudTmodel(x,yy,opts)
y = @(b,x) nctcdf(x,b(2),b(1));
OLS = @(b) sum((y(b,nctinv(x,1,0)) - yy).^2);
B = fminsearch(OLS, rand(2,1), opts);
end

function B=Lattimoremodel(x,yy,opts)
y = @(b,x) (b(1)*x.^b(2))./((b(1)*x.^b(2)) + (1-x).^b(2));
OLS = @(b) sum((y(b,x) - yy).^2);
B = fminsearch(OLS, rand(2,1), opts);
end