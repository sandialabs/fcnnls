function [x, Pset, residterm] = fcnnls(A, y, x, Pset, Uset, formxp)
% Perform non-negatively constrained least squares with multiple RHS
% This routine minimizes ||y-A*x|| using crossproduct matrices
%
% I/O:  [x, Pset, E] = fcnnls(A, y, oldx, Pset, Uset, formxp);
%       [x, Pset] = fcnnls(A, y);
%       x = fcnnls(A, y, oldx, Pset, Uset);
%       x = fcnnls(A, y);
%
%       INPUTS:
%       A:  matrix of independent variables (dimensioned n by f)
%       y:  matrix of data (dimensioned n by m)
%       oldx: initial estimate for the solution (dimensioned f by m)
%       Pset:  passive constraint set (dimensioned f by m)
%       Uset: set of variables not constrained to nonnegativity (dimensioned f by m)
%            (use 1 for unconstrained, 0 for nonnegatively constrained variable)
%       formxp: omit or enter true to form cross products, enter false when 
%            feeding in cross products explicitly
%            (if formxp = false, enter A'*A and A'*y as inputs)
%       OUTPUTS:
%       x:  matrix of solutions (dimensioned f by m)
%       Pset:  passive constraint set (dimensioned f by m)
%       E:  residual term, to obtain actual SSE --> sum(y(:).^2)-E
%
% See Van Benthem and Keenan, J. Chemom., V.18(10), p441-450, 2004 for details
%
% Copyright 2019 National Technology & Engineering Solutions of Sandia, 
% LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, 
% the U.S. Government retains certain rights in this software.
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

% M.H. Van Benthem & M.R. Keenan, Sandia National Laboratories, 09/10/2001
% Revised: 09/13/2001, 01/22/2002, 12/10/2002, 11/14/2008, 11/14/2008
%          04/24/2013, 06/20/2019

narginchk(2,6)

% data not entered as the cross products, this is done here
[nobsA, nvar] = size(A);
[nobsy, ncases] = size(y);
if nobsA ~= nobsy
    error('The input matrices must be conformable')
end
if nargin < 6 || formxp
    Aty = A'*y;
    AtA = A'*A;
else
    Aty = y;
    AtA = A;
end
emptyx = false;
if nargin < 3
    x = nesolve(AtA, Aty);
elseif isempty(x)
    emptyx = true;
    x = nesolve(AtA, Aty);
elseif (size(x,1) ~= nvar) || (size(x,2) ~= ncases)
    error('oldx has incompatible dimensions')
end

if nargin < 4 || isempty(Pset)
    Pset = x > 0;
elseif (size(Pset,1) ~= nvar) || (size(Pset,2) ~= ncases)
    error('Pset has incompatible dimensions')
end

if nargin < 5 || isempty(Uset)
    Uset = false(nvar,ncases);
elseif (size(Uset,1) ~= nvar) || (size(Uset,2) ~= ncases)
    error('Uset has incompatible dimensions')
end
Pset = Pset|Uset;

if emptyx
    x(~Pset) = 0;
end

tol = sqrt(10*eps*norm(AtA,1)*nvar);
maxiter = 3*nvar;

w = zeros(size(Aty));

Uset = ~Uset;
d = x;
x = nesolve(AtA, Aty, Pset);
Fset = find(~all(Pset,1) | any(x<0 & Uset,1))';
sizx = size(x);
iter = 0;
Hset = Fset(any(x(:,Fset)<0 & Uset(:,Fset) & Pset(:,Fset),1));
while any(Hset) && iter<maxiter
    iter = iter + 1;
    [negvIdx,negcIdx] = find(x(:,Hset)<0 & Uset(:,Hset) & Pset(:,Hset));
    negIdx = sub2ind(sizx, negvIdx, Hset(negcIdx));
    zxd = d(negIdx)./(d(negIdx)-x(negIdx));
    zz = NaN(nvar,length(Hset));
    zz(sub2ind(sizx, negvIdx, negcIdx)) = zxd;
    [~,ind] = min(zz,[],1);
    alpha = sub2ind(sizx, ind(:), Hset);
    d(alpha) = 0;
    Pset(alpha) = false;
    x(:,Hset) = nesolve(AtA, Aty(:,Hset), Pset(:,Hset));
    Hset = Fset(any(x(:,Fset)<0 & Uset(:,Fset) & Pset(:,Fset),1));
end
if iter == maxiter
    warning('Sandia:NNLS','Maximum number of inner loop iterations exceeded');
end

d(:,Fset) = x(:,Fset);
w(:,Fset) = Aty(:,Fset)-AtA*x(:,Fset);
Fset = Fset(any(w(:,Fset)>tol & ~Pset(:,Fset),1));
while any(Fset) % main loop
    [~,mi] = max(w(:,Fset).*~Pset(:,Fset),[],1);
    Pset(sub2ind(sizx, mi(:), Fset)) = true;
    x(:,Fset) = nesolve(AtA, Aty(:,Fset), Pset(:,Fset));
    Hset = Fset(any(x(:,Fset)<0 & Uset(:,Fset) & Pset(:,Fset),1));
    iter = 0;
    while any(Hset) && iter<maxiter
        iter = iter + 1;
        [negvIdx,negcIdx] = find(x(:,Hset)<0 & Uset(:,Hset) & Pset(:,Hset));
        negIdx = sub2ind(sizx, negvIdx, Hset(negcIdx));
        zxd = d(negIdx)./(d(negIdx)-x(negIdx));
        zz = NaN(nvar,length(Hset));
        zz(sub2ind(sizx, negvIdx, negcIdx)) = zxd;
        [~,ind] = min(zz,[],1);
        alpha = sub2ind(sizx, ind(:), Hset);
        d(alpha) = 0;
        Pset(alpha) = false;
        x(:,Hset) = nesolve(AtA, Aty(:,Hset), Pset(:,Hset));
        Hset = Fset(any(x(:,Fset)<0 & Uset(:,Fset) & Pset(:,Fset),1));
    end
    if iter == maxiter
        warning('Sandia:NNLS','Maximum number of inner loop iterations exceeded');
    end
    d(:,Fset) = x(:,Fset);
    w(:,Fset) = Aty(:,Fset)-AtA*x(:,Fset);
    Fset = Fset(any(w(:,Fset)>tol & ~Pset(:,Fset),1));
end

if nargout >2
    residterm = sum(sum(x.*(Aty+w)));
end

function [K] = nesolve(CtC, CtA, Pset)
% Solve the set of equations CtA = CtC*K for the passive variables in set Pset
% using the fast combinatorial approach
K = zeros(size(CtA));
opts.SYM=true;
opts.POSDEF=true;
if (nargin == 2) || isempty(Pset) || all(Pset(:))
    K = linsolve(CtC, CtA, opts);
else
    [lVar,pRHS] = size(Pset);
    
    Rad2Lim = 52;
    if lVar<=Rad2Lim
        codedPset = 2.^(lVar-1:-1:0)*Pset;
        [sortedPset, sortedEset] = sort(codedPset);
        breaks = diff(sortedPset);
    else
        CodCnt = ceil(lVar./Rad2Lim);
        codedPset = zeros(pRHS,CodCnt);
        for ii = 1:CodCnt
            RadInd = (ii-1)*Rad2Lim+1:min(ii*Rad2Lim,lVar);
            codedPset(:,ii) = 2.^(length(RadInd)-1:-1:0)*Pset(RadInd,:);
        end
        [sortedPset, sortedEset] = sortrows(codedPset);
        breaks = any((diff(sortedPset,1,1)~=0)',1);
    end
    breakIdx = [0 find(breaks) pRHS];
    
    for k = 1:length(breakIdx)-1
        cols2solve = sortedEset(breakIdx(k)+1:breakIdx(k+1));
        vars = Pset(:,sortedEset(breakIdx(k)+1));
        % replace with original LS solution for R2014a
        K(vars,cols2solve) = linsolve(CtC(vars,vars), CtA(vars,cols2solve), opts);
        % K(vars,cols2solve) = CtC(vars,vars)\CtA(vars,cols2solve);
    end
end
