%UNITM Fixed unit mapping
% 
%   W = UNITM
%   B = A*UNITM
%   B = UNITM(A)
%
% INPUT
%   A   Array or dataset
%
% OUTPUT
%   W   Unit mapping, if applied to dataset, it is returned unchanged
%   B   A
%
% DESCRIPTION
% This is a fixed unit mapping that maps any dataset on itself. There is
% also a trainable unit mapping, named UNITTM.
%
% SEE ALSO
% MAPPINGS, UNITTM

% Copyright: R.P.W. Duin, r.p.w.duin@prtools.org
% Faculty EWI, Delft University of Technology
% P.O. Box 5031, 2600 GA Delft, The Netherlands

function w = unitm (a,v)

	prtrace(mfilename);

	if (nargin == 0) | (isempty(a))
		w = mapping(mfilename,'fixed');
		w = setname(w,'Unit Mapping');
  else
    nodatafile(a);
    w = a;
	end

	return
