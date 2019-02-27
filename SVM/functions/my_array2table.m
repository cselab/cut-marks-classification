function t = my_array2table( x )



[nrows,nvars] = size(x);
baseName = 'word';
varnames = matlab.internal.datatypes.numberedNames(baseName,1:nvars);
           
           
vars = mat2cell(x,nrows,ones(1,nvars));

if isempty(vars) 
    t = table.empty(nrows,nvars);
else
    t = table(vars{:});
end

t.Properties.VariableNames = varnames; 


end