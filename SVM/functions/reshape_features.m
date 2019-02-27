function rf = reshape_features( features, n )

N = size(features,1);
M = size(features,2);

m = sqrt(M);

a = reshape( features(1:N,:), [N,m,m] );

n = min( n, floor(sqrt(N)) );

col_sep = ones(m,1);
row_sep = ones(1,(m+1)*n);

rf = [];
for i=1:n
    r = [];
    for j=1:n
        ind = (i-1)*n+j;
        r = [ r , squeeze(a(ind,:,:))' , col_sep ];
    end
    rf = [ rf  ; r ; row_sep ]; 
end

rf(end,:) = [];
rf(:,end) = [];