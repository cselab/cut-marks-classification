function rf = reshape_features_color( features, n, col )



N = size(features,1);
M = size(features,2);

m = sqrt(M);

a = reshape( features(1:N,:), [N,m,m] );

n = min( n, floor(sqrt(N)) );

col_sep = ones(m,1);
row_sep = ones(1,m+2);


img = cell(3,1);

rf = [];
for i=1:n
    r = [];
    for j=1:n
        ind = (i-1)*n+j;
        
        for k=1:3
            cs = col(ind,k)*col_sep;
            rs = col(ind,k)*row_sep;
            img{k} = [ cs , squeeze(a(ind,:,:))' , cs ];
            img{k} = [ rs ; img{k} ; rs ];
        end
        
        img_ = cat(3,img{1},img{2},img{3});

        r = [ r , img_ ];
    end
    rf = [ rf  ; r ]; 
end
