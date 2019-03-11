% This sript processes the original data and saves them in a separate
% folder
%
% author: George Arampatzis (garampat@ethz.ch)


clc; clear

subf = 'nature/';
read_folder  = [ 'data/2groups/original/' subf];
write_folder = [ 'data/2groups/post_3/'   subf ];

if( strcmp(read_folder,write_folder) )
    error('Write folder is the same as read folder.')
end


imagefiles = dir( [read_folder '*.jpg' ]);      
nfiles = length(imagefiles); 


sz = zeros(nfiles,2);
for i=1:nfiles
    
    xold = imread([read_folder imagefiles(i).name]);
    
    sz(i,:) = size(xold);
    
    % zero mean, std=1
    x = double(xold);
    x = x-mean(x(:));
    x = x/std(x(:));
    
    imwrite( x,  [write_folder imagefiles(i).name] );
end