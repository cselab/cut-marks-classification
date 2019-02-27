clc; clear

% subf = 'nature/';
% read_folder  = [ 'data/2groups/original/' subf];
% write_folder = [ 'data/2groups/post_3/'   subf ];

subf='';
read_folder  = [ 'data/uncategorized/arc/' subf];
write_folder = [ 'data/uncategorized/arc_post_1/' subf ];



if( strcmp(read_folder,write_folder) )
    error('Write folder is the same as read folder.')
end


imagefiles = dir( [read_folder '*.jpg' ]);      
nfiles = length(imagefiles); 


sz = zeros(nfiles,2);
for i=1:nfiles
    
    xold = imread([read_folder imagefiles(i).name]);
    
    sz(i,:) = size(xold);
    
    x = double(xold);
    x = x-mean(x(:));
    x = x/std(x(:));
    
%     x=imresize(x,[900,250],'bilinear');

%     figure(1);clf
%     imshow(xold)
%     figure(2);clf
%     imshow(x)
%     disp(i)
%     pause
   
    imwrite( x,  [write_folder imagefiles(i).name] );
end