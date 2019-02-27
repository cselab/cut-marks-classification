


imageIndex = invertedImageIndex(bag);

addImages(imageIndex, trainSet);


%%
k=1;

figure(k); clf


a=imageIndex;

words = a.ImageWords;
file  = a.ImageLocation;

img = readimage( trainSet, k );
loc = words(k).Location;
ind = words(k).WordIndex;

imshow(img)
hold on

rect = [ loc-3*ones(size(loc,1),2) , 6*ones(size(loc,1),2) ];
% rect = [ loc-5*ones(size(loc,1),2) , 8*ones(size(loc,1),2) ];
c = jet(size(word,2));
for i=1:size(rect,1)
    r=rectangle( 'Position',rect(i,:) );
    r.EdgeColor = c(ind(i),:);
    r.FaceColor = c(ind(i),:);
end

%%








