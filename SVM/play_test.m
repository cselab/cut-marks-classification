


imgDir = './data/2groups/post_2/';
imds = imageDatastore( imgDir,'IncludeSubfolders',true,'LabelSource','foldernames' );


bag = bof( imds,'VocabularySize',Nv,'PointSelection','Grid','GridStep',[12,12],'Verbose',true,'StrongestFeatures', 0.2 );
% bag = bof( imds,'VocabularySize',Nv,'PointSelection','Detector','Verbose',true,'StrongestFeatures', 0.2 );

%%

showsurf_points_features( imds, bag, 1, 'human', 10, 0, color, demoFolder );

                
%%
img = imds.readimage(1);
k=10;
x = round( bag.Points{1}.selectStrongest(1).Location );
a=img(x(2)+(-k:k),x(1)+(-k:k));

figure(3); clf
imshow(a)

imwrite(a,'images/tmp/tmp.jpeg')
imds2 = imageDatastore( 'images/tmp','IncludeSubfolders',true,'LabelSource','foldernames' );


% bag2 = bof( imds2,'VocabularySize',Nv,'PointSelection','Grid','GridStep',[12,12],'Verbose',true,'BlockWidth',32,'StrongestFeatures', 0.2 );
bag2 = bof( imds2,'VocabularySize',Nv,'PointSelection','Detector','Verbose',true,'StrongestFeatures', 0.2 );
a = imds2.readimage(1);



%%

showsurf_points_features( imds2, bag2, 1, 'tmp', 50, 0, color, demoFolder,1);

[features, ~] = extractFeatures(img,bag2.Points{1}, 'Upright', 1);


figure(4); clf
imshow((features+1)/2)








% returns the index of the k-th image with label 'label'
function N = index_image_from_labels( k, Labels, label )

    x = categorical({label});

    tf = ( Labels == x );

    Nmax = sum(tf);

    if(k>Nmax), k=Nmax; end

    N = find( tf==1, 1, 'first' );

    N = N + k - 1;

end





function showsurf_points_features( imds, bag, ki, label, Nstrong, Nf, color, imFolder, fig_ind )
    
    if ~exist('fig_ind','var')
        fig_ind=0;
    end

    k = index_image_from_labels( ki, imds.Labels, label );

    img = readimage( imds, k );    

    % extract and plot SURF points
    points = bag.Points{k};
    strongest = points.selectStrongest( Nstrong );

    figure(fig_ind+1); clf
    imshow(img); hold on;

    t=title( [ char(imds.Labels(k)) '  ' num2str(k)] );
    t.FontSize=22; t.FontName='Times';
    
    imName = [ imFolder  'surf_img_' sprintf('%02d',k) ];
    saveas( gcf, imName ,'epsc');

    p = strongest.plot;
    strongest.Location
    
    p.Children(1).Color = color;
    p.Children(1).LineWidth = 3;

    p.Children(2).Marker = 'o';
    p.Children(2).MarkerFaceColor = color;
    p.Children(2).Color = color;
    p.Children(2).MarkerSize = 6;

    imName = [ imFolder  'surf_points_' sprintf('%02d',k) ];
    saveas( gcf, imName ,'epsc');

    if( Nf > 0 )
        % extract and plot SURF features 
        [features, ~] = extractFeatures(img, strongest);
    
        features = ( features + 1 ) / 2 ;
        
        m = reshape_features( features, Nf );

        figure(fig_ind+2); clf
        imshow( m )
        t=title( [ char(imds.Labels(k)) '  ' num2str(k) ] );
        t.FontSize=22; t.FontName='Times';

        imName = [ imFolder 'surf_features_' sprintf('%02d',k) ];
        saveas( gcf, imName ,'epsc');
    end
    
end
