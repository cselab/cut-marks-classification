clear; clc

warning('off','images:imshow:magnificationMustBeFitForDockedFigure');

addpath(genpath('./functions/'))

demoFolder = './images/demo/';


% Load image data
imgDir = './data/2groups/post_1/';
% imgDir = './data/2groups/original/';
% imgDir = './images/butterfly/';
imds = imageDatastore( imgDir,'IncludeSubfolders',true,'LabelSource','foldernames' );



tbl = countEachLabel(imds);
categories = tbl.Label;

%%
Nstrong = 5;   % Show the Nstrong strongest points
Nf = 7;         % Show Nf*Nf features
Nv = 16;        % Size of Vocabulary, total number of words

color_table = [ 233,163,201 ;
                252,141,89  ;
                228,26,28   ;
                252,141,98  ;
                178,223,138 ;
                ]/255;
color = color_table(2,:);



%%

bag = bof( imds,...
                    'VocabularySize', Nv ,...
                    'PointSelection','Detector',...
                    'Verbose',true,...
                    'StrongestFeatures', 0.2,...
                    'Upright', true);
                
% bag = bof( imds,...
%                     'VocabularySize',Nv,...
%                     'PointSelection','Grid',...
%                     'GridStep',[36,36],...
%                     'Verbose',true,...
%                     'StrongestFeatures', 1,...
%                     'Upright', true);


%%

showsurf_points_features( imds, bag, 1, 'human', 40, 0, color, demoFolder );
return



%%
if 1
    showsurf_points_features( imds, bag, 1, 'human', Nstrong, Nf, color, demoFolder );
    showsurf_points_features( imds, bag, 2, 'human', Nstrong, Nf, color, demoFolder );
    showsurf_points_features( imds, bag, 3, 'human', Nstrong, Nf, color, demoFolder );
    showsurf_points_features( imds, bag, 4, 'human', Nstrong, Nf, color, demoFolder );
    
    showsurf_points_features( imds, bag, 1, 'nature', Nstrong, Nf, color, demoFolder );
    showsurf_points_features( imds, bag, 2, 'nature', Nstrong, Nf, color, demoFolder );
    showsurf_points_features( imds, bag, 3, 'nature', Nstrong, Nf, color, demoFolder );
    showsurf_points_features( imds, bag, 4, 'nature', Nstrong, Nf, color, demoFolder );
    
end


%%
cmap = parula(Nv);

show_vocabulary( bag.Vocabulary,  cmap )

%%
imgInd = invImgInd( bag );
addImages( imgInd, imds );

words = imgInd.ImageWords;
file  = imgInd.ImageLocation;

%%

show_image_with_words( imds, words, 1, 'human', cmap)
show_image_with_words( imds, words, 1, 'nature', cmap)





%%
%==========================================================================
%==========================================================================
%                  Local Functions
%==========================================================================
%==========================================================================


function show_image_with_words( imds, words, ki, label, cmap)

    imFolder = './images/demo/';
    
    k = index_image_from_labels( ki, imds.Labels, label );
    
    Nv = size(cmap,1);
    
    img = readimage( imds, k );
    loc = words(k).Location;
    ind = words(k).WordIndex;

    figure(4); clf
    
    imshow( img )
    hold on

    col = zeros( size(loc,1), 3);
    for i=1:Nv
        ii = ( ind==i );
        col(ii,:) = repmat( cmap(i,:), sum(ii), 1);
    end

    p=scatter( loc(:,1), loc(:,2),[], col );
    p.MarkerFaceColor = p.MarkerEdgeColor;

    t=title( [ char(imds.Labels(k)) '  ' num2str(k)] );
    t.FontSize=22; t.FontName='Times';

    imName = [ imFolder  'surf_img_words_' sprintf('%02d',k) ];
    saveas( gcf, imName ,'epsc');
    
    
    
    figure(5); clf
    count = histc( words(k).WordIndex,[0.5:1:(Nv+1)] ); count(end)=[];
    
    for i=1:Nv
        p=bar( i, count(i) ); hold on
        p.FaceColor = cmap(i,:);
    end

    ax=gca;
    ax.XAxis.TickValues = 1:1:Nv;
    axis square
    grid on

    t=title( [ char(imds.Labels(k)) '  ' num2str(k)] );
    t.FontSize=22; t.FontName='Times';

    imName = [ imFolder 'surf_hist_' sprintf('%02d',k) ];
    saveas( gcf, imName ,'epsc');

end






function showsurf_points_features( imds, bag, ki, label, Nstrong, Nf, color, imFolder )
    
    k = index_image_from_labels( ki, imds.Labels, label );

    img = readimage( imds, k );    

    % extract and plot SURF points
    points = bag.Points{k};
    strongest = points.selectStrongest( Nstrong );

    figure(1); clf
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

        figure(2); clf
        imshow( m )
        t=title( [ char(imds.Labels(k)) '  ' num2str(k) ] );
        t.FontSize=22; t.FontName='Times';

        imName = [ imFolder 'surf_features_' sprintf('%02d',k) ];
        saveas( gcf, imName ,'epsc');
    end
    
end


% returns the index of the k-th image with label 'label'
function N = index_image_from_labels( k, Labels, label )

    x = categorical({label});

    tf = ( Labels == x );

    Nmax = sum(tf);

    if(k>Nmax), k=Nmax; end

    N = find( tf==1, 1, 'first' );

    N = N + k - 1;

end


function show_vocabulary( img, col )

    imFolder = './images/demo/';
    
    img = ( img + 1 ) / 2 ;
    
    m = reshape_features_color( img, 10, col );   
    
    figure(3); clf
    
    imshow( m )
    
    t=title('Vocabulary (centroids)'); 
    t.FontSize=22; t.FontName='Times';
    
    imName = [ imFolder 'vocabulary' ];
    saveas( gcf, imName ,'epsc');

end
