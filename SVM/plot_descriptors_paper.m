clear;
clc

addpath(genpath('./functions/'))

load for_plot_2.mat


imFolder = 'data/2groups/';

include = {'cm27a.','cm26.','cms35.','tm.7.','tm.5.','tm.43','cms40.','tm.37.'};

f = @(x) contains(x, include  );


c = cellfun( f, imds.Files,'UniformOutput',false);

ind = find([c{:}]==1);


color_table = [ 233,163,201 ;
                252,141,89  ;
                228,26,28   ;
                252,141,98  ;
                178,223,138 ;
                ]/255;
color = color_table(2,:);


bag_all = bof(imds, 'VocabularySize',vocSize, 'PointSelection','Detector', 'Verbose',false );

%%


for i=1:length(ind)
    showsurf_points_features( imds, bag_all, ind(i), 40, 0, color, imFolder );
end




%%
setDir = './data/2groups/post_1/';
imds = imageDatastore(setDir,'IncludeSubfolders',true,'LabelSource','foldernames');

word = double( encode(bag, imds, 'Verbose',false));
Data.X = array2table( word );
Data.Y = imds.Labels;

[label,score] = Classifier.predictPostFcn(Data.X);

%%
if(1)
    prb = score( sub2ind(size(score), 1:size(score,1), grp2idx(label)') );
    prb = floor(1000*prb)/10;

    fprintf('\n');
    cprintf('*black','\tTrue Labels \t Predicted Labels \t Probability')
    fprintf('\n');
    for i=1:numel(prb)

        if(Data.Y(i)==label(i))
            col = 'black';
        else
            col = 'red';            
        end
        
        [~,nm,~] = fileparts(imds.Files{i});

        cprintf(col, '%3d)  %8s \t\t %8s \t\t    %3.1f%%   |    %s',i, Data.Y(i), label(i), prb(i), nm  );
        fprintf('\n');

    end
    fprintf('\n')
end


%%
clc
showsurf_points_features( imds, bag_all, 30, 40, 0, color, imFolder );
showsurf_points_features( imds, bag_all, 68, 40, 0, color, imFolder );









%% =======================================================================
%% =======================================================================
%% =======================================================================


function showsurf_points_features( imds, bag, k, Nstrong, Nf, color, imFolder )
    

    img = readimage( imds, k );    

    % extract and plot SURF points
    points = bag.Points{k};
    strongest = points.selectStrongest( Nstrong );

    fig = figure(1); clf
    
    imshow(img); hold on;

    fig.Position = [506 1 862 954];
    
%     t=title( [ char(imds.Labels(k)) '  ' num2str(k)] );
%     t.FontSize=22; t.FontName='Times';
    
    [~,nm,~] = fileparts(imds.Files{k});

%     imName = [ imFolder  'surf_img_' sprintf('%02d',k) ];
    imName = [ imFolder  'surf_' nm '.jpg' ];
    saveas( gcf, imName ,'jpg');

    p = strongest.plot;
%     strongest.Location
    
    p.Children(1).Color = color;
    p.Children(1).LineWidth = 3;

    p.Children(2).Marker = 'o';
    p.Children(2).MarkerFaceColor = color;
    p.Children(2).Color = color;
    p.Children(2).MarkerSize = 6;

    
    
%     imName = [ imFolder  'surf_points_' sprintf('%02d',k) ];
    imName = [ imFolder  'surf_points_' nm '.jpg' ];
    saveas( gcf, imName ,'jpg');

    if( Nf > 0 )
        % extract and plot SURF features 
        [features, ~] = extractFeatures(img, strongest);
    
        features = ( features + 1 ) / 2 ;
        
        m = reshape_features( features, Nf );

        figure(2); clf
        imshow( m );
%         t=title( [ char(imds.Labels(k)) '  ' num2str(k) ] );
%         t.FontSize=22; t.FontName='Times';

        imName = [ imFolder 'surf_features_' sprintf('%02d',k) '.jpg'];
        saveas( gcf, imName ,'jpg');
    end
    
end