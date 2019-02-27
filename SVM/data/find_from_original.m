clear;


setDir  = fullfile('.','2groups/post_2');
imds = imageDatastore( setDir,'IncludeSubfolders',true,'LabelSource','foldernames');

setDir  = fullfile('.','2groups/post_2');
imdsO = imageDatastore( setDir,'IncludeSubfolders',true,'LabelSource','foldernames');

% rcm2, rcm8, cm4
% trampling marks: tm18, tm1, tm4