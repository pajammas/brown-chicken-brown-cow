function example_classifier
clear all;
% change this path if you install the VOC code elsewhere
addpath([cd '/VOCcode']);

% initialize VOC options
VOCinit;
VOCopts.numWords=500; % number of visual words
VOCopts.phowOpts = {'Verbose', 0, 'Sizes', 4, 'Step', 15} ;
VOCopts.quantizer='vq';
VOCopts.numSpatialX = 2;
VOCopts.numSpatialY = 2;
% train and test classifier for each class
for i=1:VOCopts.nclasses
    cls=VOCopts.classes{i};
    classifier=build_vocab(VOCopts,cls);                  % train classifier
    classifier=train(VOCopts,classifier,cls); 
    test(VOCopts,cls,classifier);                   % test classifier
    [fp,tp,auc]=VOCroc(VOCopts,'comp1',cls,true);   % compute and display ROC
    
    if i<VOCopts.nclasses
        fprintf('press any key to continue with next class...\n');
        pause;
    end
end

 % -------------------------------------------------------------------------
% train classifier
function classifier = build_vocab(VOCopts,cls)
 % -------------------------------------------------------------------------
% load 'train' image set for class
[ids,classifier.gt]=textread(sprintf(VOCopts.clsimgsetpath,cls,'train'),'%s %d');
 try
        % try to load the visual vocabulary
        load(sprintf(VOCopts.exfdpath,'visual_vocab'),'vocab');
 catch
        % extract features for each image
        tic;
        fd={};
        for i=1:length(ids)
            % display progress
            if toc>1
                fprintf('%s: build_vocab: %d/%d\n',cls,i,length(ids));
                drawnow;
                tic;
            end
                % compute and save features
                I=imread(sprintf(VOCopts.imgpath,ids{i}));
                [frames, feat] = vl_phow(im2single(I), VOCopts.phowOpts{:}) ;
                feat=single(feat);
                fd{i}=feat;
                save(sprintf(VOCopts.exfdpath,ids{i}),'feat','frames');
                %[~, fd{i}] = vl_sift(single(rgb2gray(I))) ;
        end
        feat = vl_colsubset(cat(2,fd{:}), 10e4) ;
        feat = single(feat) ;
        vocab = vl_kmeans(feat, VOCopts.numWords, 'verbose', 'algorithm', 'elkan', 'MaxNumIterations', 50) ;
        save(sprintf(VOCopts.exfdpath,'visual_vocab'), 'vocab') ;
 end
 classifier.vocab=vocab;
% ------------------------------------------------------------------------- 
function  classifier=train(VOCopts,classifier,cls)
% -------------------------------------------------------------------------
 try
     load(sprintf(VOCopts.exfdpath,'histograms'),'hists') ; 
 catch
      hists = {} ;
      [ids,~]=textread(sprintf(VOCopts.clsimgsetpath,cls,'train'),'%s %d');
      for i=1:length(ids)
            % display progress
            if toc>1
                fprintf('%s: building the histogram: %d/%d\n',cls,i,length(ids));
                drawnow;
                tic;
            end
         I=imread(sprintf(VOCopts.imgpath,ids{i}));
        hists{i} = getImageDescriptor(VOCopts,classifier,ids{i},I);
      end

      hists = cat(2, hists{:}) ;
      save(sprintf(VOCopts.exfdpath,'histograms'),'hists') ;
 end
 classifier.hist=hists;


% -------------------------------------------------------------------------
function hist = getImageDescriptor(VOCopts,classifier,idsi,im)
% -------------------------------------------------------------------------

im = im2single(im) ;
width = size(im,2) ;
height = size(im,1) ;
try
   load(sprintf(VOCopts.exfdpath,idsi),'feat','frames'); 
catch
    % get PHOW features
    [frames, feat] = vl_phow(im2single(im), VOCopts.phowOpts{:}) ;
    save(sprintf(VOCopts.exfdpath,idsi),'feat','frames');
end

% quantize local descriptors into visual words
switch VOCopts.quantizer
  case 'vq'
    [~, binsa] = min(vl_alldist(classifier.vocab, single(feat)), [], 1) ;
  %case 'kdtree'
   % binsa = double(vl_kdtreequery(VOCopts.kdtree, model.vocab, ...
    %                              single(descrs), ...
    %                              'MaxComparisons', 50)) ;
end

for i = 1:length(VOCopts.numSpatialX)
  binsx = vl_binsearch(linspace(1,width,VOCopts.numSpatialX(i)+1), frames(1,:)) ;
  binsy = vl_binsearch(linspace(1,height,VOCopts.numSpatialY(i)+1), frames(2,:)) ;

  % combined quantization
  bins = sub2ind([VOCopts.numSpatialY(i), VOCopts.numSpatialX(i), VOCopts.numWords], ...
                 binsy,binsx,binsa) ;
  hist = zeros(VOCopts.numSpatialY(i) * VOCopts.numSpatialX(i) * VOCopts.numWords, 1) ;
  hist = vl_binsum(hist, ones(size(bins)), bins) ;
  hists{i} = single(hist / sum(hist)) ;
end
hist = cat(1,hists{:}) ;
hist = hist / sum(hist) ;


 % -------------------------------------------------------------------------
% run classifier on test images
function test(VOCopts,cls,classifier)
 % -------------------------------------------------------------------------
 
% load test set ('val' for development kit)
[ids,gt]=textread(sprintf(VOCopts.clsimgsetpath,cls,VOCopts.testset),'%s %d');

% create results file
fid=fopen(sprintf(VOCopts.clsrespath,'comp1',cls),'w');

% classify each image
tic;
for i=1:length(ids)
    % display progress
    if toc>1
        fprintf('%s: test: %d/%d\n',cls,i,length(ids));
        drawnow;
        tic;
    end
    
    try
        % try to load features
        load(sprintf(VOCopts.testResult,ids{i}),'hist');
    catch
        % compute and save features
        I=imread(sprintf(VOCopts.imgpath,ids{i}));
        hist= getImageDescriptor(VOCopts,classifier,ids{i},I);
        save(sprintf(VOCopts.testResult,ids{i}),'hist');
    end

    % compute confidence of positive classification
    c=classify(VOCopts,classifier,hist);
    
    % write to results file
    fprintf(fid,'%s %f\n',ids{i},c);
end

% close results file
fclose(fid);

% trivial feature extractor: compute mean RGB
function fd = extractfd(VOCopts,I)

[~, fd] = vl_phow(I, VOCopts.phowOpts{:}) ;
% fd = [];
% [nr,nc,nz] = size(I);
% for i=1:10,
% 	for j=1:10,
% 		dv = I(floor(1+(i-1)*nr/10):floor(i*nr/10),floor(1+(j-1)*nc/10):floor(j*nc/10),:);
% 		fd = [fd;sum(sum(double(dv)))/(size(dv,1)*size(dv,2))];
% %fd=squeeze(sum(sum(double(I)))/(size(I,1)*size(I,2)));
% 	end
% end
% fd = fd(:);

% trivial classifier: compute ratio of L2 distance betweeen
% nearest positive (class) feature vector and nearest negative (non-class)
% feature vector
function c = classify(VOCopts,classifier,fd)

d=sum(fd.*fd)+sum(classifier.hist.*classifier.hist)-2*fd'*classifier.hist;
dp=min(d(classifier.gt>0));
dn=min(d(classifier.gt<0));
c=dn/(dp+eps);