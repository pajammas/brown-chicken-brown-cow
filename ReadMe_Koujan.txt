First version of the pascal project (using Dense SIFT features as a bag of features):
local folder contains the following information:

- Dense SIFT features extracted from each image: ImageID_Fd.mat

- Bag of features (visual words) constructed from the features of the images in the training set: visual_vocab_fd.mat

- Concatenated vectors that represent the histograms of the training images: histograms_fd.mat

Inside the testResult folder there are the indiviual histograms of each image in the testing set:
ImageId_testHist.mat


Note: the provided scripts are the only modified ones among the development kit scripts.


I am currently working on incorporating the linear SVM classifier