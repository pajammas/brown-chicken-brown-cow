function c = BOW_classify(VOCopts,classifier,fd)
  
    d=sum(fd.*fd)+sum(classifier.FD.*classifier.FD)-2*fd'*classifier.FD;
    dp=min(d(classifier.gt>0));
    dn=min(d(classifier.gt<0));
    c=dn/(dp+eps);
    
    
end