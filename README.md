# image2speech

This repo is my attempt to reconstruct all of the stages in the paper http://www.isle.illinois.edu/sst/pubs/2018/hasegawajohnson_isga18.pdf

It is not yet complete.  Currently it downloads the image set, and the captions, and the speech files, and their forced alignments, and generates cnnfeats from the images, and then runs XNMT to train the image-to-phone transducer.  But the phone-to-speech transducer isn't there yet.
