#!/bin/sh
#####################################################################
# image2speech
#
# Mark Hasegawa-Johnson, Alan Black, Lucas Ondel, Odette Scharenborg, and Francesco Ciannella
# August, 2018
# Revised December, 2018
#
# Distributed under CC-BY 4.0: https://creativecommons.org/licenses/by/4.0/
# You are free to share and adapt this work for any purpose, under the following terms:
# Attribution — You must give appropriate credit, and indicate if changes were made.
# No additional restrictions — You may not restrict others from doing anything the license permits.
#
# This file, run.sh, downloads all packages necessary to train and test image2speech,
# downloads the flickr8k dataset and flicker_speech datasets,
# trains an image2speech system from them, and tests it.
#
# The downloads and training are quite slow.  You will certainly want to comment them
# out, after you have done them once.
#
#####################################################################
steps="09"

#####################################################################
# Step 01
# no pre-reqs
if [ -n "`echo $steps | grep 01`" ]; then
    echo "#######################################################################"
    echo ""
    echo "01: Download Davi Frossard's Tensorflow implementation of VGG16"
    echo "Karen Simonyan and Andrew Zisserman, ICLR 2015,"
    echo "Very Deep Convolutional Networks for Large-Scale Image Recognition"
    mkdir -p frossard_vgg16
    wget https://www.cs.toronto.edu/~frossard/vgg16/vgg16.py  -nd -P frossard_vgg16
    wget https://www.cs.toronto.edu/~frossard/vgg16/imagenet_classes.py  -nd -P frossard_vgg16
    wget https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz -nd -P frossard_vgg16
    # Remove a couple of lines that were written for some type of python 2.
    # Change the 
    grep -v 'print' frossard_vgg16/vgg16.py | grep -v 'for p in preds' > frossard_vgg16/vgg16class.py
fi
    
#####################################################################
# Step 02
# pre-req: flickr8k/flickr_image_urls.txt
#
if [ -n "`echo $steps | grep 02`" ]; then
    echo "#######################################################################"
    echo ""
    echo "02: Download the flickr8k images"
    echo "Micah Hodosh, Peter Young and Julia Hockenmaier, JAIR 47:853-899, 2013,"
    echo "Framing Image Description as a Ranking Task: Data, Models, and Evaluation Metrics"
    mkdir -p flickr8k/jpg
    wget -i flickr8k/flickr_image_urls.txt -P flickr8k/jpg
fi

#####################################################################
# Step 03
# pre-req: flickr8k/vgg16_flickr8k_cnnfeats_env.yaml
if [ -n "`echo $steps | grep 03`" ]; then
    echo "#######################################################################"
    echo ""
    echo "03: Create a conda environment for Davi Frossard's TensorFlow version of VGG16"
    conda env create -f flickr8k/vgg16_flickr8k_cnnfeats_env.yaml
fi

#####################################################################
# Step 04
# pre-req: flickr8k/vgg16_flickr8k_cnnfeats.py
if [ -n "`echo $steps | grep 04`" ]; then
    echo "#######################################################################"
    echo ""
    echo "04: Run the VGG16 model"
    cd flickr8k
    source activate vgg16_flickr8k_cnnfeats_env
    python vgg16_flickr8k_cnnfeats.py
    cd ..
fi


####################################################################
# Step 05
# no pre-reqs
if [ -n "`echo $steps | grep 05`" ]; then
    echo "#######################################################################"
    echo ""
    echo "05: Download the audio captions"
    echo "David Harwath and James Glass, ASRU 2015"
    echo "Deep Multimodal Semantic Embeddings for Speech and Images"
    cd flickr8k
    wget -nd https://groups.csail.mit.edu/sls/downloads/flickraudio/downloads/flickr_audio.tar.gz 
    tar xzvf flickr_audio.tar.gz
    rm flickr_audio.tar.gz
    cd ..
fi

####################################################################
# Step 06: download the forced alignments
# no pre-reqs
if [ -n "`echo $steps | grep 06`" ]; then
    echo "#######################################################################"
    echo ""
    echo "06: Download the forced alignments"
    echo "Computed by Markus Mueller, 2017 as part of "
    echo "Odette Scharenborg et al., 'Speech Technology for Unwritten Languages,' in review."
    cd flickr8k
    wget -nd http://isle.illinois.edu/flickr_labels.tar.gz
    tar xzf flickr_labels.tar.gz
    rm flickr_labels.tar.gz
    cd ..
fi

############################################################################
# Step 07: Sort transcriptions and features into train, dev and test sets
if [ -n "`echo $steps | grep 07`" ]; then
    echo "#######################################################################"
    echo ""
    echo "07: Sort transcriptions and features into train, dev and test sets"
    cd flickr8k
    python3 convert_data_to_xnmt.py
    cd ..
fi

############################################################################
# Step 08: Download and build dynet and XNMT
if [ -n "`echo $steps | grep 08`" ]; then
    echo "#######################################################################"
    echo ""
    echo "08: Download XNMT; create a new conda env for it; install requirements in the conda env"
    git clone https://github.com/neulab/xnmt
    cd xnmt
    conda create -n xnmt python=3.6
    source activate xnmt
    pip install -r requirements.txt
    python setup.py install
    conda deactivate
    cd ..
fi

############################################################################
# Step 09: Train the CNNFEATS-to-phones translation model using XNMT on force-aligned phones
if [ -n "`echo $steps | grep 09`" ]; then
    echo "#######################################################################"
    echo ""
    echo "09: Train the CNNFEATS-to-phones translation model using XNMT on force-aligned phones"
    cd flickr8k
    source activate xnmt
    python ../xnmt/xnmt/xnmt_run_experiments.py --settings=debug flickr8k_im2ph.yaml
    conda deactivate
    cd ..
fi

# Step 10: 
# Step 11: download festival and festvox
# Step 12: train the synthesis model from force-aligned audio captions?
# Step 13: Testing: run XNMT to convert test-corpus CNNFEATS to phones
# Step 14: Testing: run festvox to generate output speech from phones







