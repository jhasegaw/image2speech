#!/bin/bash -i
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
    wget https://www.cs.toronto.edu/~frossard/vgg16/vgg16.py  -nd -P frossard_vgg16
    wget https://www.cs.toronto.edu/~frossard/vgg16/imagenet_classes.py  -nd -P frossard_vgg16
    wget https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz -nd -P frossard_vgg16
    # Remove a couple of lines that were written for some type of python 2.
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
# pre-req: frossard_vgg16/vgg16_cnnfeats_env.yaml
if [ -n "`echo $steps | grep 03`" ]; then
    echo "#######################################################################"
    echo ""
    echo "03: Create a conda environment for Davi Frossard's TensorFlow version of VGG16"
    conda env create -f frossard_vgg16/vgg16_cnnfeats_env.yaml
fi

#####################################################################
# Step 04
# pre-req: flickr8k/vgg16_flickr8k_cnnfeats.py
if [ -n "`echo $steps | grep 04`" ]; then
    echo "#######################################################################"
    echo ""
    echo "04: Run the VGG16 model to generate cnnfeats"
    cd flickr8k
    conda activate vgg16_cnnfeats_env
    python py/vgg16_flickr8k_cnnfeats.py
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
    python3 py/convert_data_to_xnmt.py
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
    conda activate xnmt
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
    conda activate xnmt
    for batcher in TrgBatcher SentShuffleBatcher; do
	for default_layer_dim in 64 128 256; do
	    for hidden_dim in 32 64 128; do
		filename=im2ph_${batcher}_${default_layer_dim}_${hidden_dim}
		echo "#######################################################################"
		echo "# creating and testing ${filename}.yaml"
		cat xnmt_work/im2ph.yaml |
		    sed "s/TrgBatcher/${batcher}/" |
		    sed "s/default_layer_dim: 128/default_layer_dim: ${default_layer_dim}/" |
		    sed "s/hidden_dim: 128/hidden_dim: ${hidden_dim}/" |
		    sed "s/name: im2ph/name: ${filename}/" |
		    sed "s:out/val.txt:out/val_${batcher}_${default_layer_dim}_${hidden_dim}.txt:" |
		    sed "s:out/text.txt:out/test_${batcher}_${default_layer_dim}_${hidden_dim}.txt:" |
		    cat > xnmt_work/${filename}.yaml
		if [ -f xnmt_work/logs/${filename}.log ]; then
		    rm xnmt_work/logs/${filename}.log
		fi
		python ../xnmt/xnmt/xnmt_run_experiments.py xnmt_work/${filename}.yaml
	    done
	done
    done
    conda deactivate
    cd ..
fi

#################################################################################
# # Step 10: Download festival and festvox
if [ -n "`echo $steps | grep 10`" ]; then
    echo "#######################################################################"
    echo ""
    echo "10: Download festival and festvox"
    
    # Try to make CMU speech tools
    wget http://festvox.org/packed/festival/2.5/speech_tools-2.5.0-release.tar.gz
    tar xzvf speech_tools-2.5.0-release.tar.gz
    cd speech_tools
    ./configure
    gnumake
    echo export PATH='$PATH':`pwd`/bin >> ~/.bashrc
    echo export ESTDIR=`pwd` >> ~/.bashrc
    echo export LD_LIBRARY_PATH='$LD_LIBRARY_PATH':`pwd`/lib >> ~/.bashrc
    echo export INCLUDE_PATH='$INCLUDE_PATH':`pwd`/include >> ~/.bashrc
    . ~/.bashrc
    cd ..
    
    # Try to make festival
    wget http://festvox.org/packed/festival/2.5/festival-2.5.0-release.tar.gz
    tar xzvf festival-2.5.0-release.tar.gz
    wget http://festvox.org/packed/festival/2.5/festlex_CMU.tar.gz
    tar xzvf festlex_CMU.tar.gz
    cd festival
    gnumake
    cd ..
    
    # Try to make festvox
    wget http://festvox.org/festvox-2.7/festvox-2.7.0-release.tar.gz
    tar xzvf festvox-2.7.0-release.tar.gz
    cd festvox
    ./configure
    make
    echo export FESTVOXDIR=`pwd` >> ~/.bashrc
    ~/.bashrc
    cd ..
fi

###############################################################################
# Step 11: create the clustergen working directory, and get a detailed listing of its contents
if [ -n "`echo $steps | grep 11`" ]; then
    source ~/.bashrc  # Get the necessary path variables
    echo "#######################################################################"
    echo ""
    echo "11: Create the clustergen working directory; copy wavs and phone transcriptions."

    mkdir flickr8k_cg
    cd flickr8k_cg
    $FESTVOXDIR/src/clustergen/setup_cg image2speech L1phones flickr8k_cg

    if [ -d wav ]; then rm -rf wav; fi
    cp -r ../flickr8k/flickr_audio/wavs wav
    awk '{printf("( %s \"",$2);for(i=3;i<NF;i++){printf("%s ",$i)};printf("%s\" )\n",$NF);}' ../flickr8k/flickr40k_tagged_train.txt > etc/txt.done.data

    # Create "phonedict.txt", a dictionary that maps each phone to itself
    python ../flickr8k/py/phonedict.py ../flickr8k/vocab.txt phonedict.txt
    # Copy its code into festvox/image2speech_L1phones_flickr8k.scm
    cp ../flickr8k/scm/phonedict.scm festvox/image2speech_L1phones_flickr8k.scm

    cd ..
fi

###############################################################################
# Step 12: use do_build to generate prompts
if [ -n "`echo $steps | grep 12`" ]; then
    source ~/.bashrc  # Get the necessary path variables
    echo "#######################################################################"
    echo ""
    echo "12: Run the do_build scripts, to generate prompt files for clustergen."
    echo "  Generates the files flickr8k_cg/prompt.lab/* and flickr8k_cg/prompt.utt/*."
    echo "  Please check those files, to make sure they look reasonable."
    cd flickr8k_cg

    ./bin/do_build build_prompts
    ./bin/do_build label
    ./bin/do_build build_utts

    cd ..
fi

###############################################################################
# Step 13: clustergen
if [ -n "`echo $steps | grep 13`" ]; then
    source ~/.bashrc  # Get the necessary path variables
    echo "#######################################################################"
    echo ""
    echo "13: Run clustergen to generate the synthetic voice image2speech_L1phones_flickr8k"

    cd flickr8k_cg

    ./bin/do_clustergen f0
    ./bin/do_clustergen mcep
    ./bin/do_clustergen voicing
    ./bin/do_clustergen combine_coeffs_v

    ./bin/do_clustergen generate_statenames
    ./bin/do_clustergen cluster
    ./bin/do_clustergen dur

    cd ..
fi
    
# Step 14: Testing: run festival(?) to generate output speech from the files
# flickr8k/flickr40k_phones_val_hyp*.txt and flickr8k/flickr40k_phones_test_hyp*.txt
# using the voice image2speech_L1phones_flickr8k.









