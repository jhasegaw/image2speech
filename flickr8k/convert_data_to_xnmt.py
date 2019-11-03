#!/opt/packages/python/Python-3.5.2-icc-mkl/bin/python3

import numpy as np
import re
import os
import sys
import h5py

##############################################################
# Read the image vectors, find each image name
imgdir = 'cnnfeats'
imgfiles = os.listdir(imgdir)
all_images = {}
for filename in imgfiles:
    iname = re.sub(r'\.npz','',filename)
    inputfeats = np.load(imgdir+'/'+filename)['arr_0']
    all_images[iname]=np.reshape(inputfeats,(14*14,512))
print('Loaded {} image vectors'.format(len(all_images)))

###################################################################################
# Read the transcripts
phone_types = set()
splits = ['train','test','val']
transcripts = { split:{} for split in splits }
no_image = set()
no_transcript = set()
for split in splits:
    with open('image_%s.txt'%(split)) as splitfile:
        for line in splitfile:
            (imagename,imageext) = os.path.splitext(os.path.basename(line.split()[0]))
            if imagename in all_images:
                transcripts[split][imagename] = []
                for cindex in range(5):
                    try:
                        with open('labels/align_%s_%s.txt'%(imagename,cindex)) as ali_file:
                            phones = [ line.split()[0] for line in ali_file.readlines() ]
                            transcripts[split][imagename].append(phones)
                    except:
                        no_transcript.add('labels/align_%s_%s.txt'%(imagename,cindex))
                if len(transcripts[split][imagename])==0:   # 
                    del transcripts[split][imagename]
                    no_transcript.add(imagename)
                elif len(transcripts[split][imagename]) < 5 and split != 'train':
                    for cindex in range(len(transcripts[split][imagename]),5):
                        transcripts[split][imagename].append(transcripts[split][imagename][-1].copy())

###################################################################################
# Write the training data
num_written = 0
with open('flickr40k_phones_train.txt','w') as textfile:
    with open('flickr40k_tagged_train.txt','w') as tagfile:
        with h5py.File('flickr40k_cnnfeats_train.h5', "w") as h5file:
            for imagename in sorted(transcripts['train'].keys()):
                for phones in transcripts['train'][imagename]:
                    textfile.write(' '.join(phones)+"\n")
                    tagfile.write('%d %s_%s '%(num_written,imagename,cindex)+' '.join(phones)+'\n')
                    h5file.create_dataset('%d'%(num_written), data=all_images[imagename])
                    num_written += 1

###################################################################################
# Write the test and val data
for split in ['val','test']:
    with h5py.File('flickr40k_cnnfeats_%s.h5'%(split), "w") as h5file:
        for (nwritten,imagename) in enumerate(sorted(transcripts[split].keys())):
            h5file.create_dataset('%d'%(nwritten), data=all_images[imagename])
    for cindex in range(5):        
        with open('flickr40k_phones_%s%d.txt'%(split,cindex),'w') as textfile:
            with open('flickr40k_tagged_%s%d.txt'%(split,cindex),'w') as tagfile:
                for (nwritten,imagename) in enumerate(sorted(transcripts[split].keys())):
                    phones=transcripts[split][imagename][cindex]
                    textfile.write(' '.join(phones)+"\n")
                    tagfile.write('%d %s_%s '%(nwritten,imagename,cindex)+' '.join(phones)+'\n')

