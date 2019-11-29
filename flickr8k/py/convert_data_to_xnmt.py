#!/opt/packages/python/Python-3.5.2-icc-mkl/bin/python3

import numpy as np
import re
import os
import sys
import h5py
import random

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
# Read the English transcripts
vocab = set()
splits = ['train','test','val']
transcripts = { split:{} for split in splits }
no_image = set()
no_transcript = set()
for split in splits:
    with open('splits/image_%s.txt'%(split)) as splitfile:
        for line in splitfile:
            (imagename,imageext) = os.path.splitext(os.path.basename(line.split()[0]))
            if imagename in all_images:
                transcripts[split][imagename] = []
                for cindex in range(5):
                    try:
                        with open('labels/align_%s_%s.txt'%(imagename,cindex)) as ali_file:
                            phones = [ line.split()[0] for line in ali_file.readlines() ]
                            if len(phones)>0:
                                transcripts[split][imagename].append(phones)
                                vocab |= set(phones)
                    except:
                        no_transcript.add('labels/align_%s_%s.txt'%(imagename,cindex))
                if len(transcripts[split][imagename])==0:   # 
                    del transcripts[split][imagename]
                    no_transcript.add(imagename)
                elif len(transcripts[split][imagename]) < 5 and split != 'train':
                    for cindex in range(len(transcripts[split][imagename]),5):
                        transcripts[split][imagename].append(transcripts[split][imagename][-1].copy())

###################################################################################
# Read the Dutch transcripts
dutch_vocab = set()
imagename2split = { i:s for s in transcripts.keys() for i in transcripts[s].keys() }
dutch_transcripts = {s:{i:[] for i in transcripts[s].keys()} for s in transcripts.keys() }
with open('dutch_asr_output.txt') as f:
    for (linenum,line) in enumerate(f):
        fields = line.rstrip().split()
        imagename = '_'.join(fields[0].split('_')[2:4])
        if imagename in imagename2split and len(fields)>1:
            dutch_transcripts[imagename2split[imagename]][imagename].append(fields[1:])
            dutch_vocab |= set(fields[1:])

# Make sure that, if any image exists in both Dutch in English, Dutch has as many copies as English
for s in splits:
    dutch_missing = set()
    for i in transcripts[s].keys():
        if i not in dutch_transcripts[s]:
            dutch_missing.add(i)
        elif len(dutch_transcripts[s][i]) == 0:
            del dutch_transcripts[s][i]
            dutch_missing.add(i)
        elif len(dutch_transcripts[s][i]) < len(transcripts[s][i]):
            for cindex in range(len(dutch_transcripts[s][i]),len(transcripts[s][i])):
                dutch_transcripts[s][i].append(dutch_transcripts[s][i][-1].copy())
    if len(dutch_missing)>0:
        print('eng %s has %d, dutch missing %d'%(s,len(transcripts[s]),len(dutch_missing)))
        print(random.sample(dutch_missing, min(5,len(dutch_missing))))
    
###################################################################################
# Write the English and Dutch training data
nwrit = 0
with open('phones/eng_train.txt','w') as etextfile:
    with open('tagged/eng_train.txt','w') as etagfile:
        with open('phones/nld_train.txt','w') as ntextfile:
            with open('tagged/nld_train.txt','w') as ntagfile:
                with h5py.File('cnnfeats_train.h5', "w") as h5file:
                    for imagename in sorted(transcripts['train'].keys()):
                        if imagename not in dutch_transcripts['train']:
                            #print('*** %s in English but not Dutch transcripts'%(imagename))
                            pass
                        else:
                            etr = transcripts['train'][imagename]
                            ntr = dutch_transcripts['train'][imagename]
                            if len(etr) != len(ntr):
                                print('%s: nld=%d but eng=%d'%(imagename,len(ntr),len(etr)))
                            for cindex in range(min(len(etr),len(ntr))):
                                eph = etr[cindex]
                                nph = ntr[cindex]
                                etextfile.write(' '.join(eph)+"\n")
                                etagfile.write('%d %s_%s '%(nwrit,imagename,cindex)+' '.join(eph)+'\n')
                                ntextfile.write(' '.join(nph)+"\n")
                                ntagfile.write('%d %s_%s '%(nwrit,imagename,cindex)+' '.join(nph)+'\n')
                                h5file.create_dataset('%d'%(nwrit), data=all_images[imagename])
                                nwrit += 1

###################################################################################
# Write the English and Dutch test and val data
for split in ['val','test']:
    writefiles = set(transcripts[split].keys()) & set(dutch_transcripts[split].keys())
    if len(writefiles) < len(transcripts[split]) or len(writefiles) < len(dutch_transcripts[split]):
        print('%s nld has %d, eng has %d'%(split,len(dutch_transcripts[split]),len(transcripts[split])))
    with h5py.File('cnnfeats_%s.h5'%(split), "w") as h5file:
        for (nwritten,imagename) in enumerate(sorted(writefiles)):
            h5file.create_dataset('%d'%(nwritten), data=all_images[imagename])
    for cindex in range(5):        
        with open('phones/eng_%s%d.txt'%(split,cindex),'w') as etextfile:
            with open('tagged/eng_%s%d.txt'%(split,cindex),'w') as etagfile:
                with open('phones/nld_%s%d.txt'%(split,cindex),'w') as ntextfile:
                    with open('tagged/nld_%s%d.txt'%(split,cindex),'w') as ntagfile:
                        for (nwritten,imagename) in enumerate(sorted(writefiles)):
                            eph=transcripts[split][imagename][cindex]
                            etextfile.write(' '.join(eph)+"\n")
                            etagfile.write('%d %s_%s '%(nwritten,imagename,cindex)+' '.join(eph)+'\n')
                            nph=dutch_transcripts[split][imagename][cindex]
                            ntextfile.write(' '.join(nph)+"\n")
                            ntagfile.write('%d %s_%s '%(nwritten,imagename,cindex)+' '.join(nph)+'\n')

###################################################################################
# Write eng_vocab.txt and nld_vocab.txt
with open('phones/eng_vocab.txt','w') as f:
    for phone in sorted(vocab):
        f.write(phone+'\n')
with open('phones/nld_vocab.txt','w') as f:
    for phone in sorted(dutch_vocab):
        f.write(phone+'\n')

