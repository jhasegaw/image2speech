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
    #print('Loaded image name {} from file {}'.format(iname, filename))
    
print('Loaded {} image vectors'.format(len(all_images)))

##############################################################
# Read the train, dev, and test lists.
# Create an ordered list of all VALID captions for each split.
# VALID means that Markus was able to create a phone alignment for that audio file.
split_lists = {'train':[], 'val':[], 'test':[]}
missing = {'train':[], 'val':[], 'test':[]}
found = {'train':[], 'val':[], 'test':[]}
print('Loading split definitions')

for split in split_lists.keys():
    split_def = 'image_{}.txt'.format(split)
    print('split {} definition file is {}'.format(split,split_def))
    with open(split_def) as f:
        for line in f:
            line1=re.sub(r'/pylon2/ci560op/dmerkx/jsalt/Flicker8k_Dataset/','',line.rstrip())
            line2=re.sub(r'\.jpg /pylon2/ci560op/dmerkx/jsalt/flickr_split/image/.*','',line1)
            if line2 not in all_images:
                missing[split].append(line2)
                pass
            else:
                found[split].append(line2)
                for cindex in range(5):
                    cname='{}_{}'.format(line2,cindex)
                    fn = 'labels/align_{}.txt'.format(cname)
                    if os.path.isfile(fn) and (line2 in all_images):
                        split_lists[split].append(cname)

    print('Found {}, missing {}, loaded {} in {}'.format(len(found[split]),len(missing[split]),len(split_lists[split]),split))

###################################################################################
# Now, for each split, for each vectorname, read in its phone transcription, and write it out
phone_types = set()
for split in split_lists.keys():
    textfilename = 'flickr40k_phones_{}.txt'.format(split)
    taggedfilename = 'flickr40k_phones_{}_tagged.txt'.format(split)
    with open(textfilename,'w') as textfile:
        with open(taggedfilename,'w') as taggedfile:
            for cname in sorted(split_lists[split]):
                fn1 = 'labels/align_{}.txt'.format(cname)
                caption = ''
                with open(fn1) as ali_file:
                    for line in ali_file:
                        fields = line.split()
                        caption += (' '+fields[0])
                        phone_types.add(fields[0])
                textfile.write(caption+"\n")
                taggedfile.write(cname+"\t"+caption+"\n")
                #print(split+" "+cname+" "+caption+"\n")

##############################################################
# Create the phone_types.txt file, and write out the phone types
with open('phone_types.txt','w') as f:
    f.write(' '.join(sorted(phone_types))+'\n')

##############################################################
# Split the image vector dictionaries, and save them
for split in split_lists.keys():
    #npzdict = {}
    #npzfilename = 'flickr40k_cnnfeats_{}.npz'.format(split)
    with h5py.File('flickr40k_cnnfeats_%s.h5'%(split), "w") as h5file:
        for (cindex,cname) in enumerate(sorted(split_lists[split])):
            imagename = cname[:-2]
            h5file.create_dataset('%d'%(cindex), data=all_images[imagename])
            #dictname = '{}_{}'.format(cname,cindex)  # add the index, so that xnmt sorts correctly
            #print('Saving image {} to {} entry {}'.format(imagename,split,dictname))
            #npzdict[dictname] = all_images[imagename]
            #np.savez(npzfilename,**npzdict)


