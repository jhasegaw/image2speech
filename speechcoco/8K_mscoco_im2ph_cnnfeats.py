#!/opt/packages/python/Python-3.5.2-icc-mkl/bin/python3

import numpy as np
import re
import os
import sys
import glob
import json

##############################################################
# First, create a directory from IMAGE_ID to JSON filename
# Hopefully, if I do this here, it will be faster than globbing
jsondirs=['/pylon2/ci560op/odette/data/mscoco/train2014/json/',
          '/pylon2/ci560op/odette/data/mscoco/val2014/json/']
#jsondirs=['/mnt/c/data/CORPUS-MSCOCO/train2014/json/']
print('Initializing the jsonmap')
sys.stdout.flush()
jsonmap = [ [] for n in range(650000) ]
print('Reading the jsonmap')
sys.stdout.flush()
for jsondir in jsondirs:
    print('Reading filenames from {}'.format(jsondir))
    sys.stdout.flush()
    jsonfilenames = os.listdir(jsondir)
    print('Read {} filenames, now parse them'.format(len(jsonfilenames)))
    sys.stdout.flush()
    for jsonfilename in jsonfilenames:
        matchdat = re.match(r'^(\d*)',jsonfilename)
        if matchdat:
            image_id = int(matchdat.group())
            jsonmap[image_id].append(jsondir+jsonfilename)
            #print(' {} -> {}'.format(image_id,jsondir+jsonfilename))
            #sys.stdout.flush()
        else:
            print(' NO IMAGE_ID in {}'.format(jsondir+jsonfilename))
            sys.stdout.flush()
##############################################################
# Define input and output data directories
imgbase = '/pylon5/ci560op/hasegawa/data/8K_mscoco/{}/cnnfeats/'
#audbase = '/pylon5/ci560op/hasegawa/data/8K_mscoco/{}/aud/'
audbase = '/pylon5/ci560op/londel/8K_mscoco/ploop_u100_s3_c4_bs400_labels/{}/'
outputbase = '/pylon5/ci560op/hasegawa/data/8K_mscoco/{}/im2ph/'
#imgbase = '/mnt/c/data/8K_mscoco/{}/cnnfeats/'
#audbase = '/mnt/c/data/8K_mscoco/{}/aud/'
#outputbase = '/mnt/c/data/8K_mscoco/{}/im2ph/'

numberoffiles=0
for subdir in ['dev','test','train']:
    ##############################################################
    # Read the image vectors, find each image name
    all_images = {}
    imgdir = imgbase.format(subdir)
    imgfiles = os.listdir(imgdir)
    print('Try to read {} image files from {}'.format(len(imgfiles),imgdir))
    sys.stdout.flush()
    for filename in imgfiles:
        iname = re.sub(r'\.npz','',filename)
        inputfeats = np.load(imgdir+'/'+filename)['arr_0']
        all_images[iname]=np.reshape(inputfeats,(14*14,512))
        #print('Loaded image name {} from file {}'.format(iname, filename))
    print('Loaded {} {} images'.format(len(all_images),subdir))
    sys.stdout.flush()

    npzdict = {}
    phones = {'phones_ref':[],'tagged_ref':[],'phones_aud':[],'tagged_aud':[]}
    nmissaud = 0
    nmissjson = 0

    ##############################################################
    # For each image file:
    for imgfilename in all_images.keys():
        # (1) Find the corresponding JSON file, and AUD file, if they exist
        filenumber = int(os.path.splitext(imgfilename)[0][-6:])
        jsonfilenames = jsonmap[filenumber]
        if len(jsonfilenames)==0:
            nmissjson += 1
            if nmissjson % 100 == 0:
                print('{} missing JSON, e.g., {}'.format(nmissjson,filenumber))
                sys.stdout.flush()
        for jsonfilename in jsonfilenames:
            jsonfilebase = os.path.splitext(os.path.basename(jsonfilename))[0]
            # First, load the reference phonelist from the JSON file
            nfiles = [ len(x) for x in phones.values() ]
            if not all([x==nfiles[0] for x in nfiles]):
                sys.exit('nfiles mismatch: {}'.format(nfiles))
                
            with open(jsonfilename) as f:
                jsondata = json.load(f)
                phonelist = [ x[2] for x in jsondata['timecode'] if x[1]=='PHO' ]
            # Second, load the audlist from the AUD file, if it exists
            audfilename = audbase.format(subdir) + jsonfilebase + '.lab'
            audlist = []
            if not os.path.exists(audfilename):
                nmissaud += 1
                if nmissaud % 100 == 0:
                    print('{} miss AUD, e.g., {}'.format(nmissaud,audfilename))
                    sys.stdout.flush()
            else:
                with open(audfilename) as f:
                    for line in f:
                        audlist.append(line.split()[0])

                # Third, generate output
                # cute pythonism: indentation of the following lines
                # determines whether they get done only if os.path.exists(audfilename),
                # or regardless of whether os.path.exists(audfilename)
                if nfiles[0] % 100 == 0:
                    print('Reading {}: {} and {}'.format(nfiles[0],jsonfilename,audfilename))
                    sys.stdout.flush()
                dictname = '{}_{}'.format(jsonfilebase,nfiles[0])
                npzdict[dictname] = all_images[imgfilename]

                phones['phones_ref'].append(' '.join(phonelist))
                phones['tagged_ref'].append(dictname+' '+' '.join(phonelist))
                phones['phones_aud'].append(' '.join(audlist))
                phones['tagged_aud'].append(dictname+' '+' '.join(audlist))

    ######################################################################
    # (3) Write the four different types of phone files, and the NPZ file
    outputdir=outputbase.format(subdir)
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)
    for phone_type in ['phones_ref','tagged_ref','phones_aud','tagged_aud']:
        with open(outputdir+phone_type+'.txt',"w") as f:
            for line in phones[phone_type]:
                f.write(line+"\n")

    npzfilename = outputdir+'cnnfeats.npz'
    np.savez(npzfilename,**npzdict)


