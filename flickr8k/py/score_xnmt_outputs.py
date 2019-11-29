import nltk
import nltk.util
import nltk.translate.bleu_score
from nltk.metrics.distance import edit_distance
import math
from collections import Counter
import numpy as np

# Goal: compute multi-ref PER and BLEU
# scores for the table in the paper.

def insert_key_and_append_item(dic,key,itm):
    if key not in dic:
        dic[key] = []
    dic[key].append(itm)

def append_if_unique(lst,itm):
    if len(lst)==0 or lst[-1] != itm:
        lst.append(itm)

def corpus_pers(list_of_reflists, list_of_hyps):
    hyp_pers = []
    for (hyp,reflist) in zip(list_of_hyps,list_of_reflists):
        ref_pers = [ edit_distance(hyp,ref)/len(ref) for ref in reflist if len(ref)>0 ]
        if len(ref_pers)>0:
            hyp_pers.append(min(ref_pers))
    return(hyp_pers)

for lang in ['eng','nld']:
    for subc in ['val','test']:
        print('##############################################################################')
        print('# Language %s, Subcorpus %s'%(lang,subc))
        
        # 1. load the list_of_reflists and unigram_frequencies
        x = []
        for num in range(5):
            with open('phones/%s_%s%d.txt'%(lang,subc,num)) as f:
                x.append([ nltk.word_tokenize(line) for line in f.readlines() ])
        list_of_reflists = [ [x[0][n],x[1][n],x[2][n],x[3][n],x[4][n]] for n in range(len(x[0])) ]
    
        # 2. load the list_of_hyps
        params='SentShuffleBatcher_64_32'
        with open('xnmt_work/out/%s_%s_%s.txt'%(lang,subc,params)) as f:
            list_of_hyps = [ nltk.word_tokenize(line) for line in f.readlines() ]
        if len(list_of_hyps) != len(list_of_reflists):
            raise RuntimeError('%s reflist=%d, hyplist=%d'%(subc,len(list_of_reflists),len(list_of_hyps)))

        # Report BLEU and PER
        subc_bleu = nltk.translate.bleu_score.corpus_bleu(list_of_reflists,list_of_hyps)
        subc_pers = corpus_pers(list_of_reflists, list_of_hyps)
        print('BLEU=%g, PER=%g (of %d)'%(subc_bleu, np.average(subc_pers), len(subc_pers)))

        # Figure out which unigram is most frequent
        unigram_frequencies = Counter([w for reflist in list_of_reflists for ref in reflist for w in ref])
        mft = max(unigram_frequencies.items(), key=lambda x : x[1])[0]
        print('most frequent token is %s'%(mft))

        # Report chance BLEU and PER
        chance_list = [ [mft]*min([len(x) for x in reflist]) for reflist in list_of_reflists ]
        chance_bleu = nltk.translate.bleu_score.corpus_bleu(list_of_reflists,chance_list)
        chance_pers = corpus_pers(list_of_reflists, chance_list)
        print('Chance BLEU=%g, PER=%g (of %d)'%(chance_bleu, np.average(chance_pers), len(chance_pers)))

        # Report multi-ref BLEU of each transcriber against the others
        list_of_refhyps = [ rl[n] for rl in list_of_reflists for n in range(5) ]
        list_of_refreflists = [ rl[:n]+rl[(n+1):] for rl in list_of_reflists for n in range(5) ]
        refbleu = nltk.translate.bleu_score.corpus_bleu(list_of_refreflists,list_of_refhyps)
        ref_pers = corpus_pers(list_of_refreflists, list_of_refhyps)
        print('Ref BLEU=%g, PER=%g (of %d)'%(refbleu, np.average(ref_pers),len(ref_pers)))
    
