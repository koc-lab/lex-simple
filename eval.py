## install easse and readability packages

from easse.sari import corpus_sari
from easse.fkgl import corpus_fkgl
import numpy as np
import pandas as pd

from readability import Readability



input_file_og = open("raw_data/supreme_org_test.txt","r").read().strip().split('\n')
ref_file1_og = open("raw_data/supreme_test_labels1.txt","r").read().strip().split('\n')
ref_file2_og = open("raw_data/supreme_test_labels2.txt","r").read().strip().split('\n')
ref_file3_og = open("raw_data/supreme_test_labels3.txt","r").read().strip().split('\n')
muss_test_og = open("files/muss_test_supreme.txt","r").read().strip().split('\n')
acces_test_og = open("files/access_supreme_test.txt","r").read().strip().split('\n')[:200]
recls_outputs_og = open("files/recls_supreme_test.txt","r").read().strip().split('\n')
lsbert_outputs_og = open("files/lsbert_outputs_supreme_test.txt","r").read().strip().split('\n')
lsbert_outputs_ourcwi_og = open("files/lsbert_outputs_ourcwi_supreme_test.txt","r").read().strip().split('\n')
tst_outputs_og = open("files/gector_supreme_test.txt","r").read().strip().split("\n")
uslt_noss_og = open("files/uslt_noss_test_supreme.txt","r").read().strip().split('\n') #36.228448
uslt_ss_og = open("files/uslt_supreme_test.txt","r").read().strip().split('\n') #37.470484

scores_array = np.zeros((3,8,5))
for i in range(5):
    low = i*10
    high = (i+1)*10

    input_file = input_file_og[low:high]
    muss_test = muss_test_og[low:high]
    acces_test = acces_test_og[low:high]
    recls_outputs= recls_outputs_og[low:high]
    lsbert_outputs = lsbert_outputs_og[low:high]
    lsbert_outputs_ourcwi = lsbert_outputs_ourcwi_og[low:high]
    tst_outputs = tst_outputs_og[low:high]
    uslt_noss = uslt_noss_og[low:high]
    uslt_ss = uslt_ss_og[low:high]
    ref_file1 = ref_file1_og[low:high]
    ref_file2 = ref_file2_og[low:high]
    ref_file3 = ref_file3_og[low:high]

    input_dc = Readability(' '.join(input_file)).dale_chall().score
    muss_dc = Readability(' '.join(muss_test)).dale_chall().score
    access_dc = Readability(' '.join(acces_test)).dale_chall().score
    recls_dc = Readability(' '.join(recls_outputs)).dale_chall().score
    lsbert_dc = Readability(' '.join(lsbert_outputs)).dale_chall().score
    lsbert_ourcwi_dc = Readability(' '.join(lsbert_outputs_ourcwi)).dale_chall().score
    tst_dc = Readability(' '.join(tst_outputs)).dale_chall().score
    uslt_noss_dc = Readability(' '.join(uslt_noss)).dale_chall().score
    uslt_dc = Readability(' '.join(uslt_ss)).dale_chall().score

    muss_fkgl = corpus_fkgl(muss_test)
    access_fkgl = corpus_fkgl(acces_test)
    recls_fkgl = corpus_fkgl(recls_outputs)
    lsbert_fkgl = corpus_fkgl(lsbert_outputs)
    lsbert_ourcwi_fkgl = corpus_fkgl(lsbert_outputs_ourcwi)
    tst_fkgl = corpus_fkgl(tst_outputs)
    uslt_noss_fkgl = corpus_fkgl(uslt_noss)
    uslt_fkgl = corpus_fkgl(uslt_ss)

# muss_fkgl = Readability(' '.join(muss_test)).flesch_kincaid().score
# access_fkgl = Readability(' '.join(acces_test)).flesch_kincaid().score
# recls_fkgl = Readability(' '.join(recls_outputs)).flesch_kincaid().score
# lsbert_fkgl = Readability(' '.join(lsbert_outputs)).flesch_kincaid().score
# lsbert_ourcwi_fkgl = Readability(' '.join(lsbert_outputs_ourcwi)).flesch_kincaid().score
# tst_fkgl = Readability(' '.join(tst_outputs)).flesch_kincaid().score
# uslt_noss_fkgl = Readability(' '.join(uslt_noss)).flesch_kincaid().score
# uslt_fkgl = Readability(' '.join(uslt_ss)).flesch_kincaid().score

    muss_sari = corpus_sari(orig_sents=input_file,  
                sys_sents=muss_test, 
                refs_sents=[ref_file1,
                            ref_file2,  
                            ref_file3])
    access_sari = corpus_sari(orig_sents=input_file,  
                sys_sents=acces_test, 
                refs_sents=[ref_file1,
                            ref_file2,  
                            ref_file3])
    recls_sari = corpus_sari(orig_sents=input_file,  
                sys_sents=recls_outputs, 
                refs_sents=[ref_file1,
                            ref_file2,  
                            ref_file3])
    lsbert_sari = corpus_sari(orig_sents=input_file,  
                sys_sents=lsbert_outputs, 
                refs_sents=[ref_file1,
                            ref_file2,  
                            ref_file3])
    lsbert_ourcwi_sari = corpus_sari(orig_sents=input_file,  
                sys_sents=lsbert_outputs_ourcwi, 
                refs_sents=[ref_file1,
                            ref_file2,  
                            ref_file3])
    tst_sari = corpus_sari(orig_sents=input_file,  
                sys_sents=tst_outputs, 
                refs_sents=[ref_file1,
                            ref_file2,  
                            ref_file3])
    uslt_noss_sari = corpus_sari(orig_sents=input_file,  
                sys_sents=uslt_noss, 
                refs_sents=[ref_file1,
                            ref_file2,  
                            ref_file3])
    uslt_sari = corpus_sari(orig_sents=input_file,  
                sys_sents=uslt_ss, 
                refs_sents=[ref_file1,
                            ref_file2,  
                            ref_file3])


    score_dict = {"access":[access_sari,access_fkgl,access_dc], 
                   "muss":[muss_sari,muss_fkgl,muss_dc], 
                   "recls":[recls_sari,recls_fkgl,recls_dc], 
                   "lsbert":[lsbert_sari,lsbert_fkgl,lsbert_dc], 
                   "lsbert_ourcwi":[lsbert_ourcwi_sari,lsbert_ourcwi_fkgl,lsbert_ourcwi_dc],
                   "tst":[tst_sari, tst_fkgl, tst_dc], 
                   "uslt no ss":[uslt_noss_sari,uslt_noss_fkgl,uslt_noss_dc], 
                   "uslt":[uslt_sari,uslt_fkgl,uslt_dc]}
    c = 0
    for key in score_dict:
        for metric in range(3):
            scores_array[metric,c,i] = score_dict[key][metric]
        c += 1
    
final_score_dict = np.mean(scores_array,axis=2)
df_means = pd.DataFrame(final_score_dict,index=['SARI', 'FKGL','DC'],columns=['access','muss','recls','lsbert','lsbert_ourcwi','tst','uslt no ss','uslt'])
print(df_means)
stds = np.std(scores_array,axis=2)
df_stds = pd.DataFrame(stds,index=['SARI', 'FKGL','DC'],columns=['access','muss','recls','lsbert','lsbert_ourcwi','tst','uslt no ss','uslt'])
print(df_stds)
