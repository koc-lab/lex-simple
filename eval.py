## install easse and readability packages

from easse.sari import corpus_sari
from easse.fkgl import corpus_fkgl
import numpy as np
import pandas as pd

from readability import Readability



input_file = open("files/test_sentences.txt","r").read().strip().split('\n')
ref_file1 = open("files/test_labels_1.txt","r").read().strip().split('\n')
ref_file2 = open("files/test_labels_2.txt","r").read().strip().split('\n')
ref_file3 = open("files/test_labels_3.txt","r").read().strip().split('\n')
muss_test = open("files/muss_test.txt","r").read().strip().split('\n')
acces_test = open("files/access_test.txt","r").read().strip().split('\n')
recls_outputs = open("files/recls_outputs.txt","r").read().strip().split('\n')
lsbert_outputs = open("files/lsbert_outputs.txt","r").read().strip().split('\n')
lsbert_outputs_ourcwi = open("files/lsbert_outputs_ourcwi.txt","r").read().strip().split('\n')
uslt_noss= open("files/uslt_ss.txt","r").read().strip().split('\n')
uslt_ss = open("files/uslt_outputs.txt","r").read().strip().split('\n')

input_dc = Readability(' '.join(input_file)).dale_chall().score
muss_dc = Readability(' '.join(muss_test)).dale_chall().score
access_dc = Readability(' '.join(acces_test)).dale_chall().score
recls_dc = Readability(' '.join(recls_outputs)).dale_chall().score
lsbert_dc = Readability(' '.join(lsbert_outputs)).dale_chall().score
lsbert_ourcwi_dc = Readability(' '.join(lsbert_outputs_ourcwi)).dale_chall().score
uslt_noss_dc = Readability(' '.join(uslt_noss)).dale_chall().score
uslt_dc = Readability(' '.join(uslt_ss)).dale_chall().score

muss_fkgl = corpus_fkgl(muss_test)
access_fkgl = corpus_fkgl(acces_test)
recls_fkgl = corpus_fkgl(recls_outputs)
lsbert_fkgl = corpus_fkgl(lsbert_outputs)
lsbert_ourcwi_fkgl = corpus_fkgl(lsbert_outputs_ourcwi)
uslt_noss_fkgl = corpus_fkgl(uslt_noss)
uslt_fkgl = corpus_fkgl(uslt_ss)

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

sari_score_dict = {"access":[access_sari,access_fkgl,access_dc], 
                   "muss":[muss_sari,muss_fkgl,muss_dc], 
                   "recls":[recls_sari,recls_fkgl,recls_dc], 
                   "lsbert":[lsbert_sari,lsbert_fkgl,lsbert_dc], 
                   "lsbert_ourcwi":[lsbert_ourcwi_sari,lsbert_ourcwi_fkgl,lsbert_ourcwi_dc],
                   "uslt no ss":[uslt_noss_sari,uslt_noss_fkgl,uslt_noss_dc], 
                   "uslt":[uslt_sari,uslt_fkgl,uslt_dc]}
df = pd.DataFrame(sari_score_dict,index=['SARI', 'FKGL','DC'])

print(df)
