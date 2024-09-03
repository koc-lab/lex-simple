from numpy import dot
from numpy.linalg import norm
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer,BertForMaskedLM
import pandas as pd
import torchtext
import re
import os
import scipy
import copy
import language_tool_python
tool = language_tool_python.LanguageTool('en-US')
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()

from readability import Readability

import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

glove = torchtext.vocab.GloVe(name="6B", # trained on Wikipedia 2014 corpus
                                                            dim=300)
import warnings
warnings.filterwarnings("ignore")

import spacy
nlp_ner = spacy.load("en_core_web_sm")

import nltk
nltk.download('punkt_tab')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

legal_base_tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
legal_base_model = BertForMaskedLM.from_pretrained("nlpaueb/legal-bert-base-uncased")
legal_base_model.to(device)


print("Check1: imports successful \n")

def feature_extractor(word,original_word,sentence):
  sugg = [word]
  try:
    elements = df_subtlex.loc[[word],['Zipf-value']]
    element = elements.iloc[0][0]    
    zipf = float(element)
  except:
    #print("zipf failed")
    zipf = float(1.5)
  try:
    length = (1 / len(word)) ** 3.78
  except:
    length = 0
  cos_sim = float(torch.cosine_similarity(glove[original_word].unsqueeze(0), glove[word].unsqueeze(0)))
  try:
    total_loss = calculatelmloss(sentence,original_word,word,legal_base_model,legal_base_tokenizer)
    if total_loss != 0:
      lm_perp = 1 / total_loss
    else:
      lm_perp = 0
      print("Zero LM loss found for the target word: **",original_word,"** and the suggestion: **",word,"**")
  except:
      print(r"LM loss could not be computed for the target word: **",original_word,"** and the suggestion: **",word,"**")
      lm_perp = 0
  return [cos_sim,lm_perp,zipf,length]
  
df_subtlex = pd.read_excel(r"SUBTLEX_frequency.xlsx")
subtlex_words = []
for i in range(df_subtlex.shape[0]):
        subtlex_words.append(df_subtlex.loc[i,'Word'])
df_subtlex.set_index('Word',inplace=True)
#print("Subtlex Words: \n", df_subtlex)

df_law = pd.read_excel(r"zpf_2.xlsx")
law_words = []
for i in range(df_law.shape[0]):
        law_words.append(df_law.loc[i,'Word'])
df_law.set_index('Word',inplace=True)
#print("Law Words: \n", df_law)

eng_file = open(r"english_words.txt", 'r')
Lines = eng_file.readlines()

count = 0
eng_words = []
# Strips the newline character
for line in Lines:
        line = re.sub('\n', '', line)
        count += 1
        eng_words.append(line)
longer_eng_file = open(r"longer_english_words.txt", 'r')
Lines2 = longer_eng_file.readlines()   

longer_eng_words = []
count2=0 
for line2 in Lines2:
        line2 = re.sub('\n', '', line2)
        count2 += 1
        longer_eng_words.append(line2)

eng_words += longer_eng_words
eng_words = list(set(eng_words))

for i in eng_words:
    if len(i) <= 2:
        eng_words.remove(i)

f = open(r"edited_complex_words_combined.txt",'r',encoding='ascii',errors="ignore")
complex_words_string = f.read()
complex_words_string = re.sub(', ', ',', complex_words_string)
complex_words_string = complex_words_string.lower()
complex_words = complex_words_string.split(',')
complex_words = sorted(complex_words, key=len,reverse= True)

  

def selection_maker(tokenizer,model,original_text, word_list):
        
        text = original_text
        inputs = tokenizer(original_text,return_tensors = 'pt')
        inputs.to(device)
        tokens = tokenizer.tokenize(original_text)
        inputs['labels'] = inputs.input_ids.detach().clone()
        original_token_length = copy.deepcopy(inputs['input_ids'][0]).cpu().numpy().size
        
          
        word_ids = []
        selection=[]
        replaced_words={}
        words_found = []
        data = re.sub('[^A-Za-z]+', ' ', original_text)
        
        tokenized_original_text = tokenizer.tokenize(data)

        text_list = data.split()        

        text = data
        
        token_dict = {}
        
        replaced_tokens = []
        
        selection_new = {}
        
        for i in word_list:
                                  
                if (i in text) :
                        if (len(i.split())>= 2 and i.split()[0] in text_list) or (i in text_list and ner_checker(original_text,i)):
                                toki = tokenizer.tokenize(i)
                                try:
                                        idn = tokenizer.convert_tokens_to_ids(toki[0])
                                        word_ids.append(idn)
                                except:
                                        pass
                                #text = text.replace(i,'[MASK]')
                                
                                
                                tokenized_text = tokenizer.tokenize(text)
                                
                                token_diff = 0
                                for token in tokenized_original_text:
                                  if token == toki[0] :
                                    break
                                  elif token in replaced_tokens:
                                    token_diff += 1

                                ##print("token_diff for token:",toki[0],token_diff)

                                


                                prior_masked_token_count = 0
                                for token in tokenized_text:
                                  if token == toki[0] :
                                    break
                                  elif token == '[MASK]':
                                    prior_masked_token_count += 1

                                """if prior_masked_token_count > 0 :
                                  prior_masked_token_count -= 1"""

                                ##print("prior_masked_token_count for token:",toki[0],prior_masked_token_count)    

                                if token_diff >= prior_masked_token_count:
                                  token_diff -= prior_masked_token_count
                                  
                                
                                #computes the token index difference 
                                #for tokens BEFORE the targeted word
                                #it stops when it reaches the target token, and it increments for every masked token
                                
                                for token in toki:
                                  replaced_tokens.append(token)
                                
                                inputs10 = tokenizer(text,return_tensors = 'pt')
                                                                                                                              
                                text = re.sub(i, '[MASK]', text)
                                words_found.append(i)
                                temp_word = i
                                
                                
                                nx = inputs10.input_ids[0].cpu().numpy()
                                ind = np.where(nx == idn)[0]
                                                        
                                ind = list(ind)
                                ##print(ind)
                                for j in ind:
                                
                                        new_j = j
                                        ##some change here
                                        
                                        for sel in selection:
                                          if sel > new_j:
                                            if sel in selection_new.keys():
                                              selection_new[sel] += (len(toki)-1)
                                            else:
                                              selection_new[sel] = (len(toki)-1)
                                              
                                            #selection_new is to keep track of the masked word that may have come after a token is masked.
                                            #for instance, let there be a word m masked at time t_0 at index i.
                                            #then at time t_1 let there be another word n masked at index j, such that j<i
                                            #then, it would be a problem if we didn't update the index i of the word m, as indices would slide
                                            #so, we must subtract the number of tokens in the above context from i

                                        selection.append(new_j)
                                        replaced_words[new_j] = temp_word
                                        
        masked_inputs = tokenizer(text,return_tensors = 'pt')
        masked_inputs.to(device)
        masked_token_length = copy.deepcopy(masked_inputs['input_ids'][0]).cpu().numpy().size


        token_length = original_token_length
        if masked_token_length + original_token_length > 510:
           token_length = 510 - masked_token_length
	   
           
        for selection_index in range(len(selection)):
          
          selection[selection_index] += token_length
 
        updated_replaced_words = {}
        for replaced_keys in replaced_words.keys():
          updated_replaced_words[replaced_keys + token_length] = replaced_words[replaced_keys]
		       
        prior_knowledge_dict = {}
        
        for selection_keys in selection_new.keys():
          prior_knowledge_dict[selection_keys + token_length] = selection_new[selection_keys]

        
        return word_ids,selection,updated_replaced_words,text,prior_knowledge_dict

def create_masked_lm(tokenizer,model,original_text,selection,masked_text):
        
        inputs = tokenizer(masked_text,return_tensors = 'pt')
        inputs.to(device)
        #input.cuda(1)
        tokens = tokenizer.tokenize(masked_text)
        inputs['labels'] = inputs.input_ids.detach().clone()
        
        #before_masking_tokenids = (inputs['labels'][0])

        ##apply selection index to inputs.input_ids, adding MASK tokens
        #inputs.input_ids[0,selection]=103
        ##now, the target sentence is masked according to the selection

        #encoded_seq = inputs['input_ids'][0] 
        ##input ids of the word, i.e., the encoded sequence
        
        #decoded_seq = tokenizer.decode(inputs['input_ids'][0]) 
        ##decoded sentence, containing [CLS], [SEP], [MASK] tokens
        
        

        inputs_2 = tokenizer(original_text,return_tensors = 'pt', truncation=True,max_length= 510-len(tokenizer.tokenize(original_text)))
        inputs_2.to(device)
        #input.cuda(1)
        inputs_2['labels'] = inputs_2.input_ids.detach().clone()
        for i in inputs:
                inputs[i] = torch.cat((inputs_2[i],inputs[i]),1)
        outputs = model(**inputs)

        masked_word_ids = (inputs['labels'][0]).tolist()

        return inputs, outputs,masked_word_ids

def suggestion_generator(tokenizer,model,inputs,outputs,selection,masked_word_ids,replaced_words,num_suggestions,selection_new):

        another_selection = selection.copy()

        for sele in selection_new.keys():
          index= another_selection.index(sele)
          another_selection[index] -= selection_new[sele]
        
        ##print("Selection indices:",selection)
        ##print("To be subtracted:",selection_new)
        ##print("Fixed selection:",another_selection)
        ##print("Input tokens shape:",inputs['input_ids'][0].shape)
        ##print("Output shape:", outputs.logits[0].shape)
        ##print(inputs['input_ids'][0])
        ##print(tokenizer.decode(inputs['input_ids'][0]))


        dict_suggest = {}
        

        for i in selection:

                #let's get rid of the original word probabilities 
                if i in selection_new.keys():
                  new_score = i - selection_new[i]
                else:
                  new_score  =  i
                
                try:
                  outputs.logits[0][new_score][masked_word_ids[new_score]] = 0
                except:
                  k = 0
                #   print(new_score)
                #   print(selection)
                #   print(another_selection)
                #   print(outputs.logits[0].shape)
                #   print(inputs['input_ids'][0])
                #   print(tokenizer.decode(inputs['input_ids'][0]))
                  
                  sz = outputs.logits[0].shape
                  new_score = sz[0]-1
                  
                #outputs.logits[0][i][masked_word_ids[i-len(masked_word_ids)]] = 0
                
                arr = outputs.logits[0][new_score].detach().cpu().numpy() 
                #arr is the array containing logit probabilities of i th masked word
                
                probabilities = nn.Softmax(dim=0)(outputs.logits[0][new_score])
                
                suggestion_list = []

                four_best = arr.argsort()[-num_suggestions:][::-1]
                four_prob = Nmaxelements(probabilities.tolist(),num_suggestions)
                # tensor_four_best = torch.FloatTensor(four_best.copy())
                tensor_four_best = torch.tensor(four_best.copy(),dtype=torch.int64)
                tensor_four_best.to(device)
                decoded_suggestions = tokenizer.decode(tensor_four_best)
                l_word = decoded_suggestions.split()
                
                for h in range(len(l_word)):
                        suggestion_list.append((l_word[h],four_prob[h]))
                dict_suggest[replaced_words[i]] = suggestion_list
        return dict_suggest

def substition_ranker(suggestion_dictionary,complex_words,weight_bert,weight_cos,weight_lm,original_text,model,tokenizer,weight_freq,weight_len):
        
        #global law2vec_failed,law2vec_successful
        #global subt_successful,subt_failed
        
        ranked_dictionary = {}
        suggested_word = {}
        for key in suggestion_dictionary:
                
                
                score_list = []
                for sugg in suggestion_dictionary[key]:
                        score = 0
                        candidate = sugg[0]
                        if (sugg[0] in eng_words) and (sugg[0] not in complex_words):
                                score += float(sugg[1]) * weight_bert
                                
                                try:
                                   elements = df_subtlex.loc[[sugg[0]],['Zipf-value']]
                                   element = elements.iloc[0][0]    
                                   score += float(element) * weight_freq
                                   #subt_successful += 1
                                   #print("Subt successful for",sugg[0])
                                except:
                                   score += float(1.5) * weight_freq
                                   #subt_failed += 1
                                   #print("Subt failed for",sugg[0])
                                    
                                score += weight_len * (1 / len(sugg[0])) ** 3.78
                                
                                if len(key.split()) == 1:
                                        cos_sim = float(torch.cosine_similarity(glove[key].unsqueeze(0),
                                                                            glove[sugg[0]].unsqueeze(0)))
                                        score += cos_sim * weight_cos                                    

                                        if weight_lm != 0:
                                          try:
                                              total_loss = calculatelmloss(original_text,key,sugg[0],model,tokenizer)
                                              if total_loss != 0:
                                                  lm_perp = 1 + 0.5 / total_loss + 0.5
                                              else:
                                                  lm_perp = 0
                                                  print("Zero LM loss found for the target word: **",key,"** and the suggestion: **",sugg[0],"**")
                                          except:
                                              print(r"LM loss could not be computed for the target word: **",key,"** and the suggestion: **",sugg[0],"**")
                                              lm_perp = 0
                                          score += lm_perp * weight_lm 
                                else:
                                        eff_key = key.split()[0]
                                        cos_sim = float(torch.cosine_similarity(glove[eff_key].unsqueeze(0),
                                                                                glove[sugg[0]].unsqueeze(0)))
                                        score += cos_sim * weight_cos

                                        if weight_lm != 0:
                                          total_loss = calculatelmloss(original_text,eff_key,sugg[0],model,tokenizer)
                                          
                                          if total_loss != 0:
                                              lm_perp = 1 / total_loss
                                          else:
                                              lm_perp = 1
                                              print("Zero LM loss found")
                                          score += lm_perp * weight_lm

                                pos_match = check_pos(original_text=original_text,target_word=key,suggestion=sugg[0])
                                if pos_match:
                                        pass
                                else:
                                        score = 0
                                
                        else:
                                pass
                        score_list.append([sugg[0],score])
                
                """
                score_list = []
                
                cand_list = []
                
                for sugg in suggestion_dictionary[key]:
                  cand_list.append(sugg[0])
                  
                for word1 in cand_list:
                
                    score1 = neural_ranker(word1=word1,wordlist=cand_list,original_word=key,original_sentence=original_text)
                    score_list.append([word1,score1.detach().cpu().numpy()])
                
                ranked_dictionary[key] = score_list
                
                scores=[]
                
                for i in score_list:
                        scores.append(i[1])
                scores = np.array(scores)
                index = scores.argmax()
                suggested_word[key] = score_list[index][0]
                """
                
                ranked_dictionary[key] = score_list
                scores=[]
                for i in score_list:
                        scores.append(i[1])
                scores = np.array(scores)
                index = scores.argmax()
                suggested_word[key] = score_list[index][0]
                
                #ranked dictionary is a dictionary which keeps track of the scores for different
                #substition candidates for a given complex word

                #suggested_word is a dictionary which keeps track of a complex word and 
                #its highest scored substitution pair
        
        """first_five_dict = {}
        for k in ranked_dictionary:
          points = {}
          for v in ranked_dictionary[key]:
            points[v[0]] = ranked_dictionary[key].index(v)
          best_points = sorted(Nmaxelements(list(points.keys()), 5))
          list_for_best = []
          
          for point in best_points:
             list_for_best.append(ranked_dictionary[key][points[point]])
          first_five_dict[k] = list_for_best

        print(first_five_dict)"""

        ##print(suggested_word)
        

        return suggested_word,ranked_dictionary

print("Check2: substitution ranking successful \n")

def Nmaxelements(list1, N):
                final_list = []

                for i in range(0, N): 
                        max1 = 0
                            
                        for j in range(len(list1)):     
                                if list1[j] > max1:
                                        max1 = list1[j];
                                            
                        list1.remove(max1);
                        final_list.append(max1)
                return final_list

def ner_checker(sentence,word):
    document = nlp_ner(sentence)
    tuples = [(X, X.ent_iob_, X.ent_type_) for X in document]
    for token in tuples:
        if str(token[0]) == word and token[1] == 'O':
            return True
        else:
            pass
    return False
    
def sentence_builder(text,suggested_word_dict):

        for key in suggested_word_dict:
                if key in text:
                        text = re.sub(key,suggested_word_dict[key],text)
        return text

def true_correct(text,words = None):
    #try:
        matches = tool.check(text)
        my_mistakes = []
        my_corrections = []
        start_positions = []
        end_positions = []
        
        
        for rules in matches:
                if len(rules.replacements)>0:
                        mistake = text[rules.offset:rules.errorLength+rules.offset]
                        if words != None:
                          if mistake in words:
                                  start_positions.append(rules.offset)
                                  end_positions.append(rules.errorLength+rules.offset)
                                  my_mistakes.append(text[rules.offset:rules.errorLength+rules.offset])
                                  my_corrections.append(rules.replacements[0])
                        else:
                          start_positions.append(rules.offset)
                          end_positions.append(rules.errorLength+rules.offset)
                          my_mistakes.append(text[rules.offset:rules.errorLength+rules.offset])
                          my_corrections.append(rules.replacements[0])

        my_new_text = list(text)
        
        for m in range(len(start_positions)):
                for i in range(len(text)):
                        my_new_text[start_positions[m]] = my_corrections[m]
                        if (i>start_positions[m] and i<end_positions[m]):
                                my_new_text[i]=""
                
        my_new_text = "".join(my_new_text)

        return my_new_text
    #except:
     #   return text

def check_pos(original_text,target_word,suggestion):
        original_tokens = nltk.word_tokenize(original_text)
        suggested_text = re.sub(target_word,suggestion,original_text)
        suggestion_tokens = nltk.word_tokenize(suggested_text)
        flag = True
        try:
                ind = suggestion_tokens.index(suggestion)
                flag = nltk.pos_tag(original_tokens)[ind][1][0] == nltk.pos_tag(suggestion_tokens)[ind][1][0]					
        except:
                flag = False
        return flag

def calculatelmloss(original_text,target_word,suggestion,model,tokenizer):
        text = original_text
        text = re.sub(target_word,suggestion,text) #change sentence
        inputs  =  tokenizer.encode_plus(text,  return_tensors="pt", add_special_tokens = True, truncation=True,
                                                                    padding = 'max_length', return_attention_mask = True, max_length = 256)
        inputs.to(device)
        labels  = copy.deepcopy(inputs['input_ids'])
        data = re.sub('[^A-Za-z]+', ' ', text)
        original_text_list = data.split()
        word_count = len(original_text_list)
        original_index = original_text_list.index(suggestion)
        total_loss = 0.0
        flag_next = False
        for i in range(-2,3):
                if i != 0:
                        text_list = data.split()

                        try:
                            text_list[original_index+i] = '[MASK]'
                        except:
                            flag_next = True

                        if flag_next == True :
                            try:
                                text_list[original_index+i-word_count] = '[MASK]'
                            except:
                                text_list[original_index+i+word_count] = '[MASK]'
                        
                        new_text = ' '.join(text_list)
                        #print(new_text)
                        text_list = original_text_list
                        inputs2 = tokenizer.encode_plus(new_text,  return_tensors="pt", add_special_tokens = True, truncation=True, 
                                                                                        padding = 'max_length', return_attention_mask = True, max_length = 256)
                        inputs2.to(device)
                        inputs['input_ids'] = inputs2['input_ids']
                        labels[inputs2['input_ids'] != tokenizer.mask_token_id] = -100 
                        outputs = model(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask'] , token_type_ids=inputs['token_type_ids'] , labels=labels)
                        lmloss = outputs.loss
                        lmlogits = outputs.logits
                        #print(lmloss)
                        #print(labels)
                        #print(inputs['input_ids'] )
                        arr = inputs.input_ids.cpu().numpy()[0]
                        ind = np.where(arr == 103)
                        #pred = torch.argmax(lmlogits[0][ind]).item()
                        #print("predicted token:", pred, tokenizer.convert_ids_to_tokens([pred]))	
                        labels  = copy.deepcopy(inputs['input_ids'])
                        total_loss += float(lmloss)
        return total_loss / 4
    
print("Check3: helpers successful")

def simplify_text(tokenizer,model,original_text,complex_words,weight_bert,weight_cos,weight_lm,weight_freq,weight_len):
        num_suggestions = 60
        _,new_selection,replaced_words,masked_text,selection_new = selection_maker(tokenizer,model,original_text,complex_words)
        inp, output, masked_words_ids = create_masked_lm(tokenizer,model,original_text,new_selection,masked_text)
        dict_sug = suggestion_generator(tokenizer,model,inp,output,new_selection,masked_words_ids,replaced_words,num_suggestions,selection_new)

        # print(dict_sug)

        sugg_word_dict,ranked_dict = substition_ranker(dict_sug,complex_words,weight_bert,weight_cos,weight_lm,original_text,model,tokenizer,weight_freq,weight_len)

        # print(ranked_dict)

        new_text = sentence_builder(original_text,sugg_word_dict)
        
        return [replaced_words, ranked_dict, sugg_word_dict, new_text]

def simplify_case(input_path,output_path,tokenizer,model,complex_words,weight_bert,weight_cos,weight_lm,weight_freq,weight_len):
        legal_doc=[]
        with open(input_path,'r',encoding = "latin", errors="ignore") as f:
                for line in f:
                        original_text = line
                        replaced_words, ranked_dict, sugg_word_dict, new_text = simplify_text(tokenizer, model, original_text, complex_words,weight_bert,weight_cos,weight_lm,weight_freq,weight_len)
                        #new_text = line
                        try:
                            values = sugg_word_dict.values()
                        except:
                            values = None
                        newer_text = true_correct(new_text,values)
                        legal_doc.append(newer_text)
        f.close()
        with open(output_path, 'w') as f:
                f.writelines(legal_doc)  
        f.close()
        
def simplify_case_nowrite(input_path,output_path,tokenizer,model,complex_words,weight_bert,weight_cos,weight_lm,weight_freq,weight_len):
        legal_doc=[]
        with open(input_path,'r',encoding = "latin", errors="ignore") as f:
                for line in f:
                        original_text = line
                        replaced_words, ranked_dict, sugg_word_dict, new_text = simplify_text(tokenizer, model, original_text, complex_words,weight_bert,weight_cos,weight_lm,weight_freq,weight_len)
                        #new_text = line
                        try:
                            values = sugg_word_dict.values()
                        except:
                            values = None
                        newer_text = true_correct(new_text,values)
                        legal_doc.append(newer_text.strip())
        #with open(output_path, 'w') as f:
                #f.writelines(legal_doc) 
        return legal_doc
        
trial_count = 3


if __name__ == '__main__':

        target_path = "../supreme_org_val.txt"
        simple_path = "uslt_noss_supreme_val.txt"

        bert_weight = 3.00
        lm_weight = 0.36
        cos_weight = 1.42
        freq_weight = 2.00
        len_weight = 4.61

        simple_lines = simplify_case_nowrite(target_path,simple_path, legal_base_tokenizer, legal_base_model, complex_words, bert_weight, cos_weight, lm_weight ,freq_weight,len_weight)
        simple_file = open(simple_path,"w")
        
        t = open(target_path,"r",encoding = "latin")
        tr = t.read()
        complex_lines = tr.split("\n")
        
        for i in range(len(simple_lines)):
                try:
                        simple_file.write(simple_lines[i].strip())
                except:
                        simple_file.write(complex_lines[i].strip())
                        print(f"problem with {i}")
                if i != len(simple_lines)-1:
                        simple_file.write("\n")
        
        simple_file.close()
