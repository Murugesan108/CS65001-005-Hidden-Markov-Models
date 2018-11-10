### Importing the required packages
import os
import pandas as pd
import numpy as np
from collections import Counter


#1.Hidden Markov Model 
## 1.1 Toy Problem #############


### Transition and Emission probabilities as given
table_1 = pd.DataFrame(data = {'H':[0.6,0.4,0.2],'L':[0.4,0.4,0.5],'END':[0.0,0.2,0.3]},
                       index = ['START','H','L'], columns = ['H','L','END'])

table_2 = pd.DataFrame(data = {'A':[0.2,0.3],'C':[0.3,0.2],'G':[0.3,0.2],'T':[0.2,0.3]},
                       index  = ['H','L'])

## Converting to log scale

table_1 = table_1.astype(float).apply(lambda x: np.log(x))
table_2 = table_2.astype(float).apply(lambda x: np.log(x))

seq = 'GCACTG'
seq_list = list(seq)

## Score function for Viterbi algorithm - uses table1 and table2

def score(word, k_t,k_t_min_1):
    
    if(k_t == 'END'):
        emission_prob = 0
    else:
        emission_prob = table_2.loc[k_t,word]
        
    return emission_prob + table_1.loc[k_t_min_1,k_t]


# #### Viterbi Algorithm

## Viterbi Algorithm
v = dict()
b = dict()
y_m = dict()

## Computing the score for the start tag
for k in table_2.index:
    v['1,'+ k] = score('G',k,'START')
    
for m in range(1,len(seq)):
    for k in table_2.index:
        #print(m)
        #print(k)
        
        ## Getting the value of v
        
        v[str(m+1)+','+ k] = np.max([np.sum( v[str(m)+','+ k_dash] + score(seq[m],k,k_dash)) for k_dash in table_2.index])
        
        ## Getting the value of b
        b[str(m+1)+','+ k] = table_2.index[np.argmax([np.sum( v[str(m)+','+ k_dash] + score(seq[m],k,k_dash)) for k_dash in table_2.index])]
        


#### This is equivalent to getting bm+1 
y_m[str(m+1)] =  table_2.index[ np.argmax([np.sum(v[str(m+1)+','+ each_k] + score(seq[m],'END',each_k)) for each_k in table_2.index])]

### Getting each value of y
for each_m in reversed(range(1,len(seq))):   
    y_m[str(each_m)] = b[str(each_m+1)+','+ y_m[str(each_m+1)]]


## Section 1.1
## Question 1
v_df = pd.DataFrame()
for key,values in v.items():
    row,col = key.split(",")
    v_df.loc[col,row] = values 
v_df


b_df = pd.DataFrame()
for key,values in b.items():
    row,col = key.split(",")
    b_df.loc[col,row] = values
b_df



## Section 1.1
## Question 2
y_m
## Hidden state of the DNA sequence GCACTG is HHLLLL


####################################################################################################
#1.2 POS Tagging

## Section 1.2
## Question 1. PREPROCESSING ####################
#### Reading the training data
pos_data = '.\\proj02\\'
training_data = open(pos_data+"trn.pos",'r')
training_pos = []


#### Creating a list of list for each of sentences
#### Also, adding the start and the end tokens for each sentence

for sent in training_data:
    
    ## Start token
    cur_token = [['<start>','START']]
    
    ## Splitting the sentence based on space
    words = sent.strip().split(" ")
    w_split = [w.split("/") for w in words]
    
    cur_token.extend(w_split)
    
    ## Adding the end token
    cur_token.extend([['<end>','END']])
    
    ## Storing the processed text for each sentence
    training_pos.append(cur_token)



### Getting the vocabulary list
vocab = [words for each in training_pos for words,tag in each]

## Setting the threshold (K = 10)
threshold = 10

## Calculating the frequency of each word
vocab_freq = Counter(vocab)
words_with_less_freq = {}

#### Finding out the words with frequency less then threshold
for word,count in vocab_freq.items():
    if(count < threshold):
        #print(word,": ",count)
        words_with_less_freq[word] = count


# Updating the words with UNK if the frequency is less than the threshold
training_pos_new = []
for sent in training_pos:
    sent_new = [[word,tag] if vocab_freq[word] > threshold else ['Unk',tag] for word,tag in sent]
    training_pos_new.append(sent_new)

### Creating vocabulary and tags list
vocab_list = [words for each in training_pos_new for words,tag in each]
tag_list = [tag for each in training_pos_new for words,tag in each]

#### Vocab list

print("Vocab size: ", len(set(vocab_list))-2) # Reducing 2 to remove <START> and <END> token that was added


### Creating a tuple with the format (tag, tag_next) i.e. each tag in the vocab mapped with its immediate next tag
tag_couple = [(tag_list[value],tag_list[value+1]) for value,tag in enumerate(tag_list) if tag != 'END']
tag_couple_counts = Counter(tag_couple)



## Section 1.2
## Question 2. #######################
################################ TRANSITION PROBABILITY TABLE ###################################################

### Creating a dictionary to hold the total count of each tag

dict_start_tag = {}
for yt in set(tag_list):
    for yt_min1 in set(tag_list):
        if(yt in dict_start_tag.keys()):
            dict_start_tag[yt] += tag_couple_counts[yt,yt_min1]
            
        else:
            dict_start_tag[yt] = tag_couple_counts[yt,yt_min1]


###### Creating a transition probabilty data frame
tags_all_t = list(set(tag_list))
tags_all_t.remove('START')
tags_all_t.remove('END')

index_set_t = ['START'] + tags_all_t
columns_set_t = tags_all_t + ['END']

trans_prob_empty = pd.DataFrame(columns = columns_set_t, index = index_set_t)
## Empty Data frame
transition_prob_df = trans_prob_empty




### Getting the transition probability
trainsition_prob = {}

#Creating a file of 
f = open("mr6rx-tprob.txt", "w+")

for yt in set(tag_list):
    for yt_min1 in set(tag_list):
        if(yt != 'START' and yt_min1 != 'END'):
            p_yt_yt_min1 = tag_couple_counts[yt_min1,yt]/dict_start_tag[yt_min1]
            
            transition_prob_df.loc[yt_min1,yt] = p_yt_yt_min1
            
            ## Writing out the output
            f.write(yt_min1+','+ yt+','+ str(p_yt_yt_min1)+"\n")

transition_prob_df


## Section 1.2
## Question 3. #######################
################################ Emission probability calculation ################################


#### Emission probability

### Creating a tuple with the format (vocabulary, tag) i.e. each word in the vocab mapped with its corresponding tag
word_tag_combo = [(words,tag) for each in training_pos_new for words,tag in each if tag not in ['START','END']]
word_tag_counter = Counter(word_tag_combo)


### Find the sum of counts for each of the POS tags
dict_tags = {}
for tags in set(tag_list):
    for vocab in set(vocab_list):
        if(tags in dict_tags.keys()):
            dict_tags[tags] += word_tag_counter[vocab,tags]
            
        else:
            dict_tags[tags] = word_tag_counter[vocab,tags]


## Creating a data frame for emission probabilty table
index_set_e = list(set(tag_list))
index_set_e.remove('START')
index_set_e.remove('END')

columns_set_e = list(set(vocab_list))
columns_set_e.remove('<start>')
columns_set_e.remove('<end>')

## Data frame to store the emission probability
emission_prob_empty =  pd.DataFrame(columns = columns_set_e, index = index_set_e)
emission_prob_df = emission_prob_empty


### Getting the emission probability

## File to store emission probability
f = open("mr6rx-eprob.txt", "w+")


for tags in set(tag_list):
    for vocab in set(vocab_list):
         
        if(tags not in ['START','END'] and vocab not in ['<start>','<end>']):
            
            p_xt_yt = word_tag_counter[vocab,tags]/dict_tags[tags]
            
            emission_prob_df.loc[tags, vocab ] = p_xt_yt
            
            f.write(tags+','+ vocab+','+ str(p_xt_yt)+"\n")


emission_prob_df


## Section 1.2
## Question 4. #### Handling zero probability values ######################################


### Getting the emission probability by handling zero probability values
alpha = 1

## Subtracting the 2 from vocab_list which has <start> and <end> tokens
V = len(set(vocab_list)) - 2
f = open("mr6rx-eprob-smoothed.txt", "w+")

emission_prob_df_alpha = pd.DataFrame(columns = columns_set_e, index = index_set_e)
##Calculating the probabilities
for tags in set(tag_list):
    for vocab in set(vocab_list):
        if(tags not in ['START','END'] and vocab not in ['<start>','<end>']):
            p_xt_yt = (alpha + word_tag_counter[vocab,tags])/(dict_tags[tags] +  V * alpha)
            emission_prob_df_alpha.loc[tags, vocab ] = p_xt_yt
            f.write(tags+','+ vocab+','+ str(p_xt_yt)+"\n")


### Getting the transition probability by handling zero probability values
beta = 1
N = len(set(tag_list))
#Creating a file of 
f = open("mr6rx-tprob-smoothed.txt", "w+")


transition_prob_beta_df = pd.DataFrame(columns = columns_set_t, index = index_set_t)
for yt in set(tag_list):
    for yt_min1 in set(tag_list):
        if(yt != 'START' and yt_min1 != 'END'):
            p_yt_yt_min1 = (beta + tag_couple_counts[yt_min1,yt])/(dict_start_tag[yt_min1] +  N * beta)
            
            transition_prob_beta_df.loc[yt_min1,yt] = p_yt_yt_min1
            ## Writing out the output
            f.write(yt_min1+','+ yt+','+ str(p_yt_yt_min1)+"\n")


## Section 1.2
## Question 5. Log space and Viterbi decoding ############################


### Converting the estimated probabilities into log space
transition_df = transition_prob_beta_df.astype(float).apply(lambda x: np.log(x), axis = 1)
emission_df = emission_prob_df_alpha.astype(float).apply(lambda x: np.log(x), axis = 1)


################## Reading the dev data set ############################
pos_data = '.\\proj02\\'
dev_data = open(pos_data+"dev.pos",'r')
dev_pos = []

#### Creating a list of list for each of sentences
#### Also, adding the start and the end tokens for each sentence

for sent in dev_data:
    
    ## Splitting the sentence based on space
    words = sent.strip().split(" ")
    w_split = [w.split("/") for w in words]
    
    ## Storing the processed text for each sentence
    dev_pos.append(w_split)


####### Function for score function and the viterbi algorithm ######################
## Score function for Viterbi algorithm 

def score(word, k_t,k_t_min_1, emission_df, transition_df):
    
    if(k_t == 'END'):
        emission_prob = 0
    else:
        emission_prob = emission_df.loc[k_t,word]
        
    return emission_prob + transition_df.loc[k_t_min_1,k_t]


## Viterbi Algorithm
def viterbi_algorithm(transition_df,emission_df,seq):
    ## Viterbi Algorithm
    v = dict()
    b = dict()
    y_m = dict()
    start_word = seq[0]

    ## Computing the score for the start tag
    for k in emission_df.index:
        v['1,'+ k] = score(start_word,k,'START',emission_df, transition_df)
        
    m = 0    
    for m in range(1,len(seq)):
        
        for k in emission_df.index:
            
            v[str(m+1)+','+ k] = np.max([np.sum( v[str(m)+','+ k_dash] + score(seq[m],k,k_dash,emission_df, transition_df)) for k_dash in emission_df.index])

            ## Getting the value of b
            b[str(m+1)+','+ k] = emission_df.index[np.argmax([np.sum( v[str(m)+','+ k_dash] + score(seq[m],k,k_dash,emission_df, transition_df)) \
											for k_dash in emission_df.index])]


    
    #### This is equivalent to getting bm+1 
    y_m[m+1] =  emission_df.index[ np.argmax([np.sum(v[str(m+1)+','+ each_k] + score(seq[m],'END',                                                                                     each_k,emission_df, transition_df))                                                   for each_k in emission_df.index])]

    ### Getting each value of y
    for each_m in reversed(range(1,len(seq))):   
        y_m[each_m] = b[str(each_m+1)+','+ y_m[each_m+1]]
        
    return(y_m)

######################################### END OF FUNCTION ###########################################################

### Creating lists with sentences and tags
dev_sentences = []
dev_tags = []
for dev_sent in dev_pos:
    ### Reading each line in the dev dataset and converting the tags which are not in the Vocabulary to 'Unk'
    sent_temp = [each[0] if each[0] in emission_prob_df.columns else 'Unk' for each in dev_sent]
    tags_temp = [each[1] for each in dev_sent]
    
    ## Creating list of tokens and tags
    
    dev_sentences.append(sent_temp)
    dev_tags.append(tags_temp)


### Running the Viterbi algorithm on the dev dataset
viterbi_df_out = pd.DataFrame()

## Looping through each of the sentence in dev data
for sent_no, each_sent in enumerate(dev_sentences):
    
    ##Calling the viterbi algorithm for the current line
    vit_out = viterbi_algorithm(transition_df,emission_df,each_sent)
    
    ##Recording the ground truth label for the tags
    current_tag = dev_tags[sent_no]
    current_tag.reverse()
    
    ##Creating a comparison dataframe containing both the ground truth and the predicted tags
    comparison_df = pd.DataFrame({'Grount_truth':current_tag,'Algorithm':list(vit_out.values())})
    viterbi_df_out = pd.concat([viterbi_df_out,comparison_df],axis = 0)



### Accuracy for alpha = 1 and beta = 1
100 * (viterbi_df_out.iloc[:,0] == viterbi_df_out.iloc[:,1]).sum()/len(viterbi_df_out.iloc[:,1])



## Section 1.2
## Question 6. Running the decorder on test dataset ############################

############ Reading the test data set ###########################
pos_data = '.\\'
test_data = open(pos_data+"tst.word",'r')
test_pos = []

#### Creating a list of list for each of sentences
#### Also, adding the start and the end tokens for each sentence
for sent in test_data:
    ## Splitting the sentence based on space
    words = sent.strip().split(" ")
    
    ## Storing the processed text for each sentence
    test_pos.append(words)
    

### Creating lists with sentences and tags
test_sentences = []

for test_sent in test_pos:
    ### Reading each line in the test dataset and converting the tags which are not in the Vocabulary to 'Unk'
    sent_temp = [each if each in emission_prob_df.columns else 'Unk' for each in test_sent]
    
    ## Creating list of tokens and tags
    test_sentences.append(sent_temp)


### Running the Viterbi algorithm on the test dataset
viterbi_df_out = pd.DataFrame()
f = open("mr6rx-viterbi.txt", "w+")

## Looping through each of the sentence in dev data
for sent_no, each_sent in enumerate(test_sentences):
    
    ##Calling the viterbi algorithm for the current line
    vit_out = viterbi_algorithm(transition_df,emission_df,each_sent)
    
    f.write(" ".join([word+"/"+vit_out[value+1] for value,word in enumerate(test_pos[sent_no])]) + "\n")

f.close()


## Section 1.2
## Question 7. Tuning the values of alpha and beta ############################


### Note: The below set of codes are computationally expensive 
### (takes several minutes to run for all the combinations of alpha and beta)


############## Checking different values of alpha and beta
### Getting the emission probability by handling zero probability values
set_tag_list = set(tag_list)
set_vocab_list = set(vocab_list)
N = len(set(tag_list))
V = len(set(vocab_list)) - 2



## Looping through different values of alpha values
for alpha in [1,0.5,2,3]:
    print("Alpha: ",alpha)
    emission_prob_empty =  pd.DataFrame(columns = columns_set_e, index = index_set_e)
    emission_prob_df_alpha = emission_prob_empty


    ## Subtracting the 2 from vocab_list which has <start> and <end> tokens
    ##Calculating the probabilities
    for tags in set_tag_list:
        for vocab in set_vocab_list:
            if(tags not in ['START','END'] and vocab not in ['<start>','<end>']):
                p_xt_yt = (alpha + word_tag_counter[vocab,tags])/(dict_tags[tags] +  V * alpha)
                emission_prob_df_alpha.loc[tags, vocab ] = p_xt_yt

    ## Looping through different values of beta values            
    for beta in [1,0.5,2,3]:
        print("Beta: ", beta)

        trans_prob_empty = pd.DataFrame(columns = columns_set_t, index = index_set_t)
        transition_prob_beta_df = trans_prob_empty

        for yt in set_tag_list:
            for yt_min1 in set_tag_list:
                if(yt != 'START' and yt_min1 != 'END'):
                    p_yt_yt_min1 = (beta + tag_couple_counts[yt_min1,yt])/(dict_start_tag[yt_min1] +  N * beta)

                    transition_prob_beta_df.loc[yt_min1,yt] = p_yt_yt_min1

        ### Converting the estimated probabilities into log space
        transition_df = transition_prob_beta_df.astype(float).apply(lambda x: np.log(x), axis = 1)
        emission_df = emission_prob_df_alpha.astype(float).apply(lambda x: np.log(x), axis = 1)            
    
        ## Viterbi - calling
        viterbi_df_out = pd.DataFrame()
        for sent_no, each_sent in enumerate(dev_sentences):
            vit_out = viterbi_algorithm(transition_df, emission_df,each_sent)

            current_tag = dev_tags[sent_no]
            current_tag.reverse()
            comparison_df = pd.DataFrame({'Grount_truth':current_tag,'Algorithm':list(vit_out.values())})
            viterbi_df_out = pd.concat([viterbi_df_out,comparison_df],axis = 0)


        print("Accuracy: ",100 * (viterbi_df_out.iloc[:,0] == viterbi_df_out.iloc[:,1]).sum()/len(viterbi_df_out.iloc[:,1]))


		
## Section 1.2
## Question 8. Getting the test predictions based on tuned values of alpha and beta ############################

##################### Getting the emission and transition tables based on the tuned values of alpha and beta ######
### Getting the emission probability by handling zero probability values
alpha = 0.1

## Subtracting the 2 from vocab_list which has <start> and <end> tokens
V = len(set(vocab_list)) - 2
f = open("mr6rx-eprob-smoothed.txt", "w+")

emission_prob_df_alpha = pd.DataFrame(columns = columns_set_e, index = index_set_e)
##Calculating the probabilities
for tags in set(tag_list):
    for vocab in set(vocab_list):
        if(tags not in ['START','END'] and vocab not in ['<start>','<end>']):
            p_xt_yt = (alpha + word_tag_counter[vocab,tags])/(dict_tags[tags] +  V * alpha)
            emission_prob_df_alpha.loc[tags, vocab ] = p_xt_yt
            f.write(tags+','+ vocab+','+ str(p_xt_yt)+"\n")
            
            
### Getting the transition probability by handling zero probability values
beta = 0.1
N = len(set(tag_list))
#Creating a file of 
f = open("mr6rx-tprob-smoothed.txt", "w+")


transition_prob_beta_df = pd.DataFrame(columns = columns_set_t, index = index_set_t)
for yt in set(tag_list):
    for yt_min1 in set(tag_list):
        if(yt != 'START' and yt_min1 != 'END'):
            p_yt_yt_min1 = (beta + tag_couple_counts[yt_min1,yt])/(dict_start_tag[yt_min1] +  N * beta)
            
            transition_prob_beta_df.loc[yt_min1,yt] = p_yt_yt_min1
            ## Writing out the output
            f.write(yt_min1+','+ yt+','+ str(p_yt_yt_min1)+"\n")
            
            
### Converting the estimated probabilities into log space
transition_df = transition_prob_beta_df.astype(float).apply(lambda x: np.log(x), axis = 1)
emission_df = emission_prob_df_alpha.astype(float).apply(lambda x: np.log(x), axis = 1)


### Running the Viterbi algorithm on the test dataset
viterbi_df_out = pd.DataFrame()
f = open("mr6rx-viterbi-tuned.txt", "w+")

## Looping through each of the sentence in dev data
for sent_no, each_sent in enumerate(test_sentences):
    
    ##Calling the viterbi algorithm for the current line
    vit_out = viterbi_algorithm(transition_df,emission_df,each_sent)
    
    f.write(" ".join([word+"/"+vit_out[value+1] for value,word in enumerate(test_pos[sent_no])]) + "\n")

f.close()