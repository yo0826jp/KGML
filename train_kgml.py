import os
import sys

data_path = ""
dim1 = 100
dim2 = 100
alpha = 1.0

if __name__ == "__main__":
    data_path = sys.argv[len(sys.argv)-1]
    for i in range(1,len(sys.argv)-2):
        arg = sys.argv[i]
        val = sys.argv[i+1]
        if arg == "-d1": 
            dim1 = int(val)
        if arg == "-d2": 
            dim2 = int(val)
        if arg == "-a": 
            alpha = float(val)
            
if data_path not in ["FB15k", "WN18", "WD40k"]:
    print("Please specify dataset (FB15k, WN18 or WD40k).")
    sys.exit()

print("Dataset is "+data_path)
print("dim1  = "+str(dim1)) 
print("dim2  = "+str(dim2)) 
print("alpha = "+str(alpha))
       

import time
import numpy as np
from sklearn.metrics import accuracy_score, auc
#Keras import
from keras.models import Model, Sequential
from keras.layers import Dropout, Dense, Input, Concatenate, Multiply, Embedding, Activation, Reshape
from keras import backend as K
from keras.callbacks import LearningRateScheduler
from keras.constraints import unit_norm
from keras.initializers import RandomUniform

def read_ids(path):
    ids = {}
    
    with open(path) as file:
        for line in file:
            line = line.split("\t")
            ids.update({line[0]:int(line[1])})
    print(path+"\t"+ str(len(ids)))
    return ids
            
def read_triplets(path, e_id, r_id):
    s_data = []
    o_data = []
    p_data = []
    
    with open(path) as file:
        for line in file:
            line = line.replace('\n', '').split("\t")
            s_data.append( e_id[line[0]] )
            o_data.append( e_id[line[1]] )
            p_data.append( r_id[line[2]] )
            
    print(path+"\t"+ str(len(s_data)))
    return np.array(s_data),np.array(o_data),np.array(p_data)
    
e_id = read_ids("data/"+data_path+"/entity2id.txt")
r_id = read_ids("data/"+data_path+"/relation2id.txt")
s_test,o_test,p_test = read_triplets("data/"+data_path+"/test.txt", e_id, r_id)
s_train,o_train,p_train = read_triplets("data/"+data_path+"/train.txt", e_id, r_id)
s_valid,o_valid,p_valid = read_triplets("data/"+data_path+"/valid.txt", e_id, r_id)

def combine(s_data, o_data, p_data, sources, e_id, r_id):
    so2p = {}
    sp2o = {}
    op2s = {}
    for source in sources:
        s_source = source[0]
        o_source = source[1]
        p_source = source[2]
        so2p = count_pair(s_source,o_source,p_source,so2p)
        sp2o = count_pair(s_source,p_source,o_source,sp2o)
        op2s = count_pair(o_source,p_source,s_source,op2s)
        
    s_combine = np.zeros((len(s_data),len(e_id)), dtype = bool)
    o_combine = np.zeros((len(o_data),len(e_id)), dtype = bool)
    p_combine = np.zeros((len(p_data),len(r_id)), dtype = bool)
    
    for s,o,p,i in zip(s_data,o_data,p_data,range(len(s_data))):
        s_combine[i][list(op2s[(o,p)])] = True
        o_combine[i][list(sp2o[(s,p)])] = True
        p_combine[i][list(so2p[(s,o)])] = True
        
    print(s_combine.sum()/(len(s_data)*len(e_id)))
    print(o_combine.sum()/(len(o_data)*len(e_id)))
    print(p_combine.sum()/(len(p_data)*len(r_id)))
    return s_combine,o_combine,p_combine
    
def count_pair(A,B,C,p2i):
    for a,b,c in zip(A,B,C):
        if (a,b) not in p2i:
            p2i.update({(a,b):set()})
        p2i[(a,b)].add(c)
    return p2i

s_train_comb,o_train_comb,p_train_comb = combine(s_train,o_train,p_train,[[s_train,o_train,p_train]], e_id, r_id)
s_test_comb,o_test_comb,p_test_comb = combine(s_test,o_test,p_test,[[s_train,o_train,p_train],[s_test,o_test,p_test],[s_valid,o_valid,p_valid]], e_id, r_id)


ent_voc = len(e_id)
rel_voc = len(r_id)

def dynamic_weighted_binary_crossentropy(l,alpha=0.5):
    
    def loss(y_true, y_pred):
        w_neg = K.sum(y_true) / l
        w_pos = 1 - w_neg
        r = 2*w_neg*w_pos
        w_neg /= r
        w_pos /= r
        
        b_ce = K.binary_crossentropy(y_true, y_pred)
        w_b_ce = b_ce *y_true* w_pos + b_ce * (1-y_true) * w_neg 
        return K.mean(w_b_ce) * alpha + K.mean(b_ce) * (1-alpha)
    
    return loss

def P(y_true, y_pred):
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.20), 'float32'))
    pred_positives = K.sum(K.cast(K.greater(K.clip(y_pred, 0, 1), 0.50), 'float32'))

    precision = true_positives / (pred_positives + K.epsilon())
    return precision

def R(y_true, y_pred):
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.20), 'float32'))
    poss_positives = K.sum(K.cast(K.greater(K.clip(y_true, 0, 1), 0.50), 'float32'))

    recall = true_positives / (poss_positives + K.epsilon())
    return recall
    
def learning_rate(epoch):
    if epoch > 40:   return 0.00001
    elif epoch > 30: return 0.000033
    elif epoch > 20: return 0.0001
    elif epoch > 10: return 0.00033
    return 0.001
    
sample_weight = np.zeros((len(s_train),))
so2p = count_pair(s_train,o_train,p_train,{})
for s,o,i in zip(s_train, o_train,range(len(s_train))):
    sample_weight[i] = 1/ len(so2p[(s,o)])
sample_weight /= sample_weight.sum()/len(s_train)
del so2p

S_input = Input(shape=(1,), dtype="int32", name="S_input")
O_input = Input(shape=(1,), dtype="int32", name="O_input")

embed_layer = Embedding(ent_voc, dim1)
S_embed = embed_layer(S_input)
O_embed = embed_layer(O_input)


reshape_layer = Reshape((dim1,), input_shape=(1,dim1))
S_reshape = reshape_layer(S_embed)
O_reshape = reshape_layer(O_embed)

SO_merged = Multiply()([S_reshape, O_reshape])
SO_merged = Concatenate()([S_reshape, O_reshape,SO_merged])

SO_merged = Dropout(0.3)(SO_merged)
SO_merged = Dense(int(dim2*1.5), activation="relu")(SO_merged)
SO_merged = Dropout(0.1)(SO_merged)
SO_merged = Dense(int(dim2*1.5/2), activation="relu")(SO_merged)
SO_merged = Dropout(0.1)(SO_merged)

P_pred = Dense(rel_voc, activation="sigmoid")(SO_merged)

model = Model([S_input, O_input], P_pred)

lr_cb = LearningRateScheduler(learning_rate)
model.compile(optimizer="adam",
              loss=dynamic_weighted_binary_crossentropy(rel_voc,alpha=alpha),
              metrics=["binary_accuracy",P,R])
              
model.fit([s_train, o_train], p_train_comb, 
            epochs=50,
          sample_weight=sample_weight,
         callbacks=[lr_cb])
         
model.save("kgml_"+data_path+".h5")
