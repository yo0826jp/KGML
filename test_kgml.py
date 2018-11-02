from keras.models import load_model
from keras import backend as K

import sys
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

model_path = sys.argv[1]
if len(sys.argv) >= 3:
    data_path = sys.argv[2]
else:
    data_path = "FB15k"
print(model_path)

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

def combine(x_data, y_data, z_data, x_source, y_source,z_source, z_voc):
    xy2z = count_pair(x_source,y_source,z_source)
        
    z_combine = np.zeros((len(x_data),z_voc), dtype = bool)
    
    for x,y,z,i in zip(x_data,y_data,z_data,range(len(x_data))):
        z_combine[i][list(xy2z[(x,y)])] = True
        
    print(z_combine.sum()/(z_combine.shape[0]*z_combine.shape[1]))
    return z_combine
    
def count_pair(x_data, y_data, z_data):
    xy2z = {}
    for x,y,z in zip(x_data, y_data, z_data):
        if (x,y) not in xy2z:
            xy2z.update({(x,y):set()})
        xy2z[(x,y)].add(z)
    return xy2z
    
e_id = read_ids("../data/"+data_path+"/entity2id.txt")
r_id = read_ids("../data/"+data_path+"/relation2id.txt")
s_test,o_test,p_test = read_triplets("../data/"+data_path+"/test.txt", e_id, r_id)
s_train,o_train,p_train = read_triplets("../data/"+data_path+"/train.txt", e_id, r_id)
s_valid,o_valid,p_valid = read_triplets("../data/"+data_path+"/valid.txt", e_id, r_id)
s_all = np.concatenate((s_test,s_train,s_valid))
o_all = np.concatenate((o_test,o_train,o_valid))
p_all = np.concatenate((p_test,p_train,p_valid))

ent_voc = len(e_id)
rel_voc = len(r_id)
p_test_comb = combine(s_test,o_test,p_test,s_all,o_all,p_all, rel_voc)

def dynamic_weighted_binary_crossentropy(l):
    
    def loss(y_true, y_pred):
        w_neg = K.sum(y_true) / l
        w_pos = 1 - w_neg
        r = 2*w_neg*w_pos
        w_neg /= K.sqrt(r)
        w_pos /= K.sqrt(r)
        
        b_ce = K.binary_crossentropy(y_true, y_pred)
        w_b_ce = b_ce *y_true* w_pos + b_ce * (1-y_true) * w_neg 
        return K.mean(w_b_ce)
    
    return loss

def P(y_true, y_pred):
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.20), 'float32'))
    pred_positives = K.sum(K.cast(K.greater(K.clip(y_pred, 0, 1), 0.20), 'float32'))

    precision = true_positives / (pred_positives + K.epsilon())
    return precision

def R(y_true, y_pred):
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.20), 'float32'))
    poss_positives = K.sum(K.cast(K.greater(K.clip(y_true, 0, 1), 0.20), 'float32'))

    recall = true_positives / (poss_positives + K.epsilon())
    return recall
my_functions = {
    "loss": dynamic_weighted_binary_crossentropy(len(r_id)),
    "P": P,
    "R": R
}
model = load_model(model_path,custom_objects=my_functions)
p_preds = model.predict([s_test, o_test])

def hits(inputs1, inputs2, preds, trues, k = 1, th = 0.5):
    i2p = count_pair(s_all,o_all,p_all)
    hits_k_raw = 0.
    hits_k_filt = 0.
    hits_pos = 0.
    
    for i1, i2, pred, true in zip(inputs1, inputs2, preds, trues):
        for p in np.argsort(pred)[::-1][:k]:
            if p == true:
                hits_k_raw += 1
                break
        
        count = 0
        for p in np.argsort(pred)[::-1]:
            if p == true:
                hits_k_filt += 1
                break
            if (i1,i2) in i2p:
                if p in i2p[(i1,i2)]:
                    continue
            count += 1
            if count >= k:
                break
            
        if pred[true] >= th:
            hits_pos += 1
            
    hits_k_raw /= len(inputs1)
    hits_k_filt /= len(inputs1)
    hits_pos /= len(inputs1)
    
    print("hits@"+str(k)+" raw:\t" + str(hits_k_raw))
    print("hits@"+str(k)+" filt:\t" + str(hits_k_filt))
    print("hits@positive:\t" + str(hits_pos))
    
hits(s_test,o_test,p_preds,p_test)
hits(s_test,o_test,p_preds,p_test,k=10)

def ROC(preds, trues, th = 0.5):
    acc = 0.
    auc = 0.
    precision = 0.
    recall = 0.
    f1 = 0.
    
    count = 0
    
    for pred, true in zip(preds, trues):
        count += 1
        auc += roc_auc_score(true, pred)
    
        pred = pred > th
        if pred.sum() == 0:
            continue
            
        acc += accuracy_score(true,pred)
        pre = precision_score(true, pred, average='binary')  
        rec = recall_score(true, pred, average='binary') 
        precision += pre
        recall += rec
        if (pre+rec) > 0:
            f1 += 2*pre*rec /(pre+rec)
    
    acc /= count
    auc /= count
    precision /= count
    recall /= count
    f1 /= count
    
    print("Accuracy:\t" + str(acc))
    print("AUC:     \t" + str(auc))
    print("P:       \t" + str(precision))
    print("R:       \t" + str(recall))
    print("F1:     \t" + str(f1))

ROC(p_preds,p_test_comb )    
