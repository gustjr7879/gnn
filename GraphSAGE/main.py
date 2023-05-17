from model import *
from ttv import *
import dgl
import dgl.data as Data
import torch
#cora dataset ipmort
def set_seed(seed):
    torch.manual_seed(seed)
seed = 42
set_seed(seed)

dataset = Data.CoraGraphDataset(raw_dir='')
graph = dataset[0]

train_mask = graph.ndata['train_mask']
valid_mask = graph.ndata['val_mask']
test_mask = graph.ndata['test_mask']

features = graph.ndata['feat']
labels = graph.ndata['label']

n_features = features.shape[1]
n_labels = int(max(labels))+1

features = torch.tensor(features)

# 배운점 : label은 torch로 안보내도된다
model = SAGE(n_features,16,n_labels)
optim = torch.optim.Adam(model.parameters(),lr=0.01)

for i in range(200):
    loss = train(model,graph,features,labels,train_mask,optimizer=optim)
    acc = evaluate(model,graph,features,labels,valid_mask)
    if i%50 ==0:
        print('train loss and valid acc',loss,acc)
test_acc = evaluate(model,graph,features,labels,test_mask)
print('test acc is',test_acc)