import torch
import torch.nn.functional as F
import torch.nn as nn
def train(model,graph,features,labels,mask,optimizer):
    model.train()
    logits = model(graph,features)
    loss = F.cross_entropy(logits[mask],labels[mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    #print('train loss is',loss.item())
    return loss.item()
def evaluate(model,graph,features,labels,mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph,features)
        logits = logits[mask]
        labels = labels[mask]
        _,indices = torch.max(logits,dim=1)
        correct = torch.sum(indices==labels)
        #print(correct.item()*1.0/len(labels))
    return correct.item()*1.0/len(labels)


