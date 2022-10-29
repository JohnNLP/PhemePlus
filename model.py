import os
import pickle
import torch
import torch.nn as nn
import random
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
from pytorch_transformers import AdamW
from scipy.special import softmax
import numpy as np
import json

#configurables
EPOCHS = 25
MAX_LEN = 35
N_EVIDENCE_TO_USE = 1

padding_token = "<pad>"
model_name = "roberta-base" #"albert-large-v2"  #1/4
config = RobertaConfig.from_pretrained(model_name)  #2/4
##config.update({"add_pooling_layer": True})
tokenizer = RobertaTokenizer.from_pretrained(model_name)  #3/4


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare(text):
    out = " ".join([i.lower() for i in text])
    out = tokenizer.encode(out,max_length=MAX_LEN, truncation=True) #with new models, make sure [BOS] and [EOS] tokens are being added
    if len(out) < MAX_LEN:
        pad = tokenizer.encode(padding_token, add_special_tokens=False) #coref is <pad>
        if len(pad) != 1:
            print("Wrong padding token used!")
            print(pad)
            exit()
        for i in range(MAX_LEN-len(out)):
            out += pad
    return out

#todo: is [40(sent - eos - pad) - 40(sent2 - eos - pad)] correct/optimal?
#todo: alternately try max40(sent - eos) - max40(sent - eos) - pad up to 80
def prepareJoint(text, text2):
    out = " ".join([i for i in text])
    out = tokenizer.encode(out,max_length=MAX_LEN, truncation=True)
    out2 = " ".join([i for i in text2])
    out2 = tokenizer.encode(out2,max_length=MAX_LEN+1, truncation=True)
    out2 = out2[1:] #remove bos token (is now the correct length also)
    out += out2
    pad = tokenizer.encode(padding_token, add_special_tokens=False)
    if len(pad) != 1:
        print("Wrong padding token used!")
        print(pad)
        exit()
    if len(out) < MAX_LEN*2:
        for i in range(MAX_LEN*2-len(out)):
            out += pad
    return out

def numberiseAnnotation(annotation):
    if annotation == "false" or annotation == "FALSE":
        return 0
    elif annotation == "true" or annotation == "TRUE":
        return 1
    elif annotation == "unverified" or annotation == "UNVERIFIED":
        return 2
    else:
        print("Unexpected annotation found")
        exit()

def stringifyAnnotation(annotation):
    if annotation == 0:
        return "TRUE"
    elif annotation == 1:
        return "FALSE"
    elif annotation == 2:
        return "UNVERIFIED"
    else:
        print("Unexpected annotation found")
        exit()

def makeTrainable(dict):
    x = []
    y = []
    tags = []
    sentences = []
    rumours = []
    dump = []

    for r in dict:
        if len(dict[r]["evidence"]) != 5:
            #print("Evidence is missing for rumour", r)
            continue
        for v, evidence in enumerate(dict[r]["evidence"]):
            if v >= N_EVIDENCE_TO_USE:
                break
            x.append(prepareJoint(dict[r]["rumour"], evidence)) #todo: try different orderings of this!
            #x.append(prepare(evidence))
            #x.append(prepare(dict[r]["rumour"]))
            y.append(numberiseAnnotation(dict[r]["annotation"]))
            tags.append(r)
            sentences.append(evidence)
            rumours.append(dict[r]["rumour"])
            dump.append(["ID", r, "EVENT", dict[r]["event"], "RUM", dict[r]["rumour"], "EVI", evidence, "VALUE", dict[r]["annotation"], "PRED", None])
    return x, y, tags, sentences, rumours, dump

def printMatrix(labels, pred):
    x = confusion_matrix(labels, pred)
    print("      Predicted:")
    print("          F  T U")
    print("Actual: F", x[0][0], x[0][1], x[0][2])
    print("        T", x[1][0], x[1][1], x[1][2])
    print("        U", x[2][0], x[2][1], x[2][2])
    print("Macro F1:", round(f1_score(labels, pred, average="macro"), 4))
    a, b, c = f1_score(labels, pred, average=None)
    print("False F1:", round(a, 4))
    print("True F1:", round(b, 4))
    print("Unverified F1:", round(c, 4))

def printData(labels, pred):
    for v, i in enumerate(pred):
        print("\nr:", rumours[v])
        print("s:", sentences[v])
        print("p:", stringifyAnnotation(i), "a:", stringifyAnnotation(labels[v]))

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.bertFlavour = RobertaModel(config)
        self.drop = nn.Dropout(0.1)
        self.linear = nn.Linear(768,3)

    def forward(self, data, attention_mask):
        out = self.bertFlavour(data, attention_mask=attention_mask)

        out = out["pooler_output"]

        out = self.drop(out)
        out = self.linear(out)
        return out

#last steps
with open("save.pkl", 'rb') as f:
    all_data = pickle.load(f)

test_y_all = []
pred_actual_all = []
preds = []

events = ["charliehebdo-all-rnr-threads","ferguson-all-rnr-threads","germanwings-crash-all-rnr-threads","ottawashooting-all-rnr-threads","sydneysiege-all-rnr-threads"]

for event in events:
    print("\nNEXT:",event)
    test_data = {}
    train_data = {}

    for v, i in enumerate(all_data):
        if all_data[i]["event"] == event:
            test_data[i] = all_data[i]
        else:
            train_data[i] = all_data[i]

    train_x, train_y, _, _, _, _ = makeTrainable(train_data)
    train_x_mask = [[int(j != 1) for j in i] for i in train_x]
    test_x, test_y, test_tags, sentences, rumours, data_dump = makeTrainable(test_data)
    test_x_mask = [[int(j != 1) for j in i] for i in test_x]

    #shuffle
    temp = list(zip(train_x, train_x_mask, train_y))
    random.shuffle(temp)
    train_x, train_x_mask, train_y = zip(*temp)

    #account for class imbalance
    a = train_y.count(0)
    b = train_y.count(1)
    c = train_y.count(2)
    weights = torch.tensor([1/a, 1/b, 1/c], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = CustomModel().to(device)
    optimizer = AdamW(model.parameters(), lr=3e-5, correct_bias=True) #3e-5 for base, 9e-6? for large

    #make tensors and move them to gpu
    train_x = torch.tensor(train_x).to(device)
    train_x_mask = torch.tensor(train_x_mask).to(device)
    test_x = torch.tensor(test_x).to(device)
    test_x_mask = torch.tensor(test_x_mask).to(device)
    train_y = torch.tensor(train_y).to(device)

    #train model
    BATCH_SIZE = 20
    train_x = torch.split(train_x, BATCH_SIZE)
    train_x_mask = torch.split(train_x_mask, BATCH_SIZE)
    train_y = torch.split(train_y, BATCH_SIZE)

    test_x = torch.split(test_x, 200)
    test_x_mask = torch.split(test_x_mask, 200)

    for epoch in range(EPOCHS):
        model.train()
        loss_readout = 0
        for v, i in enumerate(train_x):
            y_pred = model(train_x[v], attention_mask=train_x_mask[v])
            loss = criterion(y_pred, train_y[v])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_readout += loss.item()
        print("EPOCH:", epoch + 1, "loss:", round(loss_readout, 4))

        #get predictions
        model.eval()
        pred = []
        for v, i in enumerate(test_x):
            with torch.no_grad():
                output = model(i, attention_mask=test_x_mask[v])
            predicted = output.cpu().numpy()
            pred += [j for j in predicted]

        pred = np.array([softmax(i) for i in pred])
        pred_actual = []
        for i in pred:
            if i[0] >= i[1] and i[0] >= i[2]:
                pred_actual.append(0)
            elif i[1] >= i[0] and i[1] >= i[2]:
                pred_actual.append(1)
            else:
                pred_actual.append(2)

        #log data here
        for v, i in enumerate(data_dump):
            data_dump[v][11] = pred_actual[v]

        #printData(test_y, pred_actual)
        print("\n"+event)
        printMatrix(test_y, pred_actual)

        if epoch == EPOCHS-1:
            test_y_all += test_y
            pred_actual_all += pred_actual
            preds += data_dump

#final evaluation
print("\nOVERALL RESULTS")
printMatrix(test_y_all, pred_actual_all)

with open('data.json', 'w') as f:
    json.dump(preds, f)