# -*- coding: utf-8 -*-
"""
Created on Mon May 18 15:57:18 2020

@author: Home
"""

import copy
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

rand = 0
max_epochs = 10000
LEARNING_RATE = 1e-4
n_batches = 10

data = pd.read_csv('')
y_values = data['y']
x_values = data.drop(['y'], axis=1)

x_train_p, x_test_p, y_train_p, y_test_p = train_test_split(x_values, y_values, 
                                                            test_size=0.2, 
                                                            shuffle=False, 
                                                            random_state=rand)

x_train = copy.deepcopy(x_train_p)
x_test = copy.deepcopy(x_test_p)
y_train = copy.deepcopy(y_train_p)
y_test = copy.deepcopy(y_test_p)
x_train.reset_index(drop=True, inplace=True)
x_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
 
        # encoder
        self.enc1 = nn.Linear(in_features=20, out_features=14)
        self.enc2 = nn.Linear(in_features=14, out_features=9)
        self.enc3 = nn.Linear(in_features=9, out_features=5)
 
        # decoder 
        self.dec1 = nn.Linear(in_features=5, out_features=9)
        self.dec2 = nn.Linear(in_features=9, out_features=14)
        self.dec3 = nn.Linear(in_features=14, out_features=20)
 
    def forward(self, initial):
        x_enc = torch.relu(self.enc1(initial))
        x_enc = torch.relu(self.enc2(x_enc))
        x_enc = torch.relu(self.enc3(x_enc))
 
        x = torch.relu(self.dec1(x_enc))
        x = torch.relu(self.dec2(x))
        x = torch.relu(self.dec3(x))
        return x, x_enc

def train(net, trainloader, max_epochs):
    train_loss = []
    for epoch in range(max_epochs):
        for i in range(n_batches):
            # Local batches and labels
            local_x = torch.from_numpy(x_train[i*n_batches:(i+1)*n_batches].values).float()
            running_loss = 0.0
            outputs, encoded_output = net(local_x)
            loss = criterion(outputs, local_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        loss = running_loss / len(trainloader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch+1, max_epochs, loss))
    
        #if epoch % 5 == 0:
        #save_decoded_image(outputs.cpu().data, epoch)

    return train_loss

net = Autoencoder()

criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

# get the computation device
device = get_device()
print(device)
# load the neural network onto the device
net.to(device)
 
# train the network
train_loss = train(net, x_train, max_epochs)
plt.figure()
plt.plot(train_loss)
plt.title('Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('deep_ae_fashionmnist_loss.png')

outp, encoded_train_data = net.forward(torch.from_numpy(x_train.values).float())
enc_data_np = pd.DataFrame(encoded_train_data.detach().numpy())
enc_data_np['y'] = y_train

outp, encoded_test_data = net.forward(torch.from_numpy(x_test.values).float())
enc_data_test_np = pd.DataFrame(encoded_test_data.detach().numpy())
enc_data_test_np['y'] = y_test

encoded_data = enc_data_np.append(enc_data_test_np)






import copy
import lime
from matplotlib import pyplot as plt
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

n_estimators=1000
bootstrap=False
shap_use = False
shap_explained = 0
lime_use = False
lime_explained = 12

data = encoded_data
y_values = data['y']
x_values = data.drop(['y'], axis=1)

x_train_p, x_test_p, y_train_p, y_test_p = train_test_split(x_values, y_values, 
                                                            test_size=0.8, 
                                                            shuffle=True, 
                                                            random_state=rand)

x_train = copy.deepcopy(x_train_p)
x_test = copy.deepcopy(x_test_p)
y_train = copy.deepcopy(y_train_p)
y_test = copy.deepcopy(y_test_p)
x_train.reset_index(drop=True, inplace=True)
x_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

random_forest = RandomForestClassifier(n_estimators=n_estimators,
                                       bootstrap=bootstrap,
                                       n_jobs=2,
                                       random_state=rand,
                                       verbose=0)
random_forest.fit(x_train, y_train)
y_pred_from_train = random_forest.predict(x_train)
y_pred = random_forest.predict(x_test)
print(random_forest.feature_importances_)

if shap_use:
    shap_explainer = shap.TreeExplainer(random_forest)
    print('shap_explainer done')
    shap_values = shap_explainer.shap_values(x_train)
    print('shap_values')
    print(shap_values)
    shap.summary_plot(shap_values, x_train, plot_type='bar')

if lime_use:
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(x_train.to_numpy(), feature_names=x_train.columns.to_list(), 
                                                            class_names=['y'])
    print('lime_explainer done')
    lime_explanation = lime_explainer.explain_instance(x_train.to_numpy()[lime_explained], random_forest.predict_proba)    
    print(lime_explanation.as_list())
    lime_aggreg_expl = x_train.apply(lime_explainer.explain_instance, 
                                     predict_fn=random_forest.predict_proba)
    print(lime_aggreg_expl.as_list())

conf_matrix = confusion_matrix(y_test, y_pred)

correct = 0
for i in range(len(conf_matrix)):
    correct += conf_matrix[i][i]

wrong = 0
for i in range(len(conf_matrix)):
    for j in range(len(conf_matrix[i])):
        if i != j:
            wrong += conf_matrix[i][j]

perc_correct = correct / (correct + wrong)

print(conf_matrix)
print("correct {}".format(correct))
print("wrong {}".format(wrong))
print("perc_correct {}".format(perc_correct))



























