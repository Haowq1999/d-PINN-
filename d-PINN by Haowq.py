import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter
import sklearn.model_selection
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m
import os
import warnings
#####################################################################################

warnings.filterwarnings("ignore")    # Ignore all warnings
path = r'Your path'    # (Please switch to your work path)
os.chdir(path)

# ======================================Precision Calculation Function======================================
def accuracy(y, y_pred):
    MSE = sklearn.metrics.mean_squared_error(y, y_pred)
    MAPE = sklearn.metrics.mean_absolute_percentage_error(y, y_pred)
    SD = 0
    for row in range(len(y)):
        SD += (m.fabs((y[row] - y_pred[row]) / y[row]) - MAPE)**2
    SD = m.sqrt(SD / (len(y) - 1))
    R2 = sklearn.metrics.r2_score(y, y_pred)
    return [MSE, MAPE, SD, R2]

# ======================================Input Data Conversion Functions======================================
def data_tran_to_tensor(data):
    if(len(np.shape(data)) == 1):
        data = torch.unsqueeze(torch.tensor(data), dim=1)
    else:
        data = torch.tensor(data)
    return Variable(data.to(torch.float32))

# ======================================Data Inverse Normalization and Damage Segmentation Functions======================================
def data_innom_t(t_bat):
    t_innom = []
    for t in t_bat:
        t_innom.append(torch.log10(10**(t)/3600))
    return t_innom
def data_splt(X):
    X_backup = np.zeros((np.shape(X)[0], np.shape(X)[1]-1))
    for i in range(np.shape(X)[0]):
        X_backup[i] = np.delete(X[i], [0])
    X_backup = data_tran_to_tensor(X_backup)
    X_c = X_backup[:,0:5]
    X_f = X_backup[:,5:11]
    X_o = X_backup[:,11:16]
    return [X_c, X_f, X_o]

# ======================================Entering data and converting data formats======================================
X = data_tran_to_tensor(pd.read_excel('Dimensionless data.xlsx', sheet_name='Train').values[:,1:18])
X_origin = data_tran_to_tensor(pd.read_excel('Dimensionless data.xlsx', sheet_name='Train').values[:,1:18])
load_type = data_tran_to_tensor(pd.read_excel('Dimensionless data.xlsx', sheet_name='Train').values[:,1])    # 1: PC; 2: PF; 3:CFI; 4 TMF
t_cf_true = data_tran_to_tensor(pd.read_excel('Dimensionless data.xlsx', sheet_name='Train').values[:,18])
X_new = data_tran_to_tensor(pd.read_excel('Dimensionless data.xlsx', sheet_name='Predict').values[:,1:18])
X_new_origin = data_tran_to_tensor(pd.read_excel('Dimensionless data.xlsx', sheet_name='Predict').values[:,1:18])
t_CFI_TMF_true = data_tran_to_tensor(pd.read_excel('Dimensionless data.xlsx', sheet_name='Predict').values[:,18])
N_cf_true = data_tran_to_tensor(pd.read_excel('Dimensionless data.xlsx', sheet_name='Train').values[:,20])
N_cf_pred = torch.zeros_like(N_cf_true)
N_CFI_TMF_true = data_tran_to_tensor(pd.read_excel('Dimensionless data.xlsx', sheet_name='Predict').values[:,20])
N_CFI_TMF_pred = torch.zeros_like(N_CFI_TMF_true)
E_cf = data_tran_to_tensor(pd.read_excel('Dimensionless data.xlsx', sheet_name='Train').values[:,21])
E_CFI_TMF = data_tran_to_tensor(pd.read_excel('Dimensionless data.xlsx', sheet_name='Predict').values[:,21])
# print(t_CFI_TMF_true)
# print(t_cf_true, '\n', t_CFI_TMF_true)

# ======================================d-PINN network structure and forward propagation======================================
class PINN(nn.Module):
    # Declare the layers of the network with model parameters
    def __init__(self):
        # The constructor of parent class Module is called to perform the necessary initialization
        super().__init__()
        # Creep damage neural network initialization
        self.hidden_dc_1 = nn.Linear(5, 16)
        self.hidden_dc_2 = nn.Linear(16, 16)
        self.hidden_dc_3 = nn.Linear(16, 8)
        self.out_dc = nn.Linear(8, 1)
        # Fatigue damage neural network initialization
        self.hidden_df_1 = nn.Linear(6, 16)
        self.hidden_df_2 = nn.Linear(16, 16)
        self.hidden_df_3 = nn.Linear(16, 8)
        self.out_df = nn.Linear(8, 1)
        # Oxidation damage neural network initialization
        self.hidden_do_1 = nn.Linear(5, 16)
        self.hidden_do_2 = nn.Linear(16, 16)
        self.hidden_do_3 = nn.Linear(16, 8)
        self.out_do = nn.Linear(8, 1)
        # Liftime prediction neural network initialization
        self.hidden_tcf_1 = nn.Linear(3, 8)
        self.hidden_tcf_2 = nn.Linear(8, 8)
        self.hidden_tcf_3 = nn.Linear(8, 8)
        self.out_tcf = nn.Linear(8, 1)
    # Define the weight initialization of the model
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    # Define the forward propagation of the model
    def forward(self, X):
        data_type = X[:, 0]
        [X_c, X_f, X_o] = data_splt(X)
        X_c.requires_grad_()
        X_f.requires_grad_()
        X_o.requires_grad_()
        # Creep damage neural network
        x_c = F.relu(self.hidden_dc_1(X_c))
        x_c = F.relu(self.hidden_dc_2(x_c))
        x_c = F.relu(self.hidden_dc_3(x_c))
        D_c = self.out_dc(x_c)
        # Fatigue damage neural network
        x_f = F.relu(self.hidden_df_1(X_f))
        x_f = F.relu(self.hidden_df_2(x_f))
        x_f = F.relu(self.hidden_df_3(x_f))
        D_f = self.out_df(x_f)
        # Oxidation damage neural network
        x_o = F.relu(self.hidden_do_1(X_o))
        x_o = F.relu(self.hidden_do_2(x_o))
        x_o = F.relu(self.hidden_do_3(x_o))
        D_o = self.out_do(x_o)
        # Liftime prediction neural network
        for ttt in range(len(data_type)):
            if(data_type[ttt] == 1):
                D_f[ttt] = 0
            if(data_type[ttt] == 2):
                D_c[ttt] = 0
            else:
                continue
        X_cfo = torch.cat((D_c, D_f, D_o), dim=1)
        # print(X_cfo)
        x_cfo = F.relu(self.hidden_tcf_1(X_cfo))
        x_cfo = F.relu(self.hidden_tcf_2(x_cfo))
        x_cfo = F.relu(self.hidden_tcf_3(x_cfo))
        t_cf = self.out_tcf(x_cfo)
        return [D_c, D_f, D_o, t_cf, X_c, X_f, X_o]

# ======================================Neural Network Model Functions for Training, Validation, Testing and Prediction======================================
def nn_model(X_train, t_train, learning_rate, weight_decay, batch_size, num_epochs, lambda_phy):
    writer = SummaryWriter('logs')
    writer_loss = pd.ExcelWriter('Result_PINN_loss.xlsx')
    loss_matrix = []
    loss_matrix.append(['Epoch', 'Data', 'DF', 'DS', 'tD', 'DC', 'Df', 'Do'])
    CF_PINN = PINN()
    # CF_PINN = torch.load('PINN_wq.pth')
    optimizer = torch.optim.Adam(CF_PINN.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_func = torch.nn.MSELoss()
    train_dataset = torch.utils.data.TensorDataset(X_train, t_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        loss_sum = 0
        loss_num = 0
        for inputs, labels in train_dataloader:
            D_c, D_f, D_o, t_cf, X_c, X_f, X_o = CF_PINN(inputs)    # Forward propagation
            D_cfo_sum = D_c + D_f + D_o
            loss_data = loss_func(t_cf, labels)    # Data loss
            loss_phy_damage = (F.relu(-D_c) + F.relu(D_c-1) + F.relu(-D_f) + F.relu(D_f-1) + F.relu(-D_o) + F.relu(D_o-1)).mean()    # Physical constraint 1
            loss_phy_damage_sum = (F.relu(-D_cfo_sum) + F.relu(D_cfo_sum-1)).mean()    # Physical constraint 1
            grad_outputs = torch.ones_like(t_cf)
            d_t_d_Dc = torch.autograd.grad(t_cf, D_c, grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]
            d_t_d_Df = torch.autograd.grad(t_cf, D_f, grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]
            d_t_d_Do = torch.autograd.grad(t_cf, D_o, grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]
            loss_phy_td = (F.relu(d_t_d_Dc) + F.relu(d_t_d_Df) + F.relu(d_t_d_Do)).mean()    # Physical constraint 3
            grad_outputs = torch.ones_like(D_c)
            d_Dc_d_Xc = torch.autograd.grad(D_c, X_c, grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]
            loss_phy_dxc = (F.relu(-d_Dc_d_Xc[:, 0]) + F.relu(-d_Dc_d_Xc[:, 1]) + F.relu(-d_Dc_d_Xc[:, 3]) + F.relu(-d_Dc_d_Xc[:, 4])).mean()    # Physical constraint 2
            grad_outputs = torch.ones_like(D_f)
            d_Df_d_Xf = torch.autograd.grad(D_f, X_f, grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]
            loss_phy_dxf = (F.relu(-d_Df_d_Xf[:, 3]) + F.relu(-d_Df_d_Xf[:, 4])).mean()    # Physical constraint 2
            grad_outputs = torch.ones_like(D_o)
            d_Do_d_Xo = torch.autograd.grad(D_o, X_o, grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]
            loss_phy_dxo = (F.relu(-d_Do_d_Xo[:, 0]) + F.relu(-d_Do_d_Xo[:, 2]) + F.relu(-d_Do_d_Xo[:, 3]) + F.relu(-d_Do_d_Xo[:, 4])).mean()    # Physical constraint 2
            loss = loss_data + lambda_phy*(loss_phy_damage + loss_phy_damage_sum + loss_phy_td + loss_phy_dxc + loss_phy_dxf + loss_phy_dxo)
            optimizer.zero_grad()    # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            loss_sum += loss
            loss_num += 1
        if((epoch+1) % 100 == 0):
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {(loss_sum/loss_num).item()}")
            print(f"Data: {format(loss_data.item(), '.4f')}, DF: {format(loss_phy_damage.item(), '.4f')}, DS: {format(loss_phy_damage_sum.item(), '.4f')}"
                + f", tD: {format(loss_phy_td.item(), '.4f')}, Dc: {format(loss_phy_dxc.item(), '.4f')}, Df: {format(loss_phy_dxf.item(), '.4f')}, Do: {format(loss_phy_dxo.item(), '.4f')}")
            # torch.save(CF_PINN, 'PINN%d.pth' %(epoch+1))    # Regular preservation of the network
        writer.add_scalar('Loss', loss, epoch)
        loss_matrix.append([epoch+1, loss_data.detach().numpy(), loss_phy_damage.detach().numpy(), loss_phy_damage_sum.detach().numpy(), 
            loss_phy_td.detach().numpy(), loss_phy_dxc.detach().numpy(), loss_phy_dxf.detach().numpy(), loss_phy_dxo.detach().numpy()])
    pd.DataFrame(loss_matrix).to_excel(writer_loss, sheet_name='Loss')
    pd.DataFrame([['lr', 'wd', 'bs', 'ne', 'lp'], [learning_rate, weight_decay, batch_size, num_epochs, lambda_phy]]).to_excel(writer_loss, sheet_name='HP')
    writer.close()
    writer_loss.close()
    return CF_PINN

# ======================================Entering data and slicing it======================================
X_train, X_test, t_train, t_test = sklearn.model_selection.train_test_split(X, t_cf_true, 
    test_size=0.3, stratify=load_type, shuffle=True, random_state=2023)
X_train_origin, X_test_origin, N_cf_train_true, N_cf_test_true = sklearn.model_selection.train_test_split(X_origin, N_cf_true, 
    test_size=0.3, stratify=load_type, shuffle=True, random_state=2023)
E_cf_train, E_cf_test = sklearn.model_selection.train_test_split(E_cf, 
    test_size=0.3, stratify=load_type, shuffle=True, random_state=2023)
N_cf_train_pred = torch.zeros_like(N_cf_train_true)
N_cf_test_pred = torch.zeros_like(N_cf_test_true)
# # ======================================Training and preservation of neural networks======================================
# model = nn_model(X_train, t_train, learning_rate=0.005, weight_decay=0.0005, batch_size=200, num_epochs=2500, lambda_phy=0.2)
# torch.save(model, 'd-PINN-Training.pth')
# ======================================Extracting the trained network to predict======================================
model = torch.load('d-PINN-Instance.pth')
# =====================================lifetime generalizable prediction======================================
t_cf_pred = model(X)[3]
t_cf_train_pred = model(X_train)[3]
t_cf_test_pred = model(X_test)[3]
t_CFI_TMF_pred = model(X_new)[3]
[t_cf_true, t_cf_pred, t_train, t_cf_train_pred, t_test, t_cf_test_pred, t_CFI_TMF_true, t_CFI_TMF_pred] = data_innom_t([t_cf_true, 
    t_cf_pred, t_train, t_cf_train_pred, t_test, t_cf_test_pred, t_CFI_TMF_true, t_CFI_TMF_pred])
# print(X_new_origin[0][5], X_new_origin[0][6], X_new_origin[0][14], X_new_origin[0][15])
# print(t_CFI_TMF_pred)
New_CFI_ext = 0
New_TMF_ext = 0
[t_c_ture, t_c_pred, t_c_train_ture, t_c_train_pred, t_c_test_ture, t_c_test_pred] = [torch.zeros(87, 1), torch.zeros(87, 1), torch.zeros(60, 1), torch.zeros(60, 1), torch.zeros(27, 1), torch.zeros(27, 1)]
[N_f_ture, N_f_pred, N_f_train_ture, N_f_train_pred, N_f_test_ture, N_f_test_pred] = [torch.zeros(90, 1), torch.zeros(90, 1), torch.zeros(63, 1), torch.zeros(63, 1), torch.zeros(27, 1), torch.zeros(27, 1)]
[N_CFI_true, N_CFI_pred] = [torch.zeros(124+New_CFI_ext, 1), torch.zeros(124+New_CFI_ext, 1)]
[N_TMF_true, N_TMF_pred] = [torch.zeros(14+New_TMF_ext, 1), torch.zeros(14+New_TMF_ext, 1)]
# ======================================Post-processing of various types of data======================================
num_c = 0
num_f = 0
for i in range(t_cf_pred.shape[0]): # Cycle life calculation
    if(X_origin[i][0] == 2):    # PF
        N_cf_pred[i][0] = 10**t_cf_pred[i][0] * 3600
        N_cf_pred[i][0] /= (E_cf[i][0] * 2 / 0.5)
        N_cf_pred[i][0] = torch.log10(N_cf_pred[i][0])
        N_cf_true[i][0] = torch.log10(N_cf_true[i][0])
        N_f_ture[num_f][0] = N_cf_true[i][0]
        N_f_pred[num_f][0] = N_cf_pred[i][0]
        num_f += 1
    elif(X_origin[i][0] == 1):    # PC
        t_c_ture[num_c][0] = t_cf_true[i][0]
        t_c_pred[num_c][0] = t_cf_pred[i][0]
        num_c += 1
# print(N_f_pred, num_c, num_f)
num_c = 0
num_f = 0
for i in range(t_cf_train_pred.shape[0]): # Cycle life calculation for the training set
    if(X_train_origin[i][0] == 2):    # PF
        N_cf_train_pred[i][0] = 10**t_cf_train_pred[i][0] * 3600
        N_cf_train_pred[i][0] /= (E_cf_train[i][0] * 2 / 0.5)
        N_cf_train_pred[i][0] = torch.log10(N_cf_train_pred[i][0])
        N_cf_train_true[i][0] = torch.log10(N_cf_train_true[i][0])
        N_f_train_ture[num_f][0] = N_cf_train_true[i][0]
        N_f_train_pred[num_f][0] = N_cf_train_pred[i][0]
        num_f += 1
    elif(X_train_origin[i][0] == 1):    # PC
        t_c_train_ture[num_c][0] = t_train[i][0]
        t_c_train_pred[num_c][0] = t_cf_train_pred[i][0]
        num_c += 1
# print(t_c_train_ture, num_c, num_f)
num_c = 0
num_f = 0
for i in range(t_cf_test_pred.shape[0]): # Cycle life calculation for the test set
    if(X_test_origin[i][0] == 2):    # PF
        N_cf_test_pred[i][0] = 10**t_cf_test_pred[i][0] * 3600
        N_cf_test_pred[i][0] /= (E_cf_test[i][0] * 2 / 0.5)
        N_cf_test_pred[i][0] = torch.log10(N_cf_test_pred[i][0])
        N_cf_test_true[i][0] = torch.log10(N_cf_test_true[i][0])
        N_f_test_ture[num_f][0] = N_cf_test_true[i][0]
        N_f_test_pred[num_f][0] = N_cf_test_pred[i][0]
        num_f += 1
    elif(X_test_origin[i][0] == 1):    # PC
        t_c_test_ture[num_c][0] = t_test[i][0]
        t_c_test_pred[num_c][0] = t_cf_test_pred[i][0]
        num_c += 1
# print(N_f_test_pred, num_c, num_f)
sign_TMF = 1
num_CFI = 0
num_TMF  = 0
for i in range(t_CFI_TMF_pred.shape[0]): # Cycle life calculation of CFI and TMF data
    if(X_new_origin[i][0] == 3):    # CFI
        N_CFI_TMF_pred[i][0] = 10**t_CFI_TMF_pred[i][0] * 3600
        N_CFI_TMF_pred[i][0] /= (E_CFI_TMF[i][0] * 2 / 0.5)
        N_CFI_TMF_pred[i][0] = torch.log10(N_CFI_TMF_pred[i][0])
        N_CFI_TMF_true[i][0] = torch.log10(N_CFI_TMF_true[i][0])
        N_CFI_true[num_CFI][0] = N_CFI_TMF_true[i][0]
        N_CFI_pred[num_CFI][0] = N_CFI_TMF_pred[i][0]
        num_CFI += 1
    if(X_new_origin[i][0] == 4):    # TMF
        N_CFI_TMF_pred[i][0] = 10**t_CFI_TMF_pred[i][0] * 3600
        if(sign_TMF <= 6):
            N_CFI_TMF_pred[i][0] /= 180
        elif(sign_TMF > 6 and sign_TMF <= 14):
            N_CFI_TMF_pred[i][0] /= 120
        sign_TMF += 1
        N_CFI_TMF_pred[i][0] = torch.log10(N_CFI_TMF_pred[i][0])
        N_CFI_TMF_true[i][0] = torch.log10(N_CFI_TMF_true[i][0])
        N_TMF_true[num_TMF][0] = N_CFI_TMF_true[i][0]
        N_TMF_pred[num_TMF][0] = N_CFI_TMF_pred[i][0]
        num_TMF += 1
# print(N_CFI_TMF_true)
# print(N_TMF_pred, num_CFI, num_TMF)
for i in range(t_CFI_TMF_pred.shape[0]): # Time-life calculation of CFI and TMF data
    if(X_new_origin[i][0] == 3):    # CFI
        t_cycle = 10**t_CFI_TMF_pred[i][0] * 3600 / (10**N_CFI_TMF_pred[i][0])
        t_cycle += 60
        t_CFI_TMF_pred[i][0] = (t_cycle * 10**N_CFI_TMF_pred[i][0]) / 3600
        t_CFI_TMF_pred[i][0] = torch.log10(t_CFI_TMF_pred[i][0])
    if(X_new_origin[i][0] == 4):    # TMF
        t_cycle = 10**t_CFI_TMF_pred[i][0] * 3600 / (10**N_CFI_TMF_pred[i][0])
        t_cycle += 0
        t_CFI_TMF_pred[i][0] = (t_cycle * 10**N_CFI_TMF_pred[i][0]) / 3600
        t_CFI_TMF_pred[i][0] = torch.log10(t_CFI_TMF_pred[i][0])
# print(t_CFI_TMF_pred)
# ======================================Predictive damage factor extraction======================================
td_cf_pred = model(X)[0:3]
td_cf_sum_pred = td_cf_pred[0] + td_cf_pred[1] + td_cf_pred[2]
td_cf_train_pred = model(X_train)[0:3]
td_cf_sum_train_pred = td_cf_train_pred[0] + td_cf_train_pred[1] + td_cf_train_pred[2]
td_cf_test_pred = model(X_test)[0:3]
td_cf_test_sum_pred = td_cf_test_pred[0] + td_cf_test_pred[1] + td_cf_test_pred[2]
td_CFI_TMF_pred = model(X_new)[0:3]
td_CFI_TMF_sum_pred = td_CFI_TMF_pred[0] + td_CFI_TMF_pred[1] + td_CFI_TMF_pred[2]
# ======================================Time Life Plotting======================================
x_sca = np.linspace(-1, 4, 100)
y_sca15 = x_sca + np.log10(1.5)
y_sca15_min = x_sca - np.log10(1.5)
y_sca2 = x_sca + np.log10(2)
y_sca2_min = x_sca - np.log10(2)
y_sca5 = x_sca + np.log10(3)
y_sca5_min = x_sca - np.log10(3)
plt.plot(x_sca, y_sca15, x_sca, y_sca15_min, x_sca, y_sca2, 
    x_sca, y_sca2_min, x_sca, y_sca5, x_sca, y_sca5_min)
# PC and PF
plt.scatter(t_train.detach(), t_cf_train_pred.detach(), label='Train', color='b')
plt.scatter(t_test.detach(), t_cf_test_pred.detach(), label='Test', color='r')
# CFI and TMF
plt.scatter(t_CFI_TMF_true.detach(), t_CFI_TMF_pred.detach(), label='New-CFI_TMF', color='g')
plt.savefig('Result_d-PINN_t.png', dpi=600)
# plt.show()
plt.close()
# ======================================Cycle Life Plotting======================================
x_sca = np.linspace(0, 5, 100)
y_sca15 = x_sca + np.log10(1.5)
y_sca15_min = x_sca - np.log10(1.5)
y_sca2 = x_sca + np.log10(2)
y_sca2_min = x_sca - np.log10(2)
y_sca5 = x_sca + np.log10(3)
y_sca5_min = x_sca - np.log10(3)
plt.plot(x_sca, y_sca15, x_sca, y_sca15_min, x_sca, y_sca2, 
    x_sca, y_sca2_min, x_sca, y_sca5, x_sca, y_sca5_min)
# CFI and TMF
plt.scatter(N_CFI_true.detach(), N_CFI_pred.detach(), label='New-CFI_TMF', color='b')
plt.scatter(N_TMF_true.detach(), N_TMF_pred.detach(), label='New-CFI_TMF', color='r')
plt.savefig('Result_d-PINN_N.png', dpi=600)
# plt.show()
# ======================================Damage Factor Plotting======================================
plt.scatter(10**t_train.detach(), td_cf_sum_train_pred.detach(), label='Train', color='b')
plt.scatter(10**t_test.detach(), td_cf_test_sum_pred.detach(), label='Test', color='r')
plt.scatter(10**t_CFI_TMF_true.detach(), td_CFI_TMF_sum_pred.detach(), label='New-CFI_TMF', color='g')
plt.savefig('Result_d-PINN_Damage.png', dpi=600)
# plt.show()
# ======================================Predicted longevity and damage preservation results======================================
writer = pd.ExcelWriter('Result_all_d-PINN.xlsx')
pd.DataFrame(torch.cat((t_cf_true.detach(), t_cf_pred.detach(), 10**t_cf_true.detach(), 10**t_cf_pred.detach(), N_cf_true.detach(), N_cf_pred.detach(), 10**N_cf_true.detach(), 10**N_cf_pred.detach(), 
    td_cf_pred[0].detach(), td_cf_pred[1].detach(), td_cf_pred[2].detach(), td_cf_sum_pred.detach()), dim=1).numpy()).to_excel(writer, sheet_name='All')
pd.DataFrame(torch.cat((t_train.detach(), t_cf_train_pred.detach(), 10**t_train.detach(), 10**t_cf_train_pred.detach(), N_cf_train_true.detach(), N_cf_train_pred.detach(), 10**N_cf_train_true.detach(), 10**N_cf_train_pred.detach(), 
    td_cf_train_pred[0].detach(), td_cf_train_pred[1].detach(), td_cf_train_pred[2].detach(), td_cf_sum_train_pred.detach()), dim=1).numpy()).to_excel(writer, sheet_name='Train')
pd.DataFrame(torch.cat((t_test.detach(), t_cf_test_pred.detach(), 10**t_test.detach(), 10**t_cf_test_pred.detach(), N_cf_test_true.detach(), N_cf_test_pred.detach(), 10**N_cf_test_true.detach(), 10**N_cf_test_pred.detach(), 
    td_cf_test_pred[0].detach(), td_cf_test_pred[1].detach(), td_cf_test_pred[2].detach(), td_cf_test_sum_pred.detach()), dim=1).numpy()).to_excel(writer, sheet_name='Test')
pd.DataFrame(torch.cat((t_CFI_TMF_true.detach(), t_CFI_TMF_pred.detach(), 10**t_CFI_TMF_true.detach(), 10**t_CFI_TMF_pred.detach(), N_CFI_TMF_true.detach(), N_CFI_TMF_pred.detach(), 10**N_CFI_TMF_true.detach(), 10**N_CFI_TMF_pred.detach(), 
    td_CFI_TMF_pred[0].detach(), td_CFI_TMF_pred[1].detach(), td_CFI_TMF_pred[2].detach(), td_CFI_TMF_sum_pred.detach()), dim=1).numpy()).to_excel(writer, sheet_name='CFI_TMF_new')
# ======================================Calculation and preservation of quantitative indicators======================================
[t_c_ture, t_c_pred, t_c_train_ture, t_c_train_pred, t_c_test_ture, t_c_test_pred] = [t_c_ture.reshape(-1).tolist(), t_c_pred.reshape(-1).tolist(), 
    t_c_train_ture.reshape(-1).tolist(), t_c_train_pred.reshape(-1).tolist(), t_c_test_ture.reshape(-1).tolist(), t_c_test_pred.reshape(-1).tolist()]
[N_f_ture, N_f_pred, N_f_train_ture, N_f_train_pred, N_f_test_ture, N_f_test_pred] = [N_f_ture.reshape(-1).tolist(), N_f_pred.reshape(-1).tolist(), 
    N_f_train_ture.reshape(-1).tolist(), N_f_train_pred.reshape(-1).tolist(), N_f_test_ture.reshape(-1).tolist(), N_f_test_pred.reshape(-1).tolist()]
[N_CFI_true, N_CFI_pred] = [N_CFI_true.reshape(-1).tolist(), N_CFI_pred.reshape(-1).tolist()]
[N_TMF_true, N_TMF_pred] = [N_TMF_true.reshape(-1).tolist(), N_TMF_pred.reshape(-1).tolist()]
[acy_c, acy_c_train, acy_c_test, acy_f, acy_f_train, acy_f_test] = [accuracy(t_c_ture, t_c_pred), accuracy(t_c_train_ture, t_c_train_pred), 
    accuracy(t_c_test_ture, t_c_test_pred), accuracy(N_f_ture, N_f_pred), accuracy(N_f_train_ture, N_f_train_pred), accuracy(N_f_test_ture, N_f_test_pred)]
[acy_CFI, acy_TMF] = [accuracy(N_CFI_true, N_CFI_pred), accuracy(N_TMF_true, N_TMF_pred)]
[acy_c.insert(0, 'PC'), acy_c_train.insert(0, 'PC_train'), acy_c_test.insert(0, 'PC_test'), acy_f.insert(0, 'PF'), 
    acy_f_train.insert(0, 'PF_train'), acy_f_test.insert(0, 'PF_test'), acy_CFI.insert(0, 'CFI'), acy_TMF.insert(0, 'TMF')]
acy_head = ['ACY', 'MSE', 'MAPE', 'SD', 'R2']
# print(acy_c, '\n', acy_c_train, '\n', acy_c_test, '\n', acy_f, '\n', acy_f_train, '\n', acy_f_test, '\n', acy_CFI, '\n', acy_TMF)
pd.DataFrame([acy_head, acy_c, acy_c_train, acy_c_test, acy_f, acy_f_train, acy_f_test, acy_CFI, acy_TMF]).to_excel(writer, sheet_name='Accuracy')
writer.close()
# ======================================Tensorboard visualization======================================
writer = SummaryWriter('logs')
dummy_input = torch.rand(1, 17)
writer.add_graph(model, dummy_input)
writer.close()