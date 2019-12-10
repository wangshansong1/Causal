import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import pandas as pd
import numpy as np
import torch

class MyDataset(Dataset):

    def __init__(self,source_data,labelpath_temp,Tipath_temp):
        self.source_data = source_data
        self.labelpath_temp = labelpath_temp
        self.Tipath_temp = Tipath_temp

    def __getitem__(self, index):

        return self.source_data[index],self.labelpath_temp[index],self.Tipath_temp[index]

    def __len__(self):
        return len(self.labelpath_temp)


def load_data(train_or_test, treatment_or_cortonl,batch_size,shuffle):
    source_data = pd.read_excel(
        '../dataset/twin/' + str(train_or_test) + '/' + str(treatment_or_cortonl) + '.xlsx')
    source_data = np.array(source_data)
    source_data = source_data[:,2:]

    labelpath_temp = []
    with open('../dataset/twin/' + str(train_or_test) + '/Yobs_' + str(
            treatment_or_cortonl) + '_'+str(train_or_test)+'.txt', 'r') as f:
        while True:
            line = f.readline()
            if line:
                line = line.replace('\n', '')
                line = line.replace('[', '')
                line = line.replace(']', '')
                line = float(line)

                labelpath_temp.append(line)
            else:
                break

    Tipath_temp = []
    with open('../dataset/twin/' + str(train_or_test) + '/Ti_' + str(
            treatment_or_cortonl) + '_' + str(train_or_test) + '.txt', 'r') as f:
        while True:
            line = f.readline()
            if line:
                line = line.replace('\n', '')
                line = line.replace('[', '')
                line = line.replace(']', '')
                line = float(line)

                Tipath_temp.append(line)
            else:
                break

    train_dataset = MyDataset(source_data, labelpath_temp,Tipath_temp)
    trainset_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    return trainset_dataloader

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1_1 = nn.Linear(50,5)
        self.fc1_2 = nn.Linear(5,1)


    def forward(self, x):
        out1 = self.fc1_1(x)
        out1 = self.sigmoid(out1)
        out1 = self.fc1_2(out1)
        out1 = self.sigmoid(out1)
        return out1

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-1 * x))

def train_M0(model, trainset_dataloader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target, Ti) in enumerate(trainset_dataloader):
        optimizer.zero_grad()
        output = model(data.float())
        loss_fn = nn.MSELoss(reduce=True, size_average=True)
        loss = loss_fn(target.float(),output.float())
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 1 == 0:
            print('Train W0 Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss: {:.6f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainset_dataloader.dataset),
                       100. * batch_idx / len(trainset_dataloader), loss.item(), loss.item(), loss.item()))


    if (epoch + 1) % 100 == 0:
        state = {'net':model.state_dict(), ' optimizer':optimizer.state_dict(), 'epoch':epoch}
        dirr = '.\weight_w0\X-learner\M_TWIN_' + str(epoch) + '.pth'
        torch.save(state, dirr)

def train_M1(model, trainset_dataloader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target, Ti) in enumerate(trainset_dataloader):
        optimizer.zero_grad()
        output = model(data.float())
        loss_fn = nn.MSELoss(reduce=True, size_average=True)
        loss = loss_fn(target.float(),output.float())
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 1 == 0:
            print('Train W1 Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss: {:.6f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainset_dataloader.dataset),
                       100. * batch_idx / len(trainset_dataloader), loss.item(), loss.item(), loss.item()))


    if (epoch + 1) % 100 == 0:
        state = {'net':model.state_dict(), ' optimizer':optimizer.state_dict(), 'epoch':epoch}
        dirr = '.\weight_w1\X-learner\M_TWIN_' + str(epoch) + '.pth'
        torch.save(state, dirr)

def train_M():
    import warnings

    warnings.filterwarnings("ignore")

    train_or_test = 'train'
    treatment_or_cortonl = 'treatment'
    trainset_dataloader1 = load_data(train_or_test, treatment_or_cortonl,64,True)

    model1 = M()
    optimizer1 = optim.Adam(model1.parameters())

    train_or_test0 = 'train'
    treatment_or_cortonl0 = 'control'
    trainset_dataloader0 = load_data(train_or_test0, treatment_or_cortonl0,64,True)

    model0 = M()
    optimizer0 = optim.Adam(model0.parameters())

    EPOCHS = 101

    for epoch in range(1, EPOCHS + 1):
        train_M0(model0, trainset_dataloader0, optimizer0, epoch)
        train_M1(model1, trainset_dataloader1, optimizer1, epoch)

def create_D():
    import warnings
    warnings.filterwarnings("ignore")

    source_data = pd.read_excel(
        '../dataset/twin/' + 'train' + '/' + 'treatment' + '.xlsx')
    source_data = np.array(source_data)
    source_data = source_data[:,2:]
    source_data = torch.from_numpy(source_data)

    model0 = M()
    checkpoint = torch.load('.\weight_w0\X-learner\M_TWIN_99.pth')
    model0.load_state_dict(checkpoint['net'])

    model0.eval()
    with torch.no_grad():
        output = model0(source_data.float())
    Yobs = []
    with open('../dataset/twin/train/Yobs_treatment_train.txt','r') as f:
        while True:
            line = f.readline()
            if line:
                line = line.replace('\n', '')
                line = line.replace('[', '')
                line = line.replace(']', '')
                line = float(line)

                Yobs.append(line)
            else:
                break
    output = output.numpy()
    Yobs = np.array(Yobs)
    Yobs = np.reshape(Yobs,(Yobs.shape[0],1))
    output1 = Yobs - output
    with open('../dataset/twin/D1.txt','w') as f:
        for i in output1:
            f.write(str(i) + '\n')

    source_data = pd.read_excel(
        '../dataset/twin/' + 'train' + '/' + 'control' + '.xlsx')
    source_data = np.array(source_data)
    source_data = source_data[:,2:]
    source_data = torch.from_numpy(source_data)

    model1 = M()
    checkpoint = torch.load('.\weight_w1\X-learner\M_TWIN_99.pth')
    model1.load_state_dict(checkpoint['net'])

    model1.eval()
    with torch.no_grad():
        output = model1(source_data.float())
    Yobs = []
    with open('../dataset/twin/train/Yobs_control_train.txt', 'r') as f:
        while True:
            line = f.readline()
            if line:
                line = line.replace('\n', '')
                line = line.replace('[', '')
                line = line.replace(']', '')
                line = float(line)

                Yobs.append(line)
            else:
                break
    output = output.numpy()
    Yobs = np.array(Yobs)
    Yobs = np.reshape(Yobs, (Yobs.shape[0], 1))
    output1 = output - Yobs
    with open('../dataset/twin/D0.txt', 'w') as f:
        for i in output1:
            f.write(str(i) + '\n')

def load_data_D(train_or_test, treatment_or_cortonl,_0_1,batch_size,shuffle):
    source_data = pd.read_excel(
        '../dataset/twin/' + str(train_or_test) + '/' + str(treatment_or_cortonl) + '.xlsx')
    source_data = np.array(source_data)
    source_data = source_data[:,2:]

    labelpath_temp = []
    with open('../dataset/twin/' +  '/D' + str(
            _0_1) + '.txt', 'r') as f:
        while True:
            line = f.readline()
            if line:
                line = line.replace('\n', '')
                line = line.replace('[', '')
                line = line.replace(']', '')
                line = float(line)

                labelpath_temp.append(line)
            else:
                break

    Tipath_temp = []
    with open('../dataset/twin/' + str(train_or_test) + '/Ti_' + str(
            treatment_or_cortonl) + '_' + str(train_or_test) + '.txt', 'r') as f:
        while True:
            line = f.readline()
            if line:
                line = line.replace('\n', '')
                line = line.replace('[', '')
                line = line.replace(']', '')
                line = float(line)

                Tipath_temp.append(line)
            else:
                break

    train_dataset = MyDataset(source_data, labelpath_temp,Tipath_temp)
    trainset_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    return trainset_dataloader

def train_T1_detail(model, trainset_dataloader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target, Ti) in enumerate(trainset_dataloader):
        optimizer.zero_grad()
        output = model(data.float())
        loss_fn = nn.MSELoss(reduce=True, size_average=True)
        loss = loss_fn(target.float(), output.float())
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 1 == 0:
            print('Train W0 Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss: {:.6f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainset_dataloader.dataset),
                       100. * batch_idx / len(trainset_dataloader), loss.item(), loss.item(), loss.item()))

    if (epoch + 1) % 100 == 0:
        state = {'net':model.state_dict(), ' optimizer':optimizer.state_dict(), 'epoch':epoch}
        dirr = '.\weight_w1\X-learner\T_TWIN_' + str(epoch) + '.pth'
        torch.save(state, dirr)


def train_T0_detail(model, trainset_dataloader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target, Ti) in enumerate(trainset_dataloader):
        optimizer.zero_grad()
        output = model(data.float())
        loss_fn = nn.MSELoss(reduce=True, size_average=True)
        loss = loss_fn(target.float(), output.float())
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 1 == 0:
            print('Train W0 Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss: {:.6f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainset_dataloader.dataset),
                       100. * batch_idx / len(trainset_dataloader), loss.item(), loss.item(), loss.item()))

    if (epoch + 1) % 100 == 0:
        state = {'net': model.state_dict(), ' optimizer': optimizer.state_dict(), 'epoch': epoch}
        dirr = '.\weight_w0\X-learner\T_TWIN_' + str(epoch) + '.pth'
        torch.save(state, dirr)


def train_T():
    import warnings

    warnings.filterwarnings("ignore")

    train_or_test = 'train'
    treatment_or_cortonl = 'treatment'
    _0_1 = '1'
    trainset_dataloader1 = load_data_D(train_or_test, treatment_or_cortonl,_0_1, 64, True)

    model1 = M()
    optimizer1 = optim.Adam(model1.parameters())

    train_or_test = 'train'
    treatment_or_cortonl = 'control'
    _0_1 = '0'
    trainset_dataloader0 = load_data_D(train_or_test, treatment_or_cortonl, _0_1, 64, True)

    model0 = M()
    optimizer0 = optim.Adam(model0.parameters())

    EPOCHS = 101

    for epoch in range(1, EPOCHS + 1):
        train_T0_detail(model0, trainset_dataloader0, optimizer0, epoch)
        train_T1_detail(model1, trainset_dataloader1, optimizer1, epoch)


def test_T0(model, trainset_dataloader):
    model.eval()
    with torch.no_grad():
        loss_ave = []
        for batch_idx, (data, target, Ti) in enumerate(trainset_dataloader):
            output_t = model(data.float())
            loss = nn.MSELoss(reduce=True, size_average=True)

            loss_pei_t = loss(output_t.float(), Ti.float()).float()
            loss_ave.append(loss_pei_t)
        loss_ate = np.mean(output_t.numpy()) / 64
        loss_pehe = sum(loss_ave) * 64 / trainset_dataloader.dataset.source_data.shape[0]
        loss_ave = sum(loss_ave) / batch_idx

        pass
        # print('\nTest set: Average loss: {:.4f}\n'.format(loss_ave))
    return loss_ave, loss_pehe, loss_ate

def test_T1(model, trainset_dataloader):
    model.eval()
    with torch.no_grad():
        loss_ave = []
        for batch_idx, (data, target, Ti) in enumerate(trainset_dataloader):
            output_t = model(data.float())
            loss = nn.MSELoss(reduce=True, size_average=True)

            loss_pei_t = loss(output_t.float(), Ti.float()).float()
            loss_ave.append(loss_pei_t)
        loss_ate = np.mean(output_t.numpy()) / 64
        loss_pehe = sum(loss_ave) * 64 / trainset_dataloader.dataset.source_data.shape[0]
        loss_ave = sum(loss_ave) / batch_idx

        pass
        # print('\nTest set: Average loss: {:.4f}\n'.format(loss_ave))
    return loss_ave, loss_pehe, loss_ate

def test1():
    import warnings
    warnings.filterwarnings("ignore")

    train_or_test = 'test'
    treatment_or_cortonl = 'treatment'
    trainset_dataloader = load_data(train_or_test, treatment_or_cortonl, 64, True)

    model1 = M()

    checkpoint = torch.load('.\weight_w1\X-learner\T_TWIN_99.pth')
    model1.load_state_dict(checkpoint['net'])

    cate1,cate11,cate12 = test_T1(model1,trainset_dataloader)

    train_or_test = 'test'
    treatment_or_cortonl = 'control'
    trainset_dataloader = load_data(train_or_test, treatment_or_cortonl, 64, True)

    model1 = M()

    checkpoint = torch.load('.\weight_w0\X-learner\T_TWIN_99.pth')
    model1.load_state_dict(checkpoint['net'])

    cate0,cate01,cate02 = test_T0(model1, trainset_dataloader)

    cate_ave = 0.2 * cate0 + 0.8 * cate1
    cate_ave1 = 0.2 * cate01 + 0.8 * cate11
    cate_ave2 = 0.2 * cate02 + 0.8 * cate12

    print('\nTest set: finall MSE_cate: {:.4f}\n'.format(cate_ave))
    print('\nTest set: finall PEHE_cate: {:.4f}\n'.format(cate_ave1))
    print('\nTest set: finall ATE_cate: {:.4f}\n'.format(cate_ave2))

def test2():
    import warnings
    warnings.filterwarnings("ignore")

    train_or_test = 'train'
    treatment_or_cortonl = 'treatment'
    trainset_dataloader = load_data(train_or_test, treatment_or_cortonl, 64, True)

    model1 = M()

    checkpoint = torch.load('.\weight_w1\X-learner\T_TWIN_99.pth')
    model1.load_state_dict(checkpoint['net'])

    cate1,cate11,cate12 = test_T1(model1,trainset_dataloader)

    train_or_test = 'train'
    treatment_or_cortonl = 'control'
    trainset_dataloader = load_data(train_or_test, treatment_or_cortonl, 64, True)

    model1 = M()

    checkpoint = torch.load('.\weight_w0\X-learner\T_TWIN_99.pth')
    model1.load_state_dict(checkpoint['net'])

    cate0,cate01,cate02 = test_T0(model1, trainset_dataloader)

    cate_ave = 0.2 * cate0 + 0.8 * cate1
    cate_ave1 = 0.2 * cate01 + 0.8 * cate11
    cate_ave2 = 0.2 * cate02 + 0.8 * cate12

    print('\nTest set: finall MSE_cate: {:.4f}\n'.format(cate_ave))
    print('\nTest set: finall PEHE_cate: {:.4f}\n'.format(cate_ave1))
    print('\nTest set: finall ATE_cate: {:.4f}\n'.format(cate_ave2))

if __name__ == '__main__':
    # train_M()
    # create_D()
    # train_T()
    test1()
    test2()