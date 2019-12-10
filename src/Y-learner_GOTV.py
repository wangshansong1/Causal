import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import pandas as pd
import numpy as np
import torch

class Y_Learner(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1_1 = nn.Linear(11,5)
        self.fc1_2 = nn.Linear(5,1)

        self.fc2_1 = nn.Linear(11, 5)
        self.fc2_2 = nn.Linear(5, 1)

        self.fc3_1 = nn.Linear(11, 5)
        self.fc3_2 = nn.Linear(5, 1)

    def forward(self, x):
        x1 = x
        x2 = x
        x3 = x
        out1 = self.fc1_1(x1)
        out1 = self.sigmoid(out1)
        out1 = self.fc1_2(out1)
        out1 = self.sigmoid(out1)

        out2 = self.fc2_1(x2)
        out2 = self.sigmoid(out2)
        out2 = self.fc2_2(out2)
        out2 = self.sigmoid(out2)

        out3 = self.fc3_1(x3)
        out3 = self.sigmoid(out3)
        out3 = self.fc3_2(out3)
        out3 = self.sigmoid(out3)
        return out1,out2,out3

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-1 * x))



class MyDataset(Dataset):

    def __init__(self,source_data,labelpath_temp,Tipath_temp):
        self.source_data = source_data
        self.labelpath_temp = labelpath_temp
        self.Tipath_temp = Tipath_temp

    def __getitem__(self, index):

        return self.source_data[index],self.labelpath_temp[index],self.Tipath_temp[index]

    def __len__(self):
        return len(self.labelpath_temp)


def load_data(train_or_test, treatment_or_cortonl):
    source_data = pd.read_excel(
        '../dataset/gerber_huber_2014_data/' + str(train_or_test) + '/' + str(treatment_or_cortonl) + '.xlsx')
    source_data = np.array(source_data)
    race_data = source_data[:, 10:-1]
    age = source_data[:, 9]
    age = age / np.mean(age)
    age = age.reshape(age.shape[0], 1)
    female = source_data[:, -1].reshape(source_data.shape[0], 1)
    race_number = np.array([1, 2, 3, 4])
    race_number = race_number.reshape(4, 1)
    race = np.dot(race_data, race_number)
    race = race / np.mean(race)
    source_data = np.hstack((source_data[:, 1:9], age, race, female))

    labelpath_temp = []
    with open('../dataset/gerber_huber_2014_data/' + str(train_or_test) + '/Yobs_' + str(
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
    with open('../dataset/gerber_huber_2014_data/' + str(train_or_test) + '/Ti_' + str(
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
    trainset_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4)

    return trainset_dataloader

def train_W0(model, trainset_dataloader, optimizer, epoch):

    model.train()

    weight_0 = [True,True,True,True,False,False,False,False,False,False,False,False]
    weight_1 = [False,False,False,False,True,True,True,True,False,False,False,False]
    weight_t = [False,False,False,False,False,False,False,False,True,True,True,True]

    for batch_idx, (data, target,Ti) in enumerate(trainset_dataloader):

        for i in [1,2,3]:
            if i == 1:
                optimizer.zero_grad()

                for index, para in enumerate(model.parameters()):
                    para.requires_grad = weight_0[index]
                output_0, output_1, output_t = model(data.float())

                loss = nn.MSELoss(reduce=True, size_average=True)

                loss_pei_0 = loss(target.float(), output_0.float())
                loss_pei_0.backward(retain_graph=True)
                optimizer.step()
            if i == 2:
                optimizer.zero_grad()

                for index, para in enumerate(model.parameters()):
                    para.requires_grad = weight_1[index]
                output_0, output_1, output_t = model(data.float())

                loss = nn.MSELoss(reduce=True, size_average=True)

                loss_pei_1 = loss(output_1.float(), torch.add(target.float(), output_t.float()).float())
                loss_pei_1.backward(retain_graph=True)
                optimizer.step()
            if i == 3:
                optimizer.zero_grad()

                for index, para in enumerate(model.parameters()):
                    para.requires_grad = weight_t[index]
                output_0, output_1, output_t = model(data.float())

                loss = nn.MSELoss(reduce=True, size_average=True)

                loss_pei_t = loss(output_t.float(),torch.sub(output_1.float(),target.float()).float())
                loss_pei_t.backward(retain_graph=True)
                optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            print('Train W0 Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss: {:.6f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainset_dataloader.dataset),
                       100. * batch_idx / len(trainset_dataloader), loss_pei_0.item(), loss_pei_1.item(), loss_pei_t.item()))

    if (epoch + 1) % 100 == 0:
        state = {'net':model.state_dict(), ' optimizer':optimizer.state_dict(), 'epoch':epoch}
        dirr = '.\weight_w0\Y-learner\GOTV_' + str(epoch) + '.pth'
        torch.save(state, dirr)

def train_W1(model, trainset_dataloader, optimizer, epoch):

    model.train()

    weight_0 = [True,True,True,True,False,False,False,False,False,False,False,False]
    weight_1 = [False,False,False,False,True,True,True,True,False,False,False,False]
    weight_t = [False,False,False,False,False,False,False,False,True,True,True,True]

    for batch_idx, (data, target,Ti) in enumerate(trainset_dataloader):

        for i in [1,2,3]:
            if i == 1:
                optimizer.zero_grad()

                for index, para in enumerate(model.parameters()):
                    para.requires_grad = weight_0[index]
                output_0, output_1, output_t = model(data.float())

                loss = nn.MSELoss(reduce=True, size_average=True)

                loss_pei_0 = loss(torch.sub(target.float(),output_t.float()).float(), output_0.float())
                loss_pei_0.backward(retain_graph=True)
                optimizer.step()
            if i == 2:
                optimizer.zero_grad()

                for index, para in enumerate(model.parameters()):
                    para.requires_grad = weight_1[index]
                output_0, output_1, output_t = model(data.float())

                loss = nn.MSELoss(reduce=True, size_average=True)

                loss_pei_1 = loss(output_1.float(), target.float())
                loss_pei_1.backward(retain_graph=True)

            if i == 3:
                optimizer.zero_grad()

                for index, para in enumerate(model.parameters()):
                    para.requires_grad = weight_t[index]
                output_0, output_1, output_t = model(data.float())

                loss = nn.MSELoss(reduce=True, size_average=True)

                loss_pei_t = loss(output_t.float(),torch.sub(target.float(),output_0.float()).float())
                loss_pei_t.backward(retain_graph=True)
                optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            print('Train W1 Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss: {:.6f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainset_dataloader.dataset),
                       100. * batch_idx / len(trainset_dataloader), loss_pei_0.item(), loss_pei_1.item(), loss_pei_t.item()))

    if (epoch + 1) % 100 == 0:
        state = {'net':model.state_dict(), ' optimizer':optimizer.state_dict(), 'epoch':epoch}
        dirr = '.\weight_w1\Y-learner\GOTV_' + str(epoch) + '.pth'
        torch.save(state, dirr)

def train():
    import warnings
    warnings.filterwarnings("ignore")

    train_or_test = 'train'
    treatment_or_cortonl = 'treatment'
    trainset_dataloader = load_data(train_or_test, treatment_or_cortonl)

    train_or_test1 = 'train'
    treatment_or_cortonl1 = 'control'
    trainset_dataloader1 = load_data(train_or_test1, treatment_or_cortonl1)

    model1 = Y_Learner()
    optimizer1 = optim.Adam(model1.parameters())

    model2 = Y_Learner()
    optimizer2 = optim.Adam(model2.parameters())

    EPOCHS = 101

    for epoch in range(1, EPOCHS + 1):
        train_W1(model1, trainset_dataloader, optimizer1, epoch)
        train_W0(model2, trainset_dataloader1, optimizer2, epoch)

def test_W0(model, trainset_dataloader):
    model.eval()
    with torch.no_grad():
        loss_ave = []
        for batch_idx, (data, target, Ti) in enumerate(trainset_dataloader):
            output_0, output_1, output_t = model(data.float())
            loss = nn.MSELoss(reduce=True, size_average=True)

            loss_pei_t = loss(output_t.float(), Ti.float()).float()
            loss_ave.append(loss_pei_t)
        loss_ate = np.mean(output_t.numpy()) / 64
        loss_pehe = sum(loss_ave) * 64 / trainset_dataloader.dataset.source_data.shape[0]
        loss_ave = sum(loss_ave) / batch_idx

        pass
        # print('\nTest set: Average loss: {:.4f}\n'.format(loss_ave))
    return loss_ave, loss_pehe, loss_ate

def test_W1(model, trainset_dataloader):
    model.eval()
    with torch.no_grad():
        loss_ave = []
        for batch_idx, (data, target, Ti) in enumerate(trainset_dataloader):
            output_0, output_1, output_t = model(data.float())
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
    trainset_dataloader = load_data(train_or_test, treatment_or_cortonl)

    model1 = Y_Learner()

    checkpoint = torch.load('.\weight_w1\Y-learner\GOTV_99.pth')
    model1.load_state_dict(checkpoint['net'])

    cate1,cate11,cate12 = test_W1(model1,trainset_dataloader)

    train_or_test = 'test'
    treatment_or_cortonl = 'control'
    trainset_dataloader = load_data(train_or_test, treatment_or_cortonl)

    model1 = Y_Learner()

    checkpoint = torch.load('.\weight_w0\Y-learner\GOTV_99.pth')
    model1.load_state_dict(checkpoint['net'])

    cate0,cate01,cate02 = test_W0(model1, trainset_dataloader)

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
    trainset_dataloader = load_data(train_or_test, treatment_or_cortonl)

    model1 = Y_Learner()

    checkpoint = torch.load('.\weight_w1\Y-learner\GOTV_99.pth')
    model1.load_state_dict(checkpoint['net'])

    cate1,cate11,cate12 = test_W1(model1,trainset_dataloader)

    train_or_test = 'train'
    treatment_or_cortonl = 'control'
    trainset_dataloader = load_data(train_or_test, treatment_or_cortonl)

    model1 = Y_Learner()

    checkpoint = torch.load('.\weight_w0\Y-learner\GOTV_99.pth')
    model1.load_state_dict(checkpoint['net'])

    cate0,cate01,cate02 = test_W0(model1, trainset_dataloader)

    cate_ave = 0.2 * cate0 + 0.8 * cate1
    cate_ave1 = 0.2 * cate01 + 0.8 * cate11
    cate_ave2 = 0.2 * cate02 + 0.8 * cate12

    print('\nTest set: finall MSE_cate: {:.4f}\n'.format(cate_ave))
    print('\nTest set: finall PEHE_cate: {:.4f}\n'.format(cate_ave1))
    print('\nTest set: finall ATE_cate: {:.4f}\n'.format(cate_ave2))



if __name__ == '__main__':
    # train()
    test1()
    test2()


