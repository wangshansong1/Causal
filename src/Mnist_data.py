# -*- coding: utf-8 -*-
import pickle
import numpy as np
import os
import matplotlib.image as mpimg

C = np.array([0,1,2,3,4,5,6,7,8,9])

mj = np.array(np.random.uniform(-3,3,10))
tj = np.array(np.random.uniform(-1,1,10))
pj = np.array(np.random.uniform(0.3,0.7,10))

data_10k_0 = []
data_10k_1 = []
data_10k_2 = []
data_10k_3 = []
data_10k_4 = []
data_10k_5 = []
data_10k_6 = []
data_10k_7 = []
data_10k_8 = []
data_10k_9 = []

data_60k_0 = []
data_60k_1 = []
data_60k_2 = []
data_60k_3 = []
data_60k_4 = []
data_60k_5 = []
data_60k_6 = []
data_60k_7 = []
data_60k_8 = []
data_60k_9 = []

mu0j_init = mj + 3*C
mu1j_init = mu0j_init + tj
ej_init = pj

def get_Mnist_data10k(sca):
    with open('../dataset_/label10k.txt','r') as f:
        line = f.readline()
        line = line.replace('\n','')
        line = line.replace(',','')
        label_10k = np.array(list(line))



    for index, value in enumerate(label_10k):
        # 读一张图片并将其转换为一维
        img = mpimg.imread('../dataset_/test10k/' + str(index) + '.png')
        img = img.reshape(28,28,1)

        mu0 = mu0j_init[int(value)]
        mu1 = mu1j_init[int(value)]
        e = ej_init[int(value)]
        epsilon = np.random.normal(loc=0, scale=sca, size=1)
        # epsilon = 0

        Yi0 = mu0 + epsilon
        Yi1 = mu1 + epsilon
        Wi = np.random.binomial(1, e, size=1)

        individual = []
        individual.append(Yi0)
        individual.append(Yi1)
        individual.append(Wi)
        individual.append(img)

        if value == '0':
            data_10k_0.append(individual)
        elif value == '1':
            data_10k_1.append(individual)
        elif value == '2':
            data_10k_2.append(individual)
        elif value == '3':
            data_10k_3.append(individual)
        elif value == '4':
            data_10k_4.append(individual)
        elif value == '5':
            data_10k_5.append(individual)
        elif value == '6':
            data_10k_6.append(individual)
        elif value == '7':
            data_10k_7.append(individual)
        elif value == '8':
            data_10k_8.append(individual)
        elif value == '9':
            data_10k_9.append(individual)

    with open('../kr_data/test/test.0', 'wb') as f:
        pickle.dump(data_10k_0, f)
    with open('../kr_data/test/test.1', 'wb') as f:
        pickle.dump(data_10k_1, f)
    with open('../kr_data/test/test.2', 'wb') as f:
        pickle.dump(data_10k_2, f)
    with open('../kr_data/test/test.3', 'wb') as f:
        pickle.dump(data_10k_3, f)
    with open('../kr_data/test/test.4', 'wb') as f:
        pickle.dump(data_10k_4, f)
    with open('../kr_data/test/test.5', 'wb') as f:
        pickle.dump(data_10k_5, f)
    with open('../kr_data/test/test.6', 'wb') as f:
        pickle.dump(data_10k_6, f)
    with open('../kr_data/test/test.7', 'wb') as f:
        pickle.dump(data_10k_7, f)
    with open('../kr_data/test/test.8', 'wb') as f:
        pickle.dump(data_10k_8, f)
    with open('../kr_data/test/test.9', 'wb') as f:
        pickle.dump(data_10k_9, f)

def get_Mnist_data60k(sca):
    with open('../dataset_/label60k.txt','r') as f:
        line = f.readline()
        line = line.replace('\n','')
        line = line.replace(',','')
        label_60k = np.array(list(line))

    for index, value in enumerate(label_60k):
        # 读一张图片并将其转换为一维
        img = mpimg.imread('../dataset_/test60k/' + str(index) + '.png')
        img = img.reshape(28,28,1)

        mu0 = mu0j_init[int(value)]
        mu1 = mu1j_init[int(value)]
        e = ej_init[int(value)]
        epsilon = np.random.normal(loc=0, scale=sca, size=1)
        # epsilon = 0

        Yi0 = mu0 + epsilon
        Yi1 = mu1 + epsilon
        Wi = np.random.binomial(1, e, size=1)

        individual = []
        individual.append(Yi0)
        individual.append(Yi1)
        individual.append(Wi)
        individual.append(img)

        if value == '0':
            data_60k_0.append(individual)
        elif value == '1':
            data_60k_1.append(individual)
        elif value == '2':
            data_60k_2.append(individual)
        elif value == '3':
            data_60k_3.append(individual)
        elif value == '4':
            data_60k_4.append(individual)
        elif value == '5':
            data_60k_5.append(individual)
        elif value == '6':
            data_60k_6.append(individual)
        elif value == '7':
            data_60k_7.append(individual)
        elif value == '8':
            data_60k_8.append(individual)
        elif value == '9':
            data_60k_9.append(individual)

    with open('../kr_data/train/train.0', 'wb') as f:
        pickle.dump(data_60k_0, f)
    with open('../kr_data/train/train.1', 'wb') as f:
        pickle.dump(data_60k_1, f)
    with open('../kr_data/train/train.2', 'wb') as f:
        pickle.dump(data_60k_2, f)
    with open('../kr_data/train/train.3', 'wb') as f:
        pickle.dump(data_60k_3, f)
    with open('../kr_data/train/train.4', 'wb') as f:
        pickle.dump(data_60k_4, f)
    with open('../kr_data/train/train.5', 'wb') as f:
        pickle.dump(data_60k_5, f)
    with open('../kr_data/train/train.6', 'wb') as f:
        pickle.dump(data_60k_6, f)
    with open('../kr_data/train/train.7', 'wb') as f:
        pickle.dump(data_60k_7, f)
    with open('../kr_data/train/train.8', 'wb') as f:
        pickle.dump(data_60k_8, f)
    with open('../kr_data/train/train.9', 'wb') as f:
        pickle.dump(data_60k_9, f)

def make_d(number,train_or_test,mu0j_init,mu1j_init,ej_init):
    path = 'C:\\Users\\wansy132\\PycharmProjects\\pycharm_causal\\dataset\\mnist\\' + train_or_test + '\\' + str(number)

    mu0 = mu0j_init[int(number)]
    mu1 = mu1j_init[int(number)]
    e = ej_init[int(number)]

    folder = os.path.exists('./Settings/' + train_or_test + '/' + str(number)  + '/')
    if not folder:
        os.makedirs('./Settings/' + train_or_test + '/' + str(number)  + '/')

    f1 = open('./Settings/' + train_or_test + '/' + str(number)  + '/' + 'Yi0.txt','w')
    f2 = open('./Settings/' + train_or_test + '/' + str(number)  + '/' + 'Yi1.txt','w')
    f3 = open('./Settings/' + train_or_test + '/' + str(number)  + '/' + 'Wi.txt','w')

    for index in range(1,len(os.listdir(path))+1):

        epsilon = np.random.normal(loc=0, scale=1, size=1)

        Yi0 = mu0 + epsilon
        Yi1 = mu1 + epsilon
        Wi = np.random.binomial(1, e, size=1)

        f1.write(str(float(Yi0)) + '\n')
        f2.write(str(float(Yi1)) + '\n')
        f3.write(str(float(Wi)) + '\n')

    f1.close()
    f2.close()
    f3.close()

def create_d():
    C = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    mj = np.array(np.random.uniform(-3, 3, 10))
    tj = np.array(np.random.uniform(-1, 1, 10))
    pj = np.array(np.random.uniform(0.3, 0.7, 10))

    mu0j_init = mj + 3 * C
    mu1j_init = mu0j_init + tj
    ej_init = pj

    for i in range(0, 10):
        make_d(i, 'train', mu0j_init, mu1j_init, ej_init)
        make_d(i, 'test', mu0j_init, mu1j_init, ej_init)

if __name__ == '__main__':
    # get_Mnist_data10k(1)
    # get_Mnist_data60k(1)
    pass
