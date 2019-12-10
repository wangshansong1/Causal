import pandas as pd
import numpy as np


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


def DGP_1(train_or_test,treatment_or_cortonl, epsilon_u0, epsilon_u1):



    source_data = pd.read_excel('../dataset/gerber_huber_2014_data/' + str(train_or_test) +'/'+ str(treatment_or_cortonl) + '.xlsx')
    source_data = np.array(source_data)
    race_data = source_data[:,10:-1]
    age = source_data[:,9]
    age = age/np.mean(age)
    age = age.reshape(age.shape[0],1)
    female = source_data[:,-1].reshape(source_data.shape[0],1)
    race_number = np.array([1,2,3,4])
    race_number = race_number.reshape(4,1)
    race = np.dot(race_data,race_number)
    race = race/np.mean(race)
    source_data = np.hstack((source_data[:,1:9],age,race,female))

    inX_u0 = np.dot(source_data, epsilon_u0.reshape(11, 1))
    inX_u1 = np.dot(source_data, epsilon_u1.reshape(11, 1))

    outX_u0 = sigmoid(inX_u0)
    outX_u1 = sigmoid(inX_u1)

    Ti = outX_u1 - outX_u0
    if treatment_or_cortonl == 'treatment':
        outY_u1 = []
        for index,value in enumerate(outX_u1):
            outY_u1.append(np.random.binomial(1, value, size=1))
        return outX_u0,outY_u1, Ti

    if treatment_or_cortonl == 'control':
        outY_u0 = []
        for index,value in enumerate(outX_u0):
            outY_u0.append(np.random.binomial(1, value, size=1))
        return outX_u1,outY_u0, Ti



def DGP_2():
    while True:
        epsilon_u0 = np.random.normal(loc=0, scale=1, size=11)
        epsilon_u1 = np.random.normal(loc=0, scale=1, size=11)

        outX_u0_tr,outY_u1_tr,Ti_tr = DGP_1('train','treatment', epsilon_u0, epsilon_u1)
        outX_u0_te,outY_u1_te,Ti_te = DGP_1('test', 'treatment', epsilon_u0, epsilon_u1)

        if np.mean(outX_u0_tr) >= 0.4 and np.mean(outX_u0_tr) <= 0.65 and np.mean(outX_u0_te) >= 0.4 and np.mean(outX_u0_te) <= 0.65\
                and np.mean(outY_u1_tr) >= 0.4 and np.mean(outY_u1_tr) <= 0.65 and np.mean(outY_u1_te) >= 0.4 and np.mean(outY_u1_te) <= 0.65:
            break



    with open('Yobs_treatment_train.txt','w') as f:
        for i in outY_u1_tr:
            f.write(str(i) + '\n')


    with open('Yobs_treatment_test.txt','w') as f:
        for i in outY_u1_te:
            f.write(str(i) + '\n')

    with open('Ti_treatment_train.txt','w') as f:
        for i in Ti_tr:
            f.write(str(i) + '\n')


    with open('Ti_treatment_test.txt','w') as f:
        for i in Ti_te:
            f.write(str(i) + '\n')

    while True:
        epsilon_u0 = np.random.normal(loc=0, scale=1, size=11)
        epsilon_u1 = np.random.normal(loc=0, scale=1, size=11)

        outX_u0_tr, outY_u1_tr, Ti_tr = DGP_1('train', 'control', epsilon_u0, epsilon_u1)
        outX_u0_te, outY_u1_te, Ti_te = DGP_1('test', 'control', epsilon_u0, epsilon_u1)

        if np.mean(outX_u0_tr) >= 0.4 and np.mean(outX_u0_tr) <= 0.65 and np.mean(outX_u0_te) >= 0.4 and np.mean(
                outX_u0_te) <= 0.65 \
                and np.mean(outY_u1_tr) >= 0.4 and np.mean(outY_u1_tr) <= 0.65 and np.mean(
            outY_u1_te) >= 0.4 and np.mean(outY_u1_te) <= 0.65:
            break

    with open('Yobs_control_train.txt', 'w') as f:
        for i in outY_u1_tr:
            f.write(str(i) + '\n')

    with open('Yobs_control_test.txt', 'w') as f:
        for i in outY_u1_te:
            f.write(str(i) + '\n')

    with open('Ti_control_train.txt', 'w') as f:
        for i in Ti_tr:
            f.write(str(i) + '\n')

    with open('Ti_control_test.txt', 'w') as f:
        for i in Ti_te:
            f.write(str(i) + '\n')

if __name__ == '__main__':
    DGP_2()


