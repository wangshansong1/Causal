import pandas as pd
import numpy as np

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def DGP_1(train_or_test,treatment_or_cortonl, epsilon_u0, epsilon_u1):

    source_data = pd.read_excel('../dataset/twin/' + str(train_or_test) + '/' + str(treatment_or_cortonl) + '.xlsx')
    source_data = np.array(source_data)
    source_data = source_data[:,2:]

    inX_u0 = np.dot(source_data, epsilon_u0.reshape(50, 1))
    inX_u1 = np.dot(source_data, epsilon_u1.reshape(50, 1))

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
    epsilon_u0 = np.random.normal(loc=0, scale=0.001, size=50)
    epsilon_u1 = np.random.normal(loc=0, scale=0.001, size=50)

    outX_u0_tr, outY_u1_tr, Ti_tr = DGP_1('train', 'treatment', epsilon_u0, epsilon_u1)
    outX_u0_te, outY_u1_te, Ti_te = DGP_1('test', 'treatment', epsilon_u0, epsilon_u1)


    with open('Yobs_treatment_train.txt', 'w') as f:
        for i in outY_u1_tr:
            f.write(str(i) + '\n')

    with open('Yobs_treatment_test.txt', 'w') as f:
        for i in outY_u1_te:
            f.write(str(i) + '\n')

    with open('Ti_treatment_train.txt', 'w') as f:
        for i in Ti_tr:
            f.write(str(i) + '\n')

    with open('Ti_treatment_test.txt', 'w') as f:
        for i in Ti_te:
            f.write(str(i) + '\n')

    epsilon_u0 = np.random.normal(loc=0, scale=0.001, size=50)
    epsilon_u1 = np.random.normal(loc=0, scale=0.001, size=50)

    outX_u0_tr, outY_u1_tr, Ti_tr = DGP_1('train', 'control', epsilon_u0, epsilon_u1)
    outX_u0_te, outY_u1_te, Ti_te = DGP_1('test', 'control', epsilon_u0, epsilon_u1)


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
