import pandas as pd
import numpy as np

from sklearn.preprocessing import normalize

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


def DGP_1(train_or_test,treatment_or_cortonl, epsilon_u0, epsilon_u1):



    source_data = pd.read_excel('../dataset/MineThatData/4/' + str(train_or_test) +'/'+ str(treatment_or_cortonl) + '.xlsx')
    source_data = np.array(source_data)
    source_data = source_data[:, :-1]
    source0 = source_data[:,0]
    source0 = np.reshape(source0, [source0.shape[0], 1])
    source1 = source_data[:, 1]
    source1 = np.reshape(source1, [source1.shape[0], 1])
    for ind,i in enumerate(source1):
        source1[ind] = int(i[0][0])
    source2 = source_data[:, 2]
    source2 = np.reshape(source2,[source2.shape[0],1])
    source2 = normalize(source2, axis=0, norm='max')
    source3 = source_data[:, 3]
    source3 = np.reshape(source3, [source1.shape[0], 1])
    source4 = source_data[:, 4]
    source4 = np.reshape(source4, [source1.shape[0], 1])
    source5 = source_data[:, 5]
    for ind,i in enumerate(source5):
        if i =='Rural':
            source5[ind] = 1
        elif i =='Urban':
            source5[ind] = 2
        elif i =='Surburban':
            source5[ind] = 3
    source5 = np.reshape(source5, [source1.shape[0], 1])
    source6 = source_data[:, 6]
    source6 = np.reshape(source6, [source1.shape[0], 1])
    source7 = source_data[:, 7]
    for ind,i in enumerate(source7):
        if i =='Web':
            source7[ind] = 1
        elif i =='Phone':
            source7[ind] = 2
        elif i =='Multichannel':
            source7[ind] = 3
    source7 = np.reshape(source7, [source1.shape[0], 1])
    source_data = np.hstack((source0,source1,source2,source3,source4,source5,source6,source7))
    source_data = source_data.astype(np.float32)


    inX_u0 = np.dot(source_data, epsilon_u0.reshape(8, 1))
    inX_u1 = np.dot(source_data, epsilon_u1.reshape(8, 1))

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
        epsilon_u0 = np.random.normal(loc=0, scale=1, size=8)
        epsilon_u1 = np.random.normal(loc=0, scale=1, size=8)

        outX_u0_tr,outY_u1_tr,Ti_tr = DGP_1('train','treatment', epsilon_u0, epsilon_u1)
        outX_u0_te,outY_u1_te,Ti_te = DGP_1('test', 'treatment', epsilon_u0, epsilon_u1)

        if np.mean(outX_u0_tr) >= 0.1 and np.mean(outX_u0_tr) <= 0.85 and np.mean(outX_u0_te) >= 0.1 and np.mean(outX_u0_te) <= 0.85\
                and np.mean(outY_u1_tr) >= 0.1 and np.mean(outY_u1_tr) <= 0.85 and np.mean(outY_u1_te) >= 0.1 and np.mean(outY_u1_te) <= 0.85:
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
        epsilon_u0 = np.random.normal(loc=0, scale=1, size=8)
        epsilon_u1 = np.random.normal(loc=0, scale=1, size=8)

        outX_u0_tr, outY_u1_tr, Ti_tr = DGP_1('train', 'control', epsilon_u0, epsilon_u1)
        outX_u0_te, outY_u1_te, Ti_te = DGP_1('test', 'control', epsilon_u0, epsilon_u1)

        if np.mean(outX_u0_tr) >= 0.1 and np.mean(outX_u0_tr) <= 0.85 and np.mean(outX_u0_te) >= 0.1 and np.mean(
                outX_u0_te) <= 0.85 \
                and np.mean(outY_u1_tr) >= 0.1 and np.mean(outY_u1_tr) <= 0.85 and np.mean(
            outY_u1_te) >= 0.1 and np.mean(outY_u1_te) <= 0.85:
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


