from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

def load_data(train_or_test, treatment_or_cortonl,dataname):
    source_data = pd.read_excel(
        '../dataset/' + str(dataname) + '/' + str(train_or_test) + '/' + str(treatment_or_cortonl) + '.xlsx')
    if dataname == 'gerber_huber_2014_data':
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
    elif dataname == 'criteo':
        source_data = np.array(source_data)
        source_data = source_data[:, :-4]
    elif dataname == 'file20':
        source_data = np.array(source_data)
        source_data = source_data[:, :-3]
    elif dataname == 'ihdp':
        source_data = np.array(source_data)
        source_data = source_data[:, 3:]
    elif dataname == 'job':
        source_data = np.array(source_data)
        source_data = source_data[:, :-1]
    elif dataname == 'twin':
        source_data = np.array(source_data)
        source_data = source_data[:, 2:]
    elif dataname == 'MineThatData/1' or dataname == 'MineThatData/2' or dataname == 'MineThatData/3' or dataname == 'MineThatData/4' :
        source_data = np.array(source_data)
        source_data = source_data[:, :-1]
        source0 = source_data[:, 0]
        source0 = np.reshape(source0, [source0.shape[0], 1])
        source1 = source_data[:, 1]
        source1 = np.reshape(source1, [source1.shape[0], 1])
        for ind, i in enumerate(source1):
            source1[ind] = int(i[0][0])
        source2 = source_data[:, 2]
        source2 = np.reshape(source2, [source2.shape[0], 1])
        source2 = normalize(source2, axis=0, norm='max')
        source3 = source_data[:, 3]
        source3 = np.reshape(source3, [source1.shape[0], 1])
        source4 = source_data[:, 4]
        source4 = np.reshape(source4, [source1.shape[0], 1])
        source5 = source_data[:, 5]
        for ind, i in enumerate(source5):
            if i == 'Rural':
                source5[ind] = 1
            elif i == 'Urban':
                source5[ind] = 2
            elif i == 'Surburban':
                source5[ind] = 3
        source5 = np.reshape(source5, [source1.shape[0], 1])
        source6 = source_data[:, 6]
        source6 = np.reshape(source6, [source1.shape[0], 1])
        source7 = source_data[:, 7]
        for ind, i in enumerate(source7):
            if i == 'Web':
                source7[ind] = 1
            elif i == 'Phone':
                source7[ind] = 2
            elif i == 'Multichannel':
                source7[ind] = 3
        source7 = np.reshape(source7, [source1.shape[0], 1])
        source_data = np.hstack((source0, source1, source2, source3, source4, source5, source6, source7))
        source_data = source_data.astype(np.float32)

    labelpath_temp = []
    with open('../dataset/' + str(dataname) + '/' + str(train_or_test) + '/Yobs_' + str(
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
    with open('../dataset/' + str(dataname) + '/' + str(train_or_test) + '/Ti_' + str(
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

    return source_data, labelpath_temp,Tipath_temp

def tree_and_random(train_source_data, train_labelpath_temp,test_source_data, test_labelpath_temp,flag):
    tree = DecisionTreeRegressor(criterion="mse")
    tree.fit(train_source_data, train_labelpath_temp)

    predict_result = tree.predict(test_source_data).tolist()
    for ind,i in enumerate(predict_result):
        if flag == '1':
            predict_result[ind] = test_labelpath_temp[ind] - i
        else:
            predict_result[ind] = i - test_labelpath_temp[ind]

    model = RandomForestRegressor(n_estimators=100,
                                   bootstrap=True,
                                   max_features='sqrt',
                                  criterion="mse")

    model.fit(train_source_data, train_labelpath_temp)
    rf_predictions = model.predict(test_source_data)

    for ind,i in enumerate(rf_predictions):
        if flag == '1':
            rf_predictions[ind] = test_labelpath_temp[ind] - i
        else:
            rf_predictions[ind] = i - test_labelpath_temp[ind]


    return predict_result,rf_predictions

def fina_tree_and_random(train_source_data, train_labelpath_temp,test_source_data, test_labelpath_temp,test_Tipath_temp,flag):

    tree = DecisionTreeRegressor(criterion="mse")
    tree.fit(train_source_data, train_labelpath_temp)

    predict_result = tree.predict(test_source_data).tolist()
    for ind, i in enumerate(predict_result):
        if flag == '1':
            predict_result[ind] = test_labelpath_temp[ind] - i
        else:
            predict_result[ind] = i - test_labelpath_temp[ind]
    for ind, i in enumerate(predict_result):
        predict_result[ind] = abs(i - test_Tipath_temp[ind])
    a = np.array(predict_result)
    b = a * a
    tree_mse = np.sum(a * a) / a.shape[0]
    tree_err = np.mean(a)

    model = RandomForestRegressor(n_estimators=100,
                                  bootstrap=True,
                                  max_features='sqrt',
                                  criterion="mse")

    model.fit(train_source_data, train_labelpath_temp)
    rf_predictions = model.predict(test_source_data)

    for ind, i in enumerate(rf_predictions):
        if flag == '1':
            rf_predictions[ind] = test_labelpath_temp[ind] - i
        else:
            rf_predictions[ind] = i - test_labelpath_temp[ind]
    for ind, i in enumerate(rf_predictions):
        rf_predictions[ind] = abs(i - test_Tipath_temp[ind])
    a = np.array(rf_predictions)
    mse = np.sum(a * a) / a.shape[0]
    err = np.mean(a)

    return tree_err, err, tree_mse, mse

# gerber_huber_2014_data  ihdp  criteo  twin job  MineThatData/4 file20
if __name__ == '__main__':
    dataset = 'file20'
    ###############################################################################################################################################
    train_or_test = 'train'
    treatment_or_cortonl = 'treatment'
    train_source_data, train_labelpath_temp,train_Tipath_temp = load_data(train_or_test, treatment_or_cortonl,dataset)

    train_or_test = 'train'
    treatment_or_cortonl = 'control'
    test_source_data, test_labelpath_temp, test_Tipath_temp = load_data(train_or_test, treatment_or_cortonl,dataset)

    train_or_test = 'test'
    treatment_or_cortonl = 'control'
    M_train_source_data, M_train_labelpath_temp, M_train_Tipath_temp = load_data(train_or_test, treatment_or_cortonl,
                                                                               dataset)

    train_or_test = 'test'
    treatment_or_cortonl = 'treatment'
    M_test_source_data, M_test_labelpath_temp, M_test_Tipath_temp = load_data(train_or_test, treatment_or_cortonl,
                                                                            dataset)

    predict_result0,rf_predictions0 = tree_and_random(train_source_data,train_labelpath_temp,test_source_data,test_labelpath_temp,'0')
    predict_result1, rf_predictions1 = tree_and_random(test_source_data,test_labelpath_temp,train_source_data, train_labelpath_temp, '1')

    tree_err1, random_err1,tree_mse1,random_mse1 = fina_tree_and_random(train_source_data, predict_result1,M_train_source_data, M_train_labelpath_temp, M_train_Tipath_temp,'1')
    rf_tree_err1, rf_random_err1, rf_tree_mse1, rf_random_mse1 = fina_tree_and_random(train_source_data, rf_predictions1,M_train_source_data, M_train_labelpath_temp,M_train_Tipath_temp, '1')

    tree_err0, random_err0, tree_mse0, random_mse0 = fina_tree_and_random(test_source_data, predict_result0,M_test_source_data, M_test_labelpath_temp,M_test_Tipath_temp, '0')
    rf_tree_err0, rf_random_err0, rf_tree_mse0, rf_random_mse0 = fina_tree_and_random(test_source_data,rf_predictions0,M_test_source_data,M_test_labelpath_temp,M_test_Tipath_temp, '0')


    print('test:')
    print('RT  err:' + str(0.8*tree_err0 + 0.2*tree_err1))
    print('RF  err:' + str(0.8*random_err0 + 0.2*random_err1))
    print('RT  mse:' + str(0.8 * tree_mse0 + 0.2 * tree_mse1))
    print('RF  mse:' + str(0.8 * random_mse0 + 0.2 * random_mse1))

    print('\n\n')
