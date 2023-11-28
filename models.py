import numpy as np
import pandas as pd
import pickle
import inquirer
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix

dataset = pd.read_csv("data.csv")
X = dataset.iloc[:, 2:-1].values
y = dataset.iloc[:, 1].values

# encoding strings to nums
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

# split dataset into the Training and Test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# scaling
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)

# choose model


def selector(question, choices):
    questions = [
        inquirer.List('answ',
                      message=question,
                      choices=choices)]
    answer = inquirer.prompt(questions)
    return answer


def SaveFile(file, name):
    save = [scale, file]
    pickle.dump(save, open(f'saved models/{name}', 'wb'))


def Regressor(model):
    if model == 'Linear Regression':
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    elif model == "SVR":
        from sklearn.svm import SVR
        kernel = selector('Choose a kernel: ', ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'])
        ker = kernel['answ']
        return SVR(kernel=ker)
    elif model == "Decision Tree(Regression)":
        from sklearn.tree import DecisionTreeRegressor
        return DecisionTreeRegressor()
    elif model == "Random Forest(Regression)":
        from sklearn.ensemble import RandomForestRegressor
        n = 0
        while n <= 0:
            n = int(input("Choose a positive integer for estimators: "))
        return RandomForestRegressor(n_estimators=n)


def Classification(model):
    if model == 'Logistic Regression':
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression()
    elif model == 'K-NN':
        from sklearn.neighbors import KNeighborsClassifier
        n = 0
        p = 0
        while n <= 0:
            n = int(input("Choose a positive integer for neighbors: "))
            choose_metric = selector('Choose a metric : ', ['minkowski', 'cityblock', 'cosine', 'euclidean',
                                                            'haversine', 'l1', 'l2', 'manhattan', 'nan_euclidean'])
        metric = choose_metric['answ']
        return KNeighborsClassifier(n_neighbors=n, metric=metric)
    elif model == 'SVC':
        from sklearn.svm import SVC
        kernel = selector('Choose a kernel: ', ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'])
        ker = kernel['answ']
        return SVC(kernel=ker)
    elif model == 'Naive Bayes':
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()
    elif model == 'Decision Tree(Classification)':
        from sklearn.tree import DecisionTreeClassifier
        criterion_choose = selector('Select criterion : ', ['gini', 'entropy', 'log_loss'])
        criterion = criterion_choose['answ']
        return DecisionTreeClassifier(criterion=criterion)
    elif model == 'Random Forest(Classification)':
        from sklearn.ensemble import RandomForestClassifier
        n = 0
        while n <= 0:
            n = int(input("Choose a positive integer for estimators: "))
        criterion_choose = selector('Select criterion : ', ['gini', 'entropy', 'log_loss'])
        criterion = criterion_choose['answ']
        return RandomForestClassifier(n_estimators=n, criterion='entropy', random_state=0)
    elif model == 'XGBoost':
        from xgboost import XGBClassifier
        return XGBClassifier()


def ANN(layers, units, activation, optimizer):
    import tensorflow as tf
    ann = tf.keras.models.Sequential()
    for l in range(layers):
        ann.add(tf.keras.layers.Dense(units=units[l], activation=activation[l]))
    ann.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return ann


method_select = selector('Select a method: ', ['Regression', 'Classification', 'Artificial Neural Network'])

if method_select['answ'] == 'Regression':
    model_reg = selector('Select a model', ['Linear Regression', 'SVR', 'Decision Tree(Regression)',
                                            'Random Forest(Regression)'])
    model_name = model_reg['answ']
    model = Regressor(model_name)
elif method_select['answ'] == 'Classification':
    model_clas = selector('Choose a model',
                          ['Logistic Regression', 'K-NN', 'SVC', 'Naive Bayes', 'Decision Tree(Classification)',
                           'Random Forest(Classification)', 'XGBoost'])
    model_name = model_clas['answ']
    model = Classification(model_name)

if method_select['answ'] == 'Artificial Neural Network':
    model_name = 'ANN'
    build = selector('', ['Build your own', 'Recommended'])
    if build['answ'] == 'Recommended':
        model = ANN(3, [6, 6, 1], ['relu', 'relu', 'sigmoid'], 'adam')
        batch_size = 32
        epochs = 100
    else:
        lay = int(input('Select number of layers: '))
        units = []
        activation = []
        for i in range(lay):
            unit_num = int(input(f'Select number of units for layer {i + 1}: '))
            act_fun = selector(f'Select an activation function for layer {i + 1} :',
                               ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu',
                                'expotential'])
            units.append(unit_num)
            activation.append(act_fun['answ'])
        optimizer_sel = selector('Select optimizer: ', ['SGD', 'RMSprop', 'adam', 'adagrad', 'adamax', 'adadelta'])
        optimizer = optimizer_sel['answ']
        batch_size = int(input('Select batch size: '))
        epochs = int(input('Select number of epochs: '))
        model = ANN(lay, units, activation, optimizer)

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
else:
    model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred = y_pred.round()
print("Predictions vs Real values")
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
r2 = r2_score(y_pred, y_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f'R_2 = {r2}')
print(f'Accuracy Score = {accuracy}')
print(f'Confusion Matrix : {cm}')

confirm = {
    inquirer.Confirm('confirmed',
                     message="Do you want to save the model? ",
                     default=True),
}
confirmation = inquirer.prompt(confirm)
print(confirmation["confirmed"])
if confirmation["confirmed"]:
    SaveFile(model, model_name)
    print('Model Saved')
