from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn import svm
import numpy as np

def main():
    #amostras = []

    '''with open('Vehicle.txt', 'r') as f:
        for linha in f.readlines():
            atrib = linha.replace('\n', '').split(',')
            amostras.append([int(atrib[0]), int(atrib[1]),
                             int(atrib[2]), int(atrib[3]),
                             int(atrib[4]), int(atrib[5]),
                             int(atrib[6]), int(atrib[7]),
                             int(atrib[8]), int(atrib[9]),
                             int(atrib[10]), int(atrib[11]),
                             int(atrib[12]), int(atrib[13]),
                             int(atrib[14]), int(atrib[15]),
                             int(atrib[16]), int(atrib[17]), (atrib[18])])'''

    carac = np.genfromtxt('vehicle.txt', delimiter=',', usecols=np.arange(0,18))
    classe = np.genfromtxt('vehicle.txt', delimiter=',', dtype="str", usecols=(18))

    print(carac)
    print('\n\n')
    print(classe)

    x_treino, x_test, y_treino, y_test = train_test_split(carac, classe, test_size=0.5, stratify=classe)

    x_validacao, x_teste, y_validacao, y_teste = train_test_split(x_test, y_test, test_size=0.5, stratify=y_test)

    #knn-----------------------------------------------------
    knn = KNeighborsClassifier(n_neighbors=17, p=2)
    knn.fit(x_treino, y_treino)
    labelknn1 = knn.predict(x_validacao)
    #print(np.sum(label1 == y_validacao))
    print('Validacao kNN: {}'.format(100*(labelknn1 == y_validacao).sum() /len(x_validacao)))
    labelknn2 = knn.predict(x_teste)
    #print(np.sum(label2 == y_teste))
    print('Teste kNN: {}'.format(100*(labelknn2 == y_teste).sum() / len(x_validacao)))
    print('accuracy: {0:.2f}%'.format(accuracy_score(y_teste, labelknn2) * 100))

    #Perceptron----------------------------------------------
    #random_state = 0

    #n_iter = 40
    #eta0 = 0.1 #mesmo que 'eta' (taxa de aprendizado)

    prc = Perceptron(n_iter = 40, eta0 = 0.1, random_state = 0)
    prc.fit(x_treino, y_treino)

    labelperc1 = prc.predict(x_validacao)
    print('Validacao Perceptron {}'.format(100*(labelperc1 == y_validacao).sum() /len(x_validacao)))
    labelperc2 = prc.predict(x_teste)
    print('Teste Perceptron {}'.format(100*(labelperc2 == y_teste).sum() /len(x_validacao)))
    print('accuracy: {0:.2f}%'.format(accuracy_score(y_teste, labelperc2) * 100))

    #Arvore de decisão---------------------------------------
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)
    clf.fit(x_treino, y_treino)
    labeltree1 = clf.predict(x_validacao)
    print('Validacao Arvore {}'.format(100*(labeltree1 == y_validacao).sum() /len(x_validacao)))
    labeltree2 = clf.predict(x_teste)
    print('Teste Arvore {}'.format(100*(labeltree2 == y_teste).sum() /len(x_validacao)))
    print('accuracy: {0:.2f}%'.format(accuracy_score(y_teste, labeltree2) * 100))

    #Naive Bayes---------------------------------------------
    nv = GaussianNB()
    nv.fit(x_treino, y_treino)
    labelnaive1 = nv.predict(x_validacao)
    print('Validacao NB {}'.format(100*(labelnaive1 == y_validacao).sum() /len(x_validacao)))
    labelnaive2 = nv.predict(x_teste)
    print('Teste NB {}'.format(100*(labelnaive2 == y_teste).sum() /len(x_validacao)))
    print('accuracy: {0:.2f}%'.format(accuracy_score(y_teste, labelnaive2) * 100))

    #Support Vector Machine----------------------------------------------
    supvm = svm.SVC(kernel='linear', C=1).fit(x_treino, y_treino)
    labelsvm1 = supvm.predict(x_validacao)
    print('Validacao SVM {}'.format(100*(labelsvm1 == y_validacao).sum() /len(x_validacao)))
    labelsvm2 = supvm.predict(x_teste)
    print('Teste SVM {}'.format(100*(labelsvm2 == y_teste).sum() /len(x_validacao)))
    print('accuracy: {0:.2f}%'.format(accuracy_score(y_teste, labelsvm2) * 100))

    #Multilayer Perceptron---------------------------------------------
    mlp = MLPClassifier(hidden_layer_sizes=(13,13,13), max_iter=500)
    mlp.fit(x_treino, y_treino)
    labelmlp1 = mlp.predict(x_validacao)
    print('Validacao MLP {}'.format(100*(labelmlp1 == y_validacao).sum() /len(x_validacao)))
    labelmlp2 = mlp.predict(x_teste)
    print('Teste MLP {}'.format(100*(labelmlp2 == y_teste).sum() /len(x_validacao)))
    print('accuracy: {0:.2f}%'.format(accuracy_score(y_teste, labelmlp2) * 100))


if __name__ == "__main__":
    main()
