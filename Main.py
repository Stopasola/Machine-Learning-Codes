from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def main():
    amostras = []

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

    knn = KNeighborsClassifier(n_neighbors=17, p=2)
    knn.fit(x_treino, y_treino)
    label1 = knn.predict(x_validacao)
    #print(np.sum(label1 == y_validacao))
    print(100*(label1 == y_validacao).sum() /len(x_validacao))
    label2 = knn.predict(x_teste)
    #print(np.sum(label2 == y_teste))
    print(100*(label2 == y_teste).sum() / len(x_validacao))



if __name__ == "__main__":
    main()
