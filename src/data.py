import numpy as np 
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.utils import to_categorical

def data_prep(nivel_ruido):
    """Pré processa os dados, realizando todos os processos de resize, normalização e ruído. Adequadamente para MLP e CNN

    parametros: nivel_ruido: float
        Nível de ruído gaussiano a ser adicionado aos dados (ex: 0.5)
    return:
        x_train_mlp: Dados de treino para MLP
        x_test_mlp:  Dados de teste para MLP
        x_train_cnn: Dados de treino para CNN
        x_test_cnn: Dados de teste para CNN
        y_train: Labels de treino
        y_test: Labels de teste

    """

    # Carregar os dados
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    #Normalização para escala [0.1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    #Ruido gaussiano
    x_train_ruido = x_train + nivel_ruido * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_ruido = x_test + nivel_ruido * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

    x_train_ruido = np.clip(x_train_ruido, 0.0, 1.0)
    x_test_ruido = np.clip(x_test_ruido, 0.0, 1.0)

    #MLP: utiliza array 28x28
    x_train_mlp = x_train_ruido.reshape(len(x_train), 784)
    x_test_mlp = x_test_ruido.reshape(len(x_test), 784)

    #CNN
    x_train_cnn = x_train_ruido.reshape(-1, 28, 28, 1)
    x_test_cnn = x_test_ruido.reshape(-1, 28, 28, 1)

    return x_train_mlp, x_test_mlp, x_train_cnn, x_test_cnn, y_train, y_test

if __name__ == "__main__":
    x_train_mlp, x_test_mlp, x_train_cnn, x_test_cnn, y_train, y_test = data_prep(0.5)
    print("Shapes:")
    print("x_train_mlp:", x_train_mlp.shape)
    print("x_test_mlp:", x_test_mlp.shape)
    print("x_train_cnn:", x_train_cnn.shape)
    print("x_test_cnn:", x_test_cnn.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)