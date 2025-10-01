import numpy as np 
import tensorflow as tf 
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping 
from src.data import data_prep
from src.models import mlp_model, cnn_model
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist


def run(model, seeds, ruido, epochs, batch_size):       
    """
    Treina MLP/CNN em diferentes seeds, usando dados limpos para treino e dados ruidosos para teste.
    Retorna uma lista de acurácias e uma lista de perdas para cada seed.
    """
    all_acc = []
    all_loss = []
    CLASSES = 10

    #carregar dados LIMPOS
    X_train_mlp, _, X_train_cnn, _, y_train_inteiro, _ = data_prep(nivel_ruido=0.0)
    Y_train_ohe = keras.utils.to_categorical(y_train_inteiro, num_classes=CLASSES)

    #dados RUIDOSOS
    _, X_test_mlp_noisy, _, X_test_cnn_noisy, _, y_test_inteiro = data_prep(nivel_ruido=ruido)
    Y_test_ohe_noisy = keras.utils.to_categorical(y_test_inteiro, num_classes=CLASSES)

    if model == 'mlp':
        X_train = X_train_mlp
        X_test = X_test_mlp_noisy
        modelo_final = mlp_model

    else:
        X_train = X_train_cnn
        X_test = X_test_cnn_noisy
        modelo_final = cnn_model


    for seed in seeds:
        np.random.seed(seed)
        keras.utils.set_random_seed(seed)

        model = modelo_final()
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(X_train, Y_train_ohe,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(X_test, Y_test_ohe_noisy), 
                            callbacks=[early_stopping])
        
        
        loss, acc = model.evaluate(X_test, Y_test_ohe_noisy, verbose=0)
        all_acc.append(acc)
        all_loss.append(loss)

        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)

        print(f"Seed {seed}: Acurácia (Ruído {ruido}) = {acc:.4f} Loss{loss:.4f}")

    #metricas com media ± std
    metrics = {
        'acuracia': (np.mean(all_acc), np.std(all_acc)),
        'loss': (np.mean(all_loss), np.std(all_loss))
    }

    return all_acc, all_loss, metrics


if __name__ == "__main__":
        # Definir seeds e ruído
    seeds = [42, 12, 8]
    ruido = [0.0, 0.2, 0.5]
    epochs = 20
    batch_size = 128
    all_results = []

    for r in ruido:
        print(f"\n--- Experimento com ruído {r} ---")
        # Rodar experimento para MLP
        print("Experimento MLP ")
        mlp_acc, mlp_loss, mlp_metrics = run(model='mlp', seeds=seeds, ruido=r, epochs=epochs, batch_size=batch_size)
        all_results.append({
            'ruido': r,
            'modelo': 'mlp',
            'acuracia_media': mlp_metrics['acuracia'][0],
            'acuracia_std': mlp_metrics['acuracia'][1],
            'loss_media': mlp_metrics['loss'][0],
            'loss_std': mlp_metrics['loss'][1],
            'mlp_acuracia': mlp_acc,
        })

        # Rodar experimento para CNN
        print("\nExperimento CNN")
        cnn_acc, cnn_loss, cnn_metrics = run(model='cnn', seeds=seeds, ruido=r, epochs=epochs, batch_size=batch_size)
        all_results.append({
            'ruido': r,
            'modelo': 'cnn',
            'acuracia_media': cnn_metrics['acuracia'][0],
            'acuracia_std': cnn_metrics['acuracia'][1],
            'loss_media': cnn_metrics['loss'][0],
            'loss_std': cnn_metrics['loss'][1],
            'cnn_acuracia': cnn_acc,
        })


        tabela = pd.DataFrame({
            'Métrica': ['Acurácia', 'Loss'],
            'MLP (média ± std)': [f"{mlp_metrics['acuracia'][0]:.4f} ± {mlp_metrics['acuracia'][1]:.4f}",
                                f"{mlp_metrics['loss'][0]:.4f} ± {mlp_metrics['loss'][1]:.4f}"],
            'CNN (média ± std)': [f"{cnn_metrics['acuracia'][0]:.4f} ± {cnn_metrics['acuracia'][1]:.4f}",
                                f"{cnn_metrics['loss'][0]:.4f} ± {cnn_metrics['loss'][1]:.4f}"]
        })

        print("\nTabela MLP vs CNN")
        print(f"MLP com ruído {r}: Acurácia {mlp_metrics['acuracia'][0]:.4f} +/- {mlp_metrics['acuracia'][1]:.4f}")
        print(f"CNN com ruído {r}: Acurácia {cnn_metrics['acuracia'][0]:.4f} +/- {cnn_metrics['acuracia'][1]:.4f}")


    summary_df = pd.DataFrame(all_results)
    print("Tabela MLP vs CNN")
    print(summary_df)

    # Salvar resultados em CSV
    summary_df.to_csv('resultados_experimento.csv', index=False)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, cellLoc='center', loc='center')
    table.scale(1, 2)
    plt.title('Resultados MLP vs CNN', fontsize=14)
    plt.savefig('resultados_experimento.png', bbox_inches='tight')