import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image
import matplotlib.cm as cm
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def get_saliency_map(model, image, class_idx):

    
    with tf.GradientTape() as tape:
        tape.watch(image)
        predictions = model(image)
        loss = predictions[:, class_idx]

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, image)
    absolute_gradient = tf.abs(gradient)

    if len(absolute_gradient.shape) == 4:
        gradient = tf.reduce_max(absolute_gradient, axis=-1)
    
    else:
        gradient = absolute_gradient

    # convert to numpy
    gradient = gradient.numpy()[0]

    # normaliz between 0 and 1
    min_val, max_val = np.min(gradient), np.max(gradient)
    smap = (gradient - min_val) / (max_val - min_val + keras.backend.epsilon())

    return smap


def plot_saliency_map(image, saliency_map, alpha=0.5):
    # Convert grayscale image to RGB
    if image.shape[-1] == 1:
        image = np.concatenate([image, image, image], axis=-1)

    # Create a color map
    colormap = cm.get_cmap('jet')

    # Apply the colormap to the saliency map
    colored_smap = colormap(saliency_map)
    colored_smap = (colored_smap[:, :, :3] * 255).astype(np.uint8)  # Drop alpha channel and convert to uint8

    # Convert to PIL images
    image_pil = Image.fromarray((image * 255).astype(np.uint8))
    smap_pil = Image.fromarray(colored_smap)

    # Blend images
    blended = Image.blend(image_pil, smap_pil, alpha=alpha)

    return blended


def plot_loss_accuracy(history):
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(f'Perda e Acurácia durante o Treinamento - CNN', fontsize=16)
    ax = fig.add_subplot(1, 2, 1)
    #Grafico de perda
    ax.plot(history.history["loss"],marker = 'o', color = 'blue', label="Perda de Treino")
    ax.plot(history.history["val_loss"],marker = 'x', color = 'purple', label="Perda de Validação")
    ax.legend()
    ax.set_title('Perda(cross_entropy)')
    ax.grid(True)


    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history.history["accuracy"],marker = 'o', color = 'blue', label="Precisão de Treino")
    ax.plot(history.history["val_accuracy"],marker = 'x', color = 'purple', label="Precisão de Validação")
    ax.legend()
    ax.set_title('Acurácia')
    ax.grid(True)
    plt.show()

    def plot_gradcam(model, img, class_idx, layer_name, title='Grad-CAM'):
        """
        Gera e plota o Grad-CAM para uma imagem e classe específica.
        """


        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(layer_name).output, model.output]
        )

        # Calcula o gradiente da classe com relação à camada
        with tf.GradientTape() as tape:
            inputs = tf.cast(img, tf.float32)
            conv_outputs, predictions = grad_model(inputs)
            loss = predictions[:, class_idx]  

        grads = tape.gradient(loss, conv_outputs)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-8)

        # Exibe o heatmap
        plt.imshow(heatmap, cmap='jet')
        plt.axis('off')
        plt.title(title)
        plt.colorbar()
        plt.show()





