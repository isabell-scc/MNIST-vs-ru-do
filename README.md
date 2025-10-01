# Projeto de Robustez: Comparação MLP vs. CNN com Ruído Gaussiano (MNIST)

## 1. Visão Geral do Projeto

Este projeto tem o objetivo de comparar a robustez de duas arquiteturas de Redes Neurais Artificiais na classificação da base de dados MNIST quando expostas a diferentes níveis de ruído gaussiano ($\sigma$).

A análise visa validar a seguinte hipótese:

> A arquitetura baseada em filtros da CNN (Rede Neural Convolucional) confere maior invariância espacial e, portanto, será significativamente mais robusta a perturbações de ruído do que a MLP (Perceptron Multicamadas).

---

## 2. Estrutura do Experimento

O experimento é desenhado para avaliar o desempenho médio e a variabilidade de cada modelo em cenários de estresse, conforme configurado em `src/train.py`.

| Parâmetro | Valor | Objetivo |
| :--- | :--- | :--- |
| **Níveis de Ruído ($\sigma$):** | `[0.0, 0.2, 0.5]` | Testar modelos em dados **limpos** (0.0), **levemente corrompidos** (0.2) e **altamente corrompidos** (0.5). |
| **Seeds de Repetição:** | `[42, 12, 8]` | Cada cenário é executado três vezes para calcular a **Acurácia Média** ($\mu$) e o **Desvio Padrão ($\pm$ std)**, garantindo validade estatística. |
| **Otimização:** | `epochs=20`, `batch_size=256` | Treinamento controlado com *Early Stopping* para otimizar a generalização. |

---

## 3. Instalação e Configuração

Para replicar os experimentos, siga os seguintes passos:


```bash
pip install -r requirements.txt
```

```bash
python src/train.py


