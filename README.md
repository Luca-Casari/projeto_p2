# SmartDigit KNN 

Aplicação web desenvolvida com Streamlit para reconhecimento de dígitos manuscritos utilizando Machine Learning. O usuário pode desenhar um número de 0 a 9 diretamente na tela, e o sistema realiza a classificação em tempo real utilizando um modelo K-Nearest Neighbors (KNN) treinado sobre imagens de dígitos.

## Aplicação Online

Acesse a aplicação:

https://projetop2-8pkdf6watuehhxkzuukrwn.streamlit.app/

---

# 1. Sobre o Projeto

O SmartDigit KNN é um projeto de Machine Learning aplicado ao reconhecimento de padrões em imagens. Seu objetivo é identificar dígitos manuscritos desenhados pelo usuário e classificá-los automaticamente entre os números de 0 a 9.

O projeto demonstra um pipeline completo de Ciência de Dados, incluindo pré-processamento de imagens, redução de dimensionalidade com PCA, treinamento de modelo supervisionado e implantação de uma interface web interativa utilizando Streamlit.

---

# 2. Objetivo

### Objetivo Geral

Desenvolver um sistema capaz de reconhecer dígitos manuscritos desenhados pelo usuário através de técnicas de Machine Learning.

### Objetivos Específicos

* Realizar o pré-processamento de imagens manuscritas.
* Aplicar redução de dimensionalidade utilizando PCA.
* Treinar um modelo K-Nearest Neighbors (KNN).
* Disponibilizar o modelo em uma aplicação web interativa.
* Permitir a classificação em tempo real de números desenhados pelo usuário.

---

# 3. Dataset

O projeto utiliza o dataset MNIST, um dos conjuntos de dados mais conhecidos para reconhecimento de dígitos manuscritos.

### Características

| Característica              | Valor                     |
| --------------------------- | ------------------------- |
| Quantidade de Classes       | 10                        |
| Classes                     | 0 a 9                     |
| Tamanho das Imagens         | 28 x 28 pixels            |
| Total de Features Originais | 784                       |
| Tipo de Problema            | Classificação Multiclasse |

### Fonte

MNIST Database of Handwritten Digits.

---

# 4. Estrutura do Projeto

```text
SmartDigit-KNN/
│
├── app.py
├── requirements.txt
├── README.md
│
├── modelo/
│   ├── knn_model.pkl
│   └── pca.pkl
│
└── imagens/
    └── exemplo.png
```

### Arquivos principais

| Arquivo          | Descrição                                           |
| ---------------- | --------------------------------------------------- |
| app.py           | Aplicação Streamlit                                 |
| knn_model.pkl    | Modelo KNN treinado                                 |
| pca.pkl          | Modelo PCA utilizado na redução de dimensionalidade |
| requirements.txt | Dependências do projeto                             |

---

# 5. Variável Alvo

A variável alvo corresponde ao dígito representado na imagem.

| Variável | Descrição                     |
| -------- | ----------------------------- |
| label    | Número manuscrito entre 0 e 9 |

---

# 6. Variáveis Preditoras

As variáveis de entrada são os pixels da imagem.

| Tipo             | Quantidade |
| ---------------- | ---------- |
| Pixels da imagem | 784        |

Cada pixel possui intensidade variando entre 0 e 255.

Após o PCA, as imagens passam a ser representadas por 50 componentes principais.

---

# 7. Metodologia

O pipeline do projeto segue as seguintes etapas:

```text
Imagem desenhada pelo usuário
            │
            ▼
1. Conversão para escala de cinza
            │
            ▼
2. Binarização da imagem
            │
            ▼
3. Recorte automático do dígito
            │
            ▼
4. Redimensionamento para 28x28
            │
            ▼
5. Centralização do dígito
            │
            ▼
6. Normalização dos pixels
            │
            ▼
7. Aplicação do PCA
            │
            ▼
8. Predição com KNN
            │
            ▼
9. Exibição do resultado
```

---

# 8. Modelo Utilizado

### K-Nearest Neighbors (KNN)

O algoritmo KNN realiza a classificação analisando os exemplos mais próximos da nova amostra.

Características:

* Aprendizado supervisionado.
* Simples implementação.
* Excelente desempenho para reconhecimento de padrões.
* Muito utilizado em problemas de classificação de imagens.

---

# 9. Redução de Dimensionalidade

Foi aplicada a técnica PCA (Principal Component Analysis) para reduzir a dimensionalidade dos dados.

### Antes do PCA

* 784 atributos (28 × 28 pixels)

### Após o PCA

* 50 componentes principais

Benefícios:

* Redução do custo computacional.
* Menor tempo de inferência.
* Remoção de ruídos.
* Preservação das características mais relevantes da imagem.

---

# 10. Funcionamento da Aplicação

A interface possui:

### Área de Desenho

O usuário desenha um número utilizando o mouse.

### Botão Classificar

Após o desenho, o sistema processa a imagem e envia os dados ao modelo treinado.

### Resultado

A aplicação retorna:

* Dígito previsto.
* Nível de confiança da predição.
* Imagem processada enviada ao modelo.

---

# 11. Tecnologias Utilizadas

| Categoria                | Tecnologia   |
| ------------------------ | ------------ |
| Linguagem                | Python       |
| Interface                | Streamlit    |
| Machine Learning         | Scikit-Learn |
| Deep Learning Utils      | PyTorch      |
| Manipulação Numérica     | NumPy        |
| Processamento de Imagem  | Pillow       |
| Persistência de Modelos  | Joblib       |
| Processamento Científico | SciPy        |

---

# 12. Como Executar o Projeto

## 1. Clonar o Repositório

```bash
git clone https://github.com/seu-usuario/smartdigit-knn.git

cd smartdigit-knn
```

## 2. Criar Ambiente Virtual

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

## 3. Instalar Dependências

```bash
pip install -r requirements.txt
```

## 4. Executar a Aplicação

```bash
streamlit run app.py
```

A aplicação ficará disponível em:

```text
http://localhost:8501
```

---

# 13. Conclusão

O SmartDigit KNN atingiu seu objetivo ao implementar um sistema capaz de reconhecer dígitos manuscritos em tempo real através de técnicas de Machine Learning.

O projeto permitiu aplicar conceitos fundamentais de Ciência de Dados e Aprendizado de Máquina, incluindo processamento de imagens, redução de dimensionalidade com PCA, treinamento supervisionado utilizando KNN e implantação de modelos em aplicações web.

Como melhorias futuras, podem ser explorados modelos mais avançados, como Redes Neurais Convolucionais (CNNs), além da comparação de desempenho com outros algoritmos de classificação.
