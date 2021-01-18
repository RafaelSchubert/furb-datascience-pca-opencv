# Aplicação de _Principal Component Analysis_ (PCA) em Reconhecimento de Faces
Este projeto é o trabalho final da disciplina de **Aprendizado de Máquina II: Aprendizado Não-Supervisionado** do curso de pós-graduação **Especialização em _Data Science_** (Universidade Regional de Blumenau — FURB).

## Objetivo
O objetivo deste projeto é usar a PCA para treinar um reconhecedor de faces, e identificar, tão corretamente quanto possível, os rostos dentro dum _data set_. Isto é, treinar o identificador, depois classificar novas faces de acordo com a similaridade as existentes no _data set_.

## Técnica
A PCA é uma técnica que decompõe a matriz formada pelas características dum _data set_ em seus principais autovalores e autovetores. Usando essas informações, é possível reduzir dimensionalmente o _data set_, diminuindo o número de características analisadas, e focando naquelas que melhor distinguem as classes de observações.

## Recursos
Optei por desenvolver este projeto em **C++** (padrão **C++17**), usando a IDE **Visual Studio 2019**. Também usei alguns dos recursos da biblioteca [**OpenCV**](https://opencv.org/), uma biblioteca aberta de visão computacional. Um dos requisitos do trabalho final era que tanto quanto possível do passo-a-passo da técnica empregada fosse desenvolvido à mão. Há recursos na OpenCV que facilitariam o desenvolvimento deste trabalho. No entanto, da forma como foi feito, pode-se observar melhor como a técnica funciona.

## Executando
O executável compilado está disponível em `./bin/Release-x64/pca_face_detection.exe`. Deve-se executá-lo por linha de comando da seguinte forma:

```
$ pca_face_detection.exe <caminho-data-set>
```

Onde `<caminho-data-set>` é um parâmetro que indica o diretório onde se encontra um _data set_ formado por arquivos `.JPG`, com nomes no formato `<ID-image>_<ID-face>.jpg` (por exemplo, `12_3.jpg` para a imagem de ID 12, que pertence ao rosto de ID 3). Um _data set_ assim está disponível em `./data/data-set.zip`.
