# Segmentador de Mexilhões
Este código recebe uma imagem de mexilhões, realiza a segmentação individual deles e faz uma estimativa do número de mexilhões na imagem (incluindo os mexilhÕes ocultos).

## Requisitos
- GPU dedicada para rodar as redes neurais.

## Instalação (Linux)
1. Instale o Python3
2. Baixe este repositório :
```bash
git clone https://github.com/alexmorimitsu/MexilhaoSeg.git
```
3. Entre na pasta baixada:
```bash
cd MexilhaoSeg
```
4. Instale as biblioteca através do comando
```bash
pip install -r requirements.txt
```
5. Faça o *download* do SAM pelo *link* https://github.com/facebookresearch/segment-anything
   5.1. Salve o arquivo dentro da pasta Modelos

## Execução do código
Para rodar a segmentação, adicione as imagens dentro da pasta Imagens e executar o comando:
```bash
python3 main.py --input_folder Imagens --yolo_model Modelos/yolo_mexilhoes.pt
```
Uma pasta *outputs* será criada com os resultados das segmentações estimativas de contagem.

## Exemplo de instalação e execução no Google Colab
Explicações de como instalar e executar este código no Google Colab podem ser encontrados no link https://colab.research.google.com/drive/1SZ4I14FfFAz9zzDoKNsekAZZHbzQ9-wO?usp=sharing

## Licença

Este projeto é de código aberto e está sob a licença **AGPL 3.0**.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.
