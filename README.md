# Sistema de Inspeção Visual de Comprimidos e Cápsulas por PDI (MVTec AD)

Projeto desenvolvido em Python utilizando **NumPy**, **OpenCV** e **Matplotlib** para inspeção visual automática de defeitos em **comprimidos (`pill`)** e **cápsulas (`capsule`)** utilizando o benchmark público **MVTec AD** (`foersben/mvtec-ad` no Hugging Face).

> **Diferencial (Regra 10 — Solução do Zero):** Todos os algoritmos centrais do pipeline — conversão RGB→HSV, filtro da mediana, filtro Gaussiano por convolução 2D, limiarização automática de Otsu, detector de bordas de Canny (Sobel + Supressão Não-Máxima vetorizada + Histerese), morfologia matemática binária (erosão, dilatação, abertura, fechamento) e transformada morfológica **Black-Hat** em escala de cinza — foram implementados **do zero apenas com NumPy** (`src/pdi_scratch.py`), com comparação direta contra o **OpenCV** (`src/pdi_opencv.py`).

---

## 1. Contextualização do Problema

Na indústria farmacêutica, o controle de qualidade visual é crítico para evitar que comprimidos ou cápsulas com rachaduras, manchas, contaminações, impressões defeituosas, quebras ou deformações cheguem aos pacientes. A inspeção manual é lenta e sujeita à fadiga visual. Este projeto implementa um pipeline clássico, determinístico e explicável de **Processamento Digital de Imagens (PDI)** para automatizar a inspeção e classificar cada item como **APROVADO** ou **REPROVADO**.

---

## 2. Base de Dados (MVTec AD — `pill` e `capsule`)

- **Origem:** Repositório público [foersben/mvtec-ad](https://huggingface.co/datasets/foersben/mvtec-ad/tree/14f50140550afeccd97ad31d4cf68fbf9bb94e37) (revisão `14f50140550afeccd97ad31d4cf68fbf9bb94e37`), categorias `pill` e `capsule`.
- **Download automático:** `src/download_mvtec.py` baixa automaticamente `train/good` (30 imagens para calibração de referência por categoria) e todo o conjunto `test/*`.
- **Características das duas bases:**
  - **`pill` (Comprimidos):** 167 imagens de teste (`800 × 800` px, PNG, RGB, fundo escuro) em 8 classes: `good` (26), `color` (25), `combined` (17), `contamination` (21), `crack` (26), `faulty_imprint` (19), `pill_type` (9) e `scratch` (24).
  - **`capsule` (Cápsulas):** 132 imagens de teste (`1000 × 1000` px, PNG, RGB, fundo claro) em 6 classes: `good` (23), `crack` (23), `faulty_imprint` (22), `poke` (21), `scratch` (23) e `squeeze` (20).

---

## 3. Metodologia (Pipeline de 8 Etapas)

1. **Aquisição e Padronização:** Leitura da imagem BGR, redimensionamento para `320 × 320` px e conversão para RGB.
2. **Filtragem:** Filtro da **Mediana (3×3)** seguido de **Filtro Gaussiano (5×5, σ = 1.0)** para redução de ruído preservando bordas.
3. **Conversão de Cores:** Conversão para **Escala de Cinza** e espaço **HSV**.
4. **Segmentação com Polaridade Automática:** **Limiarização de Otsu** com detecção automática de fundo claro/escuro (permitindo operar tanto em `pill` quanto em `capsule`), seguida de **Abertura (5×5)**, **Fechamento (7×7)** e seleção do maior componente conectado.
5. **Bordas e Morfologia em Cinza:** **Detector de Canny** e **Transformada Morfológica Black-Hat (11×11)** para destacar fissuras, riscos e defeitos superficiais.
6. **Calibração Estatística (`train/good`):** Divisão *fit/val* (20 imagens para estimar distribuição de cor e template mediano alinhado + 10 imagens de validação para calibrar limiares sem vazamento de dados).
7. **Extração de Características e Classificação:**
   - *Fragmentação / Deformação:* Solidez e razão de aspecto do contorno.
   - *Manchas / Alteração de cor:* Distância de **Mahalanobis** no espaço `[R, G, B, S]` contra pixels normais.
   - *Rachaduras / Riscos / Falhas superficiais:* Black-Hat, densidade de bordas internas (Canny) e mapa de resíduo normalizado frente ao template de referência.
8. **Visualização e Métricas:** Geração automática de painéis 2×3 das etapas intermediárias, matrizes de confusão e comparativo de Acurácia, Precisão, Recall, F1-Score e tempo médio por imagem.

---

## 4. Resultados Obtidos (MVTec AD)

| Base | Backend | Acurácia | Precisão | Recall | F1-Score | Tempo médio/img |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MVTec `pill`** (167 imgs) | **NumPy (Do Zero)** | **74,25%** | **88,89%** | **79,43%** | **83,90%** | 227,10 ms |
| **MVTec `pill`** (167 imgs) | **OpenCV** | 73,65% | 88,80% | 78,72% | 83,46% | 7,06 ms |
| **MVTec `capsule`** (132 imgs) | **NumPy (Do Zero)** | **49,24%** | **85,00%** | **46,79%** | **60,36%** | 226,56 ms |
| **MVTec `capsule`** (132 imgs) | **OpenCV** | 48,48% | 87,27% | 44,04% | 58,54% | 5,85 ms |

### Análise Crítica
- **Equivalência funcional:** A implementação do zero em NumPy atingiu métricas equivalentes (e ligeiramente superiores em F1) às funções prontas do OpenCV, validando a corretude matemática dos algoritmos; o OpenCV apresentou ganho de velocidade de ~32–39× por usar rotinas compiladas em C/C++.
- **Desempenho em `pill`:** O pipeline clássico obteve **83,90% de F1-Score**, com destaque para `pill_type` (100%), `combined` (94,1%), `contamination` (90,5%) e `color` (80,0%).
- **Desafios em `capsule`:** O sistema detectou bem deformações estruturais (`squeeze`: 80,0%) e riscos maiores (`scratch`: 52,2%), mas apresentou maior dificuldade em microdefeitos como perfurações milimétricas (`poke`: 23,8%) e reflexos especulares na superfície brilhante bicolor — evidenciando os limites naturais de abordagens puramente clássicas frente a modelos profundos em defeitos submilimétricos.

---

## 5. Tutorial de Execução

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Baixar bases MVTec AD (pill e capsule) do Hugging Face (se ainda não baixadas)
python -m src.download_mvtec

# 3. Executar avaliação completa nas duas bases comparando NumPy (do zero) vs OpenCV
python main.py --dataset mvtec-all --mode compare

# Outros exemplos:
python main.py --dataset pill --mode scratch
python main.py --dataset capsule --mode opencv
python main.py --image data/mvtec/pill/test/crack/000.png --train-good data/mvtec/pill/train/good
```
