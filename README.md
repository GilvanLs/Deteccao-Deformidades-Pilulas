# Sistema de Inspeção Visual de Comprimidos por Processamento Digital de Imagens (PDI)

Projeto desenvolvido em Python utilizando **NumPy**, **OpenCV** e **Matplotlib** para detecção automática de defeitos em comprimidos farmacêuticos e classificação de controle de qualidade (**APROVADO** vs **REPROVADO**).

> **Destaque (Implementação do Zero):** Todos os algoritmos centrais do pipeline — conversão RGB→HSV, filtro da mediana, filtro Gaussiano por convolução 2D, limiarização automática de Otsu, detector de bordas de Canny (Sobel + Supressão Não-Máxima + Histerese) e morfologia matemática binária (erosão, dilatação, abertura e fechamento) — foram implementados **do zero em NumPy** (`src/pdi_scratch.py`), com modo comparativo frente às funções equivalentes do **OpenCV** (`src/pdi_opencv.py`).

---

## 1. Contextualização do Problema

No setor farmacêutico, o controle de qualidade visual é fundamental para garantir a segurança, eficácia e aceitação dos medicamentos. Comprimidos com falhas físicas ou químicas — como rachaduras, manchas, alterações de coloração, fragmentação ou deformações — podem comprometer a dosagem correta ou indicar contaminação/degradação.

A inspeção manual por operadores humanos é sujeita à fadiga visual, lentidão e inconsistência. O **Processamento Digital de Imagens (PDI)** oferece uma solução automatizada, determinística e reproduzível para identificar essas irregularidades em linha de produção.

---

## 2. Base de Dados

O projeto inclui um gerador sintético controlado e reprodutível (`src/dataset_generator.py`) e também aceita qualquer base externa organizada em subpastas por classe (como a categoria `pill` do dataset público **MVTec AD**).

- **Origem:** Base sintética controlada gerada pelo próprio sistema (`data/synthetic/`)
- **Quantidade:** 60 imagens no total
- **Formato e Resolução:** PNG, `320 × 320` pixels, RGB (8 bits por canal)
- **Cenários de teste (5 classes):**
  - `aprovado/` (20 imagens): comprimidos íntegros com variações leves de posição, iluminação e ruído de sensor (Gaussiano + sal e pimenta)
  - `rachadura/` (10 imagens): fissuras e quebras na superfície
  - `mancha/` (10 imagens): alterações de cor e manchas superficiais
  - `fragmentacao/` (10 imagens): bordas quebradas ou partes ausentes
  - `deformacao/` (10 imagens): formato ou dimensões fora do padrão circular

---

## 3. Metodologia (Pipeline Implementado)

O pipeline em `src/pipeline.py` executa 8 etapas sequenciais:

1. **Aquisição da imagem:** Leitura da imagem BGR e conversão para RGB.
2. **Filtragem (Pré-processamento):**
   - Filtro da **Mediana (3×3)** para remoção de ruído impulsivo (sal e pimenta) preservando bordas.
   - Filtro **Gaussiano (5×5, σ = 1.2)** para suavização de ruído de alta frequência.
3. **Conversão de cores:** Conversão para **Escala de Cinza** (luminância) e para o espaço **HSV** (separando matiz, saturação e brilho para facilitar a detecção de manchas).
4. **Segmentação:** **Limiarização automática de Otsu** para separar o comprimido do fundo escuro.
5. **Pós-processamento Morfológico:** **Abertura (5×5)** seguida de **Fechamento (7×7)** e seleção do maior componente conectado para obter uma máscara limpa do comprimido.
6. **Detecção de bordas:** **Detector de Canny** para evidenciar fissuras e descontinuidades internas na superfície.
7. **Extração de características (`src/features.py`):**
   - *Fragmentação:* Solidez (`area / area_fecho_convexo`) e diferença morfológica entre fechamento amplo (15×15) e a máscara original.
   - *Deformação:* Razão de aspecto da caixa delimitadora rotacionada e circularidade ($4\pi A / P^2$).
   - *Manchas:* Proporção de pixels internos com saturação ou brilho discrepante no espaço HSV e desvio padrão da saturação.
   - *Rachaduras:* Densidade de bordas de Canny estritamente no interior erodido do comprimido (excluindo o contorno externo).
8. **Classificação e Visualização (`src/classifier.py`, `src/metrics.py`):** Classificação em `APROVADO` ou `REPROVADO` (com diagnóstico dos defeitos), geração de painéis visuais das etapas intermediárias e cálculo de Acurácia, Precisão, Recall, F1-Score, Matriz de Confusão e Tempo Médio por Imagem.

---

## 4. Estrutura do Projeto

```text
pill-defect-inspection/
├── main.py                      # Ponto de entrada CLI
├── requirements.txt             # Dependências (numpy, opencv-python, matplotlib)
├── README.md                    # Documentação completa e tutorial
├── docs/
│   └── roteiro_apresentacao.md  # Roteiro para slides e vídeo (máx. 10 min)
├── src/
│   ├── __init__.py
│   ├── pdi_scratch.py           # Algoritmos de PDI implementados do zero em NumPy
│   ├── pdi_opencv.py            # Wrappers equivalentes em OpenCV
│   ├── dataset_generator.py     # Gerador da base sintética de 60 imagens
│   ├── features.py              # Extração de características geométricas, HSV e bordas
│   ├── classifier.py            # Classificador interpretável e diagnóstico de defeitos
│   ├── pipeline.py              # Pipeline completo e geração de painéis visuais
│   └── metrics.py               # Métricas quantitativas e matriz de confusão
├── data/                        # Imagens de entrada (geradas automaticamente)
└── results/                     # Relatórios visuais e gráficos de métricas
```

---

## 5. Tutorial de Execução

### 5.1. Instalação das dependências
```bash
pip install -r requirements.txt
```

### 5.2. Executar avaliação completa (comparando Do Zero vs OpenCV)
```bash
python main.py --mode compare
```
Esse comando gera automaticamente a base em `data/synthetic/` (caso não exista), executa o pipeline nos dois modos (`scratch` e `opencv`), imprime todas as métricas no terminal e salva os resultados em `results/`:
- `results/metricas_comparacao.png` (matrizes de confusão + gráfico de barras de métricas e tempo)
- `results/etapas/` (painéis 2×3 com as etapas intermediárias de cada categoria)

### 5.3. Outros modos de execução
```bash
# Executar apenas com as implementações do zero (NumPy)
python main.py --mode scratch

# Executar apenas com OpenCV
python main.py --mode opencv

# Inspecionar uma única imagem e salvar painel detalhado
python main.py --image data/synthetic/rachadura/rachadura_01.png --mode scratch

# Avaliar uma base externa (ex.: subpastas good/ e defect/)
python main.py --data-dir caminho/para/sua_base --mode compare
```
