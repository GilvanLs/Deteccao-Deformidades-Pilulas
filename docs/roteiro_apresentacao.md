# Roteiro de Apresentação e Vídeo (Máximo 10 minutos)

Estrutura alinhada às regras (12) e (13) utilizando as bases reais **MVTec AD (`pill` e `capsule`)**.

---

## Slide 1 — Capa (0:00 – 0:30)
- **Título:** Inspeção Visual de Comprimidos e Cápsulas por Processamento Digital de Imagens
- **Subtítulo:** Detecção de Defeitos no Benchmark MVTec AD (Implementação do Zero em NumPy + OpenCV)
- **Integrantes / Disciplina / Data**

## Slide 2 — Introdução e Contextualização (0:30 – 1:45)
- Importância do controle de qualidade na indústria farmacêutica.
- Limitações da inspeção visual manual (fadiga, subjetividade, velocidade).
- Objetivo: detectar automaticamente defeitos (rachaduras, manchas, contaminações, deformações, riscos e falhas de impressão) e classificar em **APROVADO** ou **REPROVADO**.

## Slide 3 — Base de Dados: MVTec AD (`pill` e `capsule`) (1:45 – 2:45)
- Origem: repositório `foersben/mvtec-ad` no Hugging Face.
- **`pill`:** 167 imagens de teste (`800×800` PNG, fundo escuro, 8 classes) + 30 `train/good` para calibração.
- **`capsule`:** 132 imagens de teste (`1000×1000` PNG, fundo claro, 6 classes) + 30 `train/good` para calibração.
- Destacar o desafio de lidar com **dois cenários opostos** (fundo escuro vs fundo claro, superfície fosca vs brilhante bicolor).

## Slide 4 — Fundamentação Teórica e Solução do Zero (2:45 – 4:15)
- Destacar a **Regra 10**: algoritmos implementados **do zero em NumPy** (`src/pdi_scratch.py`) e comparados ao OpenCV (`src/pdi_opencv.py`):
  - Conversão RGB→HSV e Escala de Cinza
  - Filtro da Mediana (3×3) e Filtro Gaussiano por Convolução 2D (5×5)
  - Limiarização de Otsu com detecção automática de polaridade de fundo
  - Morfologia Matemática (Abertura, Fechamento e **Black-Hat** em escala de cinza)
  - Detector de Bordas de Canny (Sobel + NMS vetorizado + Histerese)

## Slide 5 — Metodologia, Calibração e Extração de Características (4:15 – 5:45)
- Pipeline de 8 etapas (`src/pipeline.py`).
- Calibração estatística *fit/val* (20/10 em `train/good`) sem vazamento de dados.
- Características por tipo de defeito:
  - **Forma (deformação/fragmentação):** solidez e razão de aspecto.
  - **Cor (mancha/contaminação):** distância de Mahalanobis no espaço `[R, G, B, S]`.
  - **Superfície (rachadura/risco):** Black-Hat, bordas internas de Canny e mapa de resíduo.

## Slide 6 — Resultados Visuais: Etapas Intermediárias (5:45 – 7:15)
- Mostrar painéis 2×3 de `results/mvtec_pill/etapas/` (ex.: `good`, `color`, `crack`, `contamination`) e `results/mvtec_capsule/etapas/` (ex.: `good`, `squeeze`, `scratch`).

## Slide 7 — Resultados Quantitativos e Comparação (7:15 – 8:45)
- Exibir `results/mvtec_pill/metricas_comparacao.png` e `results/mvtec_capsule/metricas_comparacao.png`.
- **MVTec `pill`:** F1-Score **83,90%** (NumPy do zero) vs **83,46%** (OpenCV); Precisão ~88,9%; Recall ~79,4%.
- **MVTec `capsule`:** F1-Score **60,36%** (NumPy do zero) vs **58,54%** (OpenCV); destaque para `squeeze` (80%).
- **Desempenho computacional:** ~227 ms/img (NumPy puro) vs ~6–7 ms/img (OpenCV, ~35× mais rápido).

## Slide 8 — Análise Crítica e Conclusão (8:45 – 10:00)
- **Pontos fortes:** pipeline modular, explicável, calibração automática sem redes neurais e equivalência validada entre código do zero e OpenCV.
- **Limitações observadas:** microdefeitos (`poke` em cápsulas) e reflexos especulares em superfícies brilhantes.
- **Trabalhos futuros:** iluminação controlada (polarizada), alta resolução na ROI e combinação híbrida com CNN/autoencoders para microdefeitos.
