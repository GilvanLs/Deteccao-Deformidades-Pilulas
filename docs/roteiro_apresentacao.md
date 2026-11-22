# Roteiro de Apresentação e Vídeo (Máximo 10 minutos)

Estrutura sugerida para os **slides**, **apresentação presencial** e **vídeo** exigidos nas regras (12) e (13).

---

## Slide 1 — Capa (0:00 – 0:30)
- **Título:** Inspeção Visual de Comprimidos por Processamento Digital de Imagens
- **Subtítulo:** Detecção de Defeitos e Controle de Qualidade Farmacêutico (Implementação do Zero em NumPy + OpenCV)
- **Integrantes / Disciplina / Data**

## Slide 2 — Introdução e Contextualização (0:30 – 1:45)
- Relevância do controle de qualidade na indústria farmacêutica (segurança do paciente, dosagem, integridade).
- Limitações da inspeção visual manual (fadiga, subjetividade, custo).
- Objetivo do projeto: automatizar a detecção de **rachaduras, manchas, fragmentação e deformações** em comprimidos, classificando-os em **APROVADO** ou **REPROVADO**.

## Slide 3 — Base de Dados e Cenários de Teste (1:45 – 2:45)
- Descrição das 60 imagens (`320 × 320` px, PNG, RGB) geradas com iluminação realista e ruído de sensor (Gaussiano + sal e pimenta):
  - 20 `aprovado`
  - 10 `rachadura`
  - 10 `mancha`
  - 10 `fragmentacao`
  - 10 `deformacao`
- Mostrar exemplos visuais de cada classe e explicar que o sistema também aceita bases públicas (como MVTec AD `pill`).

## Slide 4 — Fundamentação Teórica e Solução do Zero (2:45 – 4:15)
- Destacar o diferencial do projeto (Regra 10): implementação **do zero apenas com NumPy** em `src/pdi_scratch.py`, comparada ao **OpenCV** (`src/pdi_opencv.py`).
- Breve explicação dos conceitos:
  - **Filtro da Mediana & Gaussiano:** remoção de ruído impulsivo e suavização.
  - **Espaço HSV:** separação de cor/saturação (S) e brilho (V) para manchas.
  - **Limiarização de Otsu + Morfologia Matemática:** segmentação automática e limpeza de máscara.
  - **Detector de Canny:** Sobel + Supressão Não-Máxima + Histerese para bordas e fissuras.

## Slide 5 — Metodologia e Pipeline (4:15 – 5:45)
- Fluxograma das 8 etapas do `src/pipeline.py`.
- Como cada defeito é extraído em `src/features.py`:
  - **Fragmentação:** Solidez e diferença morfológica (fechamento amplo).
  - **Deformação:** Razão de aspecto e circularidade.
  - **Mancha:** Desvio de saturação/brilho no interior erodido (HSV).
  - **Rachadura:** Densidade de bordas de Canny estritamente no interior do comprimido.

## Slide 6 — Resultados Visuais (Etapas Intermediárias) (5:45 – 7:15)
- Exibir os painéis 2×3 gerados em `results/etapas/` (antes/depois, filtragem, canal S do HSV, máscara segmentada, bordas de Canny, classificação final) para pelo menos 1 comprimido aprovado e 2–3 defeituosos.

## Slide 7 — Resultados Quantitativos e Comparação (7:15 – 8:45)
- Exibir `results/metricas_comparacao.png` (Matrizes de Confusão, Acurácia, Precisão, Recall, F1-Score e tempo médio por imagem).
- Comparar a implementação **do zero (NumPy)** vs **OpenCV (C/C++ otimizado)**: mostrar que ambas atingem alta qualidade de detecção, destacando o ganho de velocidade do OpenCV frente ao valor didático da implementação do zero.

## Slide 8 — Análise Crítica e Conclusão (8:45 – 10:00)
- **Pontos fortes:** pipeline modular, explicável, sem caixa-preta, alta precisão e implementação dos algoritmos clássicos do zero.
- **Limitações e trabalhos futuros:** sensibilidade a iluminação não controlada em esteiras reais; calibração automática por lote de comprimido; integração com câmera em tempo real ou redes neurais para casos ambíguos.
- **Encerramento e agradecimentos.**
