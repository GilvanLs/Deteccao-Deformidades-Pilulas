"""
Classificador interpretavel para controle de qualidade de comprimidos e capsulas:
- Com modelo de referencia (`train/good`): utiliza limiares calibrados por validacao
  para detectar fragmentacao, deformacao, manchas (HSV/Mahalanobis) e rachaduras/riscos
  (Black-Hat + Canny + residuo de aparencia).
- Sem modelo de referencia: aplica regras fixas interpretaveis.
"""

from __future__ import annotations

from dataclasses import dataclass
from src.features import PillFeatures, ReferenceModel


@dataclass
class ClassificationResult:
    status: str  # "APROVADO" ou "REPROVADO"
    defects: list[str]
    features: PillFeatures


def classify_pill(features: PillFeatures) -> ClassificationResult:
    """Classificacao por regras fixas (sem modelo de referencia)."""
    defects: list[str] = []

    if features.area < 500:
        return ClassificationResult(status="REPROVADO", defects=["ausencia_ou_fragmento"], features=features)

    if features.solidity < 0.965 or features.morph_diff_ratio > 0.015:
        defects.append("fragmentacao")
    if features.aspect_ratio > 1.22 or features.circularity < 0.78:
        defects.append("deformacao")
    if features.stain_ratio > 0.030 or features.std_s > 16.0:
        defects.append("mancha")
    if features.internal_edge_density > 0.004:
        defects.append("rachadura")

    status = "REPROVADO" if defects else "APROVADO"
    return ClassificationResult(status=status, defects=defects, features=features)


def classify_with_reference(
    features: PillFeatures,
    ref: ReferenceModel,
) -> ClassificationResult:
    """Classifica o objeto comparando suas caracteristicas aos limiares calibrados."""
    defects: list[str] = []

    if features.area < 500:
        return ClassificationResult(status="REPROVADO", defects=["ausencia_ou_fragmento"], features=features)

    # 1. Fragmentacao (quebras na borda -> queda de solidez)
    if features.solidity < ref.lo_solidity:
        defects.append("fragmentacao")

    # 2. Deformacao (razao de aspecto fora da tolerancia normal)
    if abs(features.aspect_ratio - ref.ar_median) > ref.ar_tol:
        defects.append("deformacao")

    # 3. Mancha / Alteracao de cor (distancia de Mahalanobis no espaco RGB+HSV)
    if features.color_maha_p995 > ref.hi_thresholds["color_maha_p995"]:
        defects.append("mancha")

    # 4. Rachadura / Risco / Falha superficial (Black-Hat + Canny + residuo)
    if (
        features.blackhat_p995 > ref.hi_thresholds["blackhat_p995"]
        or features.internal_edge_density > ref.hi_thresholds["internal_edge_density"]
        or features.residual_p995 > ref.hi_thresholds["residual_p995"]
        or features.anomaly_area_ratio > ref.hi_thresholds["anomaly_area_ratio"]
    ):
        defects.append("rachadura_ou_superficie")

    status = "REPROVADO" if defects else "APROVADO"
    return ClassificationResult(status=status, defects=defects, features=features)
