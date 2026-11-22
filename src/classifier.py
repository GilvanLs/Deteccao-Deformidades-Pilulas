"""
Classificador baseado em regras interpretaveis para controle de qualidade
de comprimidos (APROVADO vs REPROVADO) e diagnostico dos defeitos encontrados.
"""

from __future__ import annotations

from dataclasses import dataclass
from src.features import PillFeatures


@dataclass
class ClassificationResult:
    status: str  # "APROVADO" ou "REPROVADO"
    defects: list[str]
    features: PillFeatures


def classify_pill(features: PillFeatures) -> ClassificationResult:
    """
    Classifica o comprimido com base nas caracteristicas extraidas:
    - Fragmentacao: baixa solidez ou diferenca morfologica elevada
    - Deformacao: razao de aspecto ou circularidade fora do padrao
    - Mancha: alta proporcao de pixels com alteracao de cor/saturacao no HSV
    - Rachadura: alta densidade de bordas internas (Canny) na superficie
    """
    defects: list[str] = []

    if features.area < 500:
        return ClassificationResult(status="REPROVADO", defects=["ausencia_ou_fragmento"], features=features)

    # 1. Fragmentacao (bordas quebradas / partes ausentes)
    if features.solidity < 0.965 or features.morph_diff_ratio > 0.015:
        defects.append("fragmentacao")

    # 2. Deformacao (formato ou dimensoes fora do padrao)
    if features.aspect_ratio > 1.22 or features.circularity < 0.78:
        defects.append("deformacao")

    # 3. Mancha (alteracoes de coloracao / manchas superficiais em HSV)
    if features.stain_ratio > 0.030 or features.std_s > 16.0:
        defects.append("mancha")

    # 4. Rachadura (fissuras internas detectadas por Canny)
    if features.internal_edge_density > 0.004:
        defects.append("rachadura")

    status = "REPROVADO" if defects else "APROVADO"
    return ClassificationResult(status=status, defects=defects, features=features)
