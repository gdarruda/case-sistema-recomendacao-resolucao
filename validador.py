import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from tqdm import tqdm


def validar_modelo(df_test, predict_function, k=5, total_produtos_catalogo=20):
    """
    Realiza a validação completa em uma única passagem, consolidando métricas
    binárias (conversão) e financeiras (receita).
    """

    precisions, recalls, hit_rates, maps, ndcgs_bin = [], [], [], [], []

    ndcgs_rev, revenues_captured, revenue_recalls = [], [], []

    produtos_recomendados_total = set()

    clientes_teste = df_test[["id_cliente", "timestamp"]].drop_duplicates()

    print(
        f"Iniciando validação completa para {clientes_teste.shape[0]} casos de teste..."
    )

    for _, row in tqdm(clientes_teste.iterrows()):
        interacoes_reais = df_test[
            (df_test["id_cliente"] == row["id_cliente"])
            & (df_test["timestamp"] == row["timestamp"])
            & (df_test["contratou"] == 1)
        ]

        if interacoes_reais.empty:
            continue

        dict_receita_real = dict(
            zip(interacoes_reais["produto"], interacoes_reais["receita_gerada"])
        )
        produtos_contratados = set(dict_receita_real.keys())
        receita_total_possivel = interacoes_reais["receita_gerada"].sum()

        ranking_predito = predict_function(row["id_cliente"], row["timestamp"])

        top_k_predito = ranking_predito.head(k)["produto"].tolist()
        produtos_recomendados_total.update(top_k_predito)

        hits = [1 if p in produtos_contratados else 0 for p in top_k_predito]
        n_hits = sum(hits)

        precisions.append(n_hits / k)
        recalls.append(n_hits / len(produtos_contratados))
        hit_rates.append(1 if n_hits > 0 else 0)

        if n_hits > 0:
            p_at_i = [
                sum(hits[: i + 1]) / (i + 1) for i, h in enumerate(hits) if h == 1
            ]
            maps.append(np.mean(p_at_i))
        else:
            maps.append(0)

        receita_no_topo = sum([dict_receita_real.get(p, 0) for p in top_k_predito])
        revenues_captured.append(receita_no_topo)
        revenue_recalls.append(
            receita_no_topo / receita_total_possivel
            if receita_total_possivel > 0
            else 0
        )

        y_score = ranking_predito["score"].values

        y_true_bin = (
            ranking_predito["produto"]
            .apply(lambda x: 1 if x in produtos_contratados else 0)
            .values
        )
        y_true_rev = (
            ranking_predito["produto"]
            .apply(lambda x: dict_receita_real.get(x, 0))
            .values
        )

        ndcgs_bin.append(ndcg_score([y_true_bin], [y_score], k=k))
        ndcgs_rev.append(ndcg_score([y_true_rev], [y_score], k=k))

    metrics = {
        "Métricas Primárias": {
            f"Precision@{k}": np.mean(precisions),
            f"NDCG@{k} (Binário)": np.mean(ndcgs_bin),
        },
        "Métricas Secundárias": {
            f"Recall@{k}": np.mean(recalls),
            f"Hit Rate@{k}": np.mean(hit_rates),
            f"MAP@{k}": np.mean(maps),
            "Catalog Coverage": len(produtos_recomendados_total)
            / total_produtos_catalogo,
        },
        "Métricas de Receita": {
            f"NDCG@{k} (Receita)": np.mean(ndcgs_rev),
            f"Avg Revenue@{k}": np.mean(revenues_captured),
            f"Revenue Recall@{k}": np.mean(revenue_recalls),
        },
    }

    return metrics


def validar_modelo_por_segmento(
    df_test, predict_function, k=5, total_produtos_catalogo=20
):
    if "segmento" not in df_test.columns:
        raise ValueError("df_test deve conter a coluna 'segmento'")

    segmentos = df_test["segmento"].unique()
    metrics_por_segmento = {}

    for segmento in segmentos:
        df_segmento = df_test[df_test["segmento"] == segmento]
        print(f"Validando segmento: {segmento}")
        metrics_por_segmento[segmento] = validar_modelo(
            df_segmento, predict_function, k, total_produtos_catalogo
        )

    return metrics_por_segmento


def validar_modelo_por_tempo_relacionamento(
    df_test, predict_function, k=5, total_produtos_catalogo=20
):
    if "qtd_meses_cliente" not in df_test.columns:
        raise ValueError(
            "df_test deve conter a coluna 'qtd_meses_cliente' oriunda do clientes.csv"
        )

    df_analise = df_test.copy()

    df_analise["faixa_relacionamento"] = pd.cut(
        df_analise["qtd_meses_cliente"],
        bins=[0, 3, 6, 12],
        labels=["<3 meses", "3 a 6 meses", "6 a 12 meses"],
        include_lowest=True,
    )

    faixas_unicas = df_analise["faixa_relacionamento"].unique()
    metrics_por_tempo = {}

    for faixa in faixas_unicas:
        if pd.isna(faixa):
            continue

        print(f"Validando clientes com meses de leracionamento: {faixa}")

        df_faixa = df_analise[df_analise["faixa_relacionamento"] == faixa]

        metrics_por_tempo[str(faixa)] = validar_modelo(
            df_faixa, predict_function, k, total_produtos_catalogo
        )

    return metrics_por_tempo


def gerar_relatorio(
    resultados: dict,
    resultados_segmento: dict,
    resultados_tempo_relacionamento: dict,
    modelo: str,
) -> pd.DataFrame:

    return (
        pd.concat(
            [pd.json_normalize(resultados).assign(grupo="geral")]
            + [
                pd.json_normalize(resultados_segmento[segmento]).assign(grupo=segmento)
                for segmento in resultados_segmento.keys()
            ]
            + [
                pd.json_normalize(
                    resultados_tempo_relacionamento[tempo_relacionamento]
                ).assign(grupo=tempo_relacionamento)
                for tempo_relacionamento in resultados_tempo_relacionamento.keys()
            ]
        )
        .assign(modelo=modelo)
        .reset_index(drop=True)
    )
