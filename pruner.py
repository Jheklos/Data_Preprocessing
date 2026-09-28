# -*- coding: utf-8 -*-

"""
Script de Poda de Features por Correlação de Pearson

Entrada:
    Dados.CSV

Saída:
    Dados_Poda.csv

Características:
- Preserva a primeira coluna (nome do arquivo)
- Utiliza a segunda coluna como classe (y)
- Preserva os cabeçalhos
- Ignora a primeira linha apenas quando necessário? NÃO:
  o cabeçalho é lido normalmente e preservado na saída
- Remove features constantes
- Remove features com |correlação de Pearson| >= 0.999
- Suporta threshold=None
- Quando threshold=None:
      mantém todas as features válidas
- Quando threshold possui valor:
      mantém somente features com |correlação| >= threshold
- Gera relatório HTML detalhado da poda
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


# ============================================================
# CONFIGURAÇÃO
# ============================================================

ENTRADA = "Dados_Normalizado.csv"
SAIDA = "Dados_Poda.csv"

RELATORIO_HTML = "relatorio_poda.html"

# None = não aplicar seleção adicional por threshold
# Exemplo:
# threshold = 0.01
threshold = None

# Correlação a partir da qual a feature é considerada "suja"
LIMIAR_SUJO = 0.999


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def calcular_correlacao(feature, y):
    """
    Calcula a correlação de Pearson entre uma feature e y.

    Retorna:
        float ou NaN
    """

    # Remove valores ausentes somente para o cálculo
    dados = pd.concat(
        [
            pd.Series(feature, name="feature"),
            pd.Series(y, name="y")
        ],
        axis=1
    ).dropna()

    if len(dados) < 2:
        return np.nan

    # Feature sem variação
    if dados["feature"].std() == 0:
        return np.nan

    # Classe sem variação
    if dados["y"].std() == 0:
        return np.nan

    return dados["feature"].corr(dados["y"], method="pearson")


def escapar_html(valor):
    """
    Escapa conteúdo para utilização segura no HTML.
    """
    return (
        str(valor)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


# ============================================================
# PODA
# ============================================================

def realizar_poda(df, threshold):
    """
    Executa a poda das features.

    Estrutura esperada:

        coluna 0 -> nome do arquivo
        coluna 1 -> classe
        coluna 2+ -> features
    """

    # --------------------------------------------------------
    # Identificação das colunas
    # --------------------------------------------------------

    coluna_nome_arquivo = df.columns[0]
    coluna_classe = df.columns[1]

    feature_columns = list(df.columns[2:])

    y = pd.to_numeric(df[coluna_classe], errors="coerce")

    resultados = []

    colunas_mantidas = [
        coluna_nome_arquivo,
        coluna_classe
    ]

    # --------------------------------------------------------
    # Análise das features
    # --------------------------------------------------------

    for coluna in feature_columns:

        serie = pd.to_numeric(df[coluna], errors="coerce")

        # ====================================================
        # 1. FEATURE CONSTANTE
        # ====================================================

        valores_validos = serie.dropna()

        if len(valores_validos) == 0:
            resultados.append({
                "feature": coluna,
                "correlacao": np.nan,
                "abs_correlacao": np.nan,
                "status": "Podada",
                "motivo": "Sem valores numéricos válidos"
            })

            continue

        if valores_validos.nunique() <= 1:

            resultados.append({
                "feature": coluna,
                "correlacao": 0.0,
                "abs_correlacao": 0.0,
                "status": "Podada",
                "motivo": "Feature constante"
            })

            continue

        # ====================================================
        # 2. CORRELAÇÃO DE PEARSON
        # ====================================================

        correlacao = calcular_correlacao(serie, y)

        if pd.isna(correlacao):

            resultados.append({
                "feature": coluna,
                "correlacao": np.nan,
                "abs_correlacao": np.nan,
                "status": "Podada",
                "motivo": "Correlação não pôde ser calculada"
            })

            continue

        abs_correlacao = abs(correlacao)

        # ====================================================
        # 3. FEATURE "SUJA"
        # ====================================================

        if abs_correlacao >= LIMIAR_SUJO:

            resultados.append({
                "feature": coluna,
                "correlacao": correlacao,
                "abs_correlacao": abs_correlacao,
                "status": "Podada",
                "motivo": "Correlação >= 0.999"
            })

            continue

        # ====================================================
        # 4. THRESHOLD
        # ====================================================

        if threshold is None:

            # Sem threshold:
            # mantém todas as features válidas
            colunas_mantidas.append(coluna)

            resultados.append({
                "feature": coluna,
                "correlacao": correlacao,
                "abs_correlacao": abs_correlacao,
                "status": "Mantida",
                "motivo": "Mantida (sem threshold)"
            })

        else:

            # Com threshold:
            # seleciona somente |correlação| >= threshold

            if abs_correlacao >= threshold:

                colunas_mantidas.append(coluna)

                resultados.append({
                    "feature": coluna,
                    "correlacao": correlacao,
                    "abs_correlacao": abs_correlacao,
                    "status": "Mantida",
                    "motivo": f"|Correlação| >= {threshold}"
                })

            else:

                resultados.append({
                    "feature": coluna,
                    "correlacao": correlacao,
                    "abs_correlacao": abs_correlacao,
                    "status": "Podada",
                    "motivo": f"|Correlação| < {threshold}"
                })

    # --------------------------------------------------------
    # Dataset final
    # --------------------------------------------------------

    df_podado = df[colunas_mantidas].copy()

    resultados_df = pd.DataFrame(resultados)

    return df_podado, resultados_df


# ============================================================
# RELATÓRIO HTML
# ============================================================

def gerar_relatorio_html(
    entrada,
    saida,
    resultados,
    quantidade_linhas,
    quantidade_features_original,
    quantidade_features_final,
    threshold
):

    quantidade_constantes = len(
        resultados[
            resultados["motivo"] == "Feature constante"
        ]
    )

    quantidade_sujas = len(
        resultados[
            resultados["motivo"] == "Correlação >= 0.999"
        ]
    )

    quantidade_threshold = len(
        resultados[
            resultados["motivo"].str.startswith(
                "|Correlação| <",
                na=False
            )
        ]
    )

    quantidade_mantidas = len(
        resultados[
            resultados["status"] == "Mantida"
        ]
    )

    quantidade_podadas = len(
        resultados[
            resultados["status"] == "Podada"
        ]
    )

    percentual_reducao = 0

    if quantidade_features_original > 0:
        percentual_reducao = (
            quantidade_podadas /
            quantidade_features_original
        ) * 100

    # --------------------------------------------------------
    # Linhas da tabela
    # --------------------------------------------------------

    tabela_html = ""

    for _, row in resultados.iterrows():

        correlacao = row["correlacao"]

        if pd.isna(correlacao):
            correlacao_texto = "N/A"
        else:
            correlacao_texto = f"{correlacao:.6f}"

        abs_corr = row["abs_correlacao"]

        if pd.isna(abs_corr):
            abs_corr_texto = "N/A"
        else:
            abs_corr_texto = f"{abs_corr:.6f}"

        if row["status"] == "Mantida":
            classe_status = "mantida"
            icone = "✓"
        else:
            classe_status = "podada"
            icone = "✕"

        tabela_html += f"""
        <tr>
            <td>{escapar_html(row["feature"])}</td>
            <td>{correlacao_texto}</td>
            <td>{abs_corr_texto}</td>
            <td class="{classe_status}">
                {icone} {escapar_html(row["status"])}
            </td>
            <td>{escapar_html(row["motivo"])}</td>
        </tr>
        """

    # --------------------------------------------------------
    # Threshold exibido
    # --------------------------------------------------------

    if threshold is None:
        threshold_texto = "None — nenhuma seleção adicional por threshold"
    else:
        threshold_texto = str(threshold)

    # --------------------------------------------------------
    # HTML
    # --------------------------------------------------------

    html = f"""
<!DOCTYPE html>

<html lang="pt-BR">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>Relatório de Poda de Features</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    padding: 30px;
    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background:
        linear-gradient(
            135deg,
            #1f2937,
            #374151
        );

    color: #333;
}}

.container {{
    max-width: 1400px;
    margin: auto;
}}

.header {{
    background: white;
    border-radius: 15px;
    padding: 30px;
    margin-bottom: 25px;
    text-align: center;
}}

.header h1 {{
    margin: 0 0 10px 0;
    color: #1f2937;
}}

.header p {{
    color: #666;
    margin: 5px;
}}

.cards {{
    display: grid;
    grid-template-columns:
        repeat(
            auto-fit,
            minmax(220px, 1fr)
        );

    gap: 20px;
    margin-bottom: 25px;
}}

.card {{
    background: white;
    border-radius: 15px;
    padding: 25px;
    box-shadow:
        0 5px 20px
        rgba(0,0,0,0.15);
}}

.card h3 {{
    margin-top: 0;
    color: #666;
    font-size: 15px;
}}

.card .number {{
    font-size: 30px;
    font-weight: bold;
    color: #1f2937;
}}

.section {{
    background: white;
    border-radius: 15px;
    padding: 25px;
    margin-bottom: 25px;
    overflow-x: auto;
}}

.section h2 {{
    margin-top: 0;
    color: #1f2937;
}}

.info {{
    background: #f3f4f6;
    border-left: 5px solid #374151;
    padding: 15px;
    margin-bottom: 20px;
    border-radius: 5px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    min-width: 800px;
}}

th {{
    background: #1f2937;
    color: white;
    padding: 12px;
    text-align: left;
}}

td {{
    padding: 10px;
    border-bottom: 1px solid #ddd;
}}

tr:hover {{
    background: #f5f5f5;
}}

.mantida {{
    color: #198754;
    font-weight: bold;
}}

.podada {{
    color: #dc3545;
    font-weight: bold;
}}

.footer {{
    text-align: center;
    color: white;
    margin-top: 20px;
}}

.badge {{
    display: inline-block;
    padding: 6px 12px;
    border-radius: 20px;
    background: #e5e7eb;
    font-weight: bold;
}}

</style>

</head>

<body>

<div class="container">

    <div class="header">

        <h1>Relatório de Poda de Features</h1>

        <p>
            Análise baseada na correlação de Pearson
        </p>

        <p>
            Gerado em:
            {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
        </p>

    </div>


    <div class="section">

        <h2>Configuração</h2>

        <div class="info">

            <p>
                <strong>Arquivo de entrada:</strong>
                {escapar_html(entrada)}
            </p>

            <p>
                <strong>Arquivo de saída:</strong>
                {escapar_html(saida)}
            </p>

            <p>
                <strong>Threshold:</strong>
                {escapar_html(threshold_texto)}
            </p>

            <p>
                <strong>Limiar de feature suja:</strong>
                |correlação| &ge; 0.999
            </p>

            <p>
                <strong>Coluna 1:</strong>
                nome do arquivo — preservada e não utilizada
                no cálculo da correlação
            </p>

            <p>
                <strong>Coluna 2:</strong>
                classe — utilizada como variável alvo (y)
            </p>

            <p>
                <strong>Features:</strong>
                colunas a partir da terceira coluna
            </p>

        </div>

    </div>


    <div class="cards">

        <div class="card">
            <h3>Linhas do Dataset</h3>
            <div class="number">
                {quantidade_linhas}
            </div>
        </div>

        <div class="card">
            <h3>Features Originais</h3>
            <div class="number">
                {quantidade_features_original}
            </div>
        </div>

        <div class="card">
            <h3>Features Mantidas</h3>
            <div class="number">
                {quantidade_mantidas}
            </div>
        </div>

        <div class="card">
            <h3>Features Podadas</h3>
            <div class="number">
                {quantidade_podadas}
            </div>
        </div>

        <div class="card">
            <h3>Features Constantes</h3>
            <div class="number">
                {quantidade_constantes}
            </div>
        </div>

        <div class="card">
            <h3>Features |corr| &ge; 0.999</h3>
            <div class="number">
                {quantidade_sujas}
            </div>
        </div>

        <div class="card">
            <h3>Podadas pelo Threshold</h3>
            <div class="number">
                {quantidade_threshold}
            </div>
        </div>

        <div class="card">
            <h3>Redução</h3>
            <div class="number">
                {percentual_reducao:.2f}%
            </div>
        </div>

    </div>


    <div class="section">

        <h2>Resumo da Poda</h2>

        <div class="info">

            <p>
                Foram analisadas
                <strong>{quantidade_features_original}</strong>
                features.
            </p>

            <p>
                Foram mantidas
                <strong>{quantidade_mantidas}</strong>
                features.
            </p>

            <p>
                Foram podadas
                <strong>{quantidade_podadas}</strong>
                features.
            </p>

            <p>
                A redução total de features foi de
                <strong>{percentual_reducao:.2f}%</strong>.
            </p>

        </div>

    </div>


    <div class="section">

        <h2>Detalhamento das Features</h2>

        <table>

            <thead>

                <tr>

                    <th>Feature</th>

                    <th>Correlação</th>

                    <th>|Correlação|</th>

                    <th>Status</th>

                    <th>Motivo</th>

                </tr>

            </thead>

            <tbody>

                {tabela_html}

            </tbody>

        </table>

    </div>


    <div class="section">

        <h2>Critérios utilizados</h2>

        <div class="info">

            <p>
                <strong>Feature constante:</strong>
                possui apenas um valor distinto
                entre as amostras.
            </p>

            <p>
                <strong>Feature suja:</strong>
                possui |correlação de Pearson| &ge; 0.999
                com a classe.
            </p>

            <p>
                <strong>Threshold = None:</strong>
                todas as features que não sejam constantes
                ou sujas são mantidas.
            </p>

            <p>
                <strong>Threshold definido:</strong>
                somente features com
                |correlação| &ge; threshold são mantidas,
                além da remoção das features sujas.
            </p>

        </div>

    </div>


    <div class="footer">

        Relatório gerado automaticamente pelo
        Script de Poda de Features

    </div>

</div>

</body>

</html>
"""

    with open(
        RELATORIO_HTML,
        "w",
        encoding="utf-8"
    ) as arquivo:

        arquivo.write(html)

    print(
        f"Relatório HTML gerado: "
        f"{os.path.abspath(RELATORIO_HTML)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PODA DE FEATURES")
    print("=" * 70)

    print(f"\nEntrada : {ENTRADA}")
    print(f"Saída   : {SAIDA}")
    print(f"Threshold: {threshold}")

    # --------------------------------------------------------
    # Verificação do arquivo
    # --------------------------------------------------------

    if not os.path.exists(ENTRADA):

        print(
            f"\nERRO: arquivo de entrada não encontrado:"
            f"\n{os.path.abspath(ENTRADA)}"
        )

        return

    # --------------------------------------------------------
    # Leitura
    #
    # Diferentemente do código antigo:
    #
    #     header=None
    #     df.iloc[1:, 1:]
    #
    # aqui o pandas utiliza a primeira linha como cabeçalho.
    #
    # Isso permite preservar os nomes das colunas no resultado.
    # --------------------------------------------------------

    print("\nLendo dataset...")

    df = pd.read_csv(
        ENTRADA,
        sep=";",
        decimal=".",
        low_memory=False
    )

    # --------------------------------------------------------
    # Validação mínima
    # --------------------------------------------------------

    if df.shape[1] < 3:

        print(
            "\nERRO: o dataset precisa possuir pelo menos "
            "3 colunas:"
        )

        print("  1ª coluna = nome do arquivo")
        print("  2ª coluna = classe")
        print("  3ª coluna em diante = features")

        return

    print(
        f"Linhas encontradas : {len(df)}"
    )

    print(
        f"Colunas encontradas: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Informações
    # --------------------------------------------------------

    quantidade_features_original = len(df.columns) - 2

    print(
        f"Features originais : "
        f"{quantidade_features_original}"
    )

    # --------------------------------------------------------
    # Executa poda
    # --------------------------------------------------------

    print("\nAnalisando features...")
    print("-" * 70)

    df_podado, resultados = realizar_poda(
        df,
        threshold
    )

    # --------------------------------------------------------
    # Salva dataset
    # --------------------------------------------------------

    df_podado.to_csv(
        SAIDA,
        index=False,
        sep=";",
        decimal="."
    )

    # --------------------------------------------------------
    # Estatísticas
    # --------------------------------------------------------

    quantidade_features_final = (
        len(df_podado.columns) - 2
    )

    quantidade_podadas = (
        quantidade_features_original -
        quantidade_features_final
    )

    print("\n" + "=" * 70)
    print("RESULTADO DA PODA")
    print("=" * 70)

    print(
        f"Features originais : "
        f"{quantidade_features_original}"
    )

    print(
        f"Features mantidas  : "
        f"{quantidade_features_final}"
    )

    print(
        f"Features podadas   : "
        f"{quantidade_podadas}"
    )

    print(
        f"\nDataset salvo em:"
        f"\n{os.path.abspath(SAIDA)}"
    )

    # --------------------------------------------------------
    # Gera relatório HTML
    # --------------------------------------------------------

    gerar_relatorio_html(
        entrada=ENTRADA,
        saida=SAIDA,
        resultados=resultados,
        quantidade_linhas=len(df),
        quantidade_features_original=
            quantidade_features_original,
        quantidade_features_final=
            quantidade_features_final,
        threshold=threshold
    )

    print("\nPoda concluída.")


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()
