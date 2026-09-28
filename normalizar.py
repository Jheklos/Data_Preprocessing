import csv


# ============================================================
# CONFIGURAÇÃO
# ============================================================

ENTRADA = "Amadey.csv"
SAIDA = "Dados_Normalizado.csv"

# Faixa desejada para a normalização
#
# Exemplos:
#
# FAIXA = (0, 1)
# FAIXA = (-1, 1)
# FAIXA = (0, 9)
#
FAIXA = (-1, 1)


# ============================================================
# FUNÇÃO DE NORMALIZAÇÃO
# ============================================================

def normalizar(valor, minimo, maximo, novo_min, novo_max):

    # Feature constante
    if maximo == minimo:
        return novo_min

    return (
        novo_min
        + (
            (valor - minimo)
            / (maximo - minimo)
        )
        * (novo_max - novo_min)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Validação da faixa
    # --------------------------------------------------------

    novo_min, novo_max = FAIXA

    if novo_min >= novo_max:
        raise ValueError(
            "FAIXA inválida. "
            "O primeiro valor deve ser menor que o segundo."
        )

    # --------------------------------------------------------
    # Leitura do CSV
    #
    # Estrutura:
    #
    # coluna 0 -> nome do arquivo
    # coluna 1 -> classe
    # coluna 2+ -> features
    # --------------------------------------------------------

    with open(
        ENTRADA,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        leitor = csv.reader(
            f,
            delimiter=";"
        )

        linhas = list(leitor)

    if not linhas:
        raise ValueError(
            "O arquivo de entrada está vazio."
        )

    # --------------------------------------------------------
    # Cabeçalho
    # --------------------------------------------------------

    cabecalho = linhas[0]

    # --------------------------------------------------------
    # Verificação da quantidade de colunas
    # --------------------------------------------------------

    if len(cabecalho) < 3:
        raise ValueError(
            "O dataset precisa possuir pelo menos "
            "3 colunas: nome do arquivo, classe e feature."
        )

    # --------------------------------------------------------
    # Identificação das colunas
    # --------------------------------------------------------

    indice_nome_arquivo = 0
    indice_classe = 1

    indices_features = list(
        range(2, len(cabecalho))
    )

    # --------------------------------------------------------
    # Dados
    # --------------------------------------------------------

    dados = linhas[1:]

    # --------------------------------------------------------
    # Descobre mínimo e máximo de cada feature
    # --------------------------------------------------------

    minimos = {}
    maximos = {}

    for indice in indices_features:

        valores = []

        for linha in dados:

            if len(linha) <= indice:
                raise ValueError(
                    f"A linha possui menos colunas "
                    f"que o cabeçalho: {linha}"
                )

            valor = float(linha[indice])

            valores.append(valor)

        minimos[indice] = min(valores)
        maximos[indice] = max(valores)

    # --------------------------------------------------------
    # Aplica normalização
    # --------------------------------------------------------

    for linha in dados:

        for indice in indices_features:

            valor = float(linha[indice])

            valor_normalizado = normalizar(
                valor,
                minimos[indice],
                maximos[indice],
                novo_min,
                novo_max
            )

            # Mantém 6 casas decimais
            linha[indice] = f"{valor_normalizado:.6f}"

    # --------------------------------------------------------
    # Salva resultado
    #
    # A estrutura original é preservada:
    #
    # coluna 0 -> nome do arquivo
    # coluna 1 -> classe
    # coluna 2+ -> features normalizadas
    # --------------------------------------------------------

    with open(
        SAIDA,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        escritor = csv.writer(
            f,
            delimiter=";",
            lineterminator="\n"
        )

        # Cabeçalho
        escritor.writerow(cabecalho)

        # Dados
        escritor.writerows(dados)

    # --------------------------------------------------------
    # Informações
    # --------------------------------------------------------

    print("=" * 60)
    print("NORMALIZAÇÃO CONCLUÍDA")
    print("=" * 60)

    print(f"Entrada : {ENTRADA}")
    print(f"Saída   : {SAIDA}")

    print(
        f"\nAmostras: {len(dados)}"
    )

    print(
        f"Features: {len(indices_features)}"
    )

    print(
        f"Faixa utilizada: "
        f"[{novo_min}, {novo_max}]"
    )

    print("\nFaixas originais das features:")
    print("-" * 60)

    for indice in indices_features:

        nome = cabecalho[indice]

        print(
            f"{nome}: "
            f"{minimos[indice]} -> "
            f"{maximos[indice]}"
        )

    print("\nArquivo normalizado gerado:")
    print(SAIDA)


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()
