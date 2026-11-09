
# ============================================================
# EDUPREDICT AI
# CLASSIFICADOR EXPERIMENTAL DE DESEMPENHO ESCOLAR
# Python + Gradio + Machine Learning + SQLite
# ============================================================

!pip install -q gradio pandas numpy scikit-learn

import os
import sqlite3
import getpass
from datetime import datetime

import gradio as gr
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

MEDIA_MINIMA = 6.0
LIMITE_FALTAS = 5
BANCO = "edupredict.db"

COLUNAS_HISTORICO = [
    "Aluno", "Disciplina", "AV1", "AV2", "AV3",
    "Média", "Faltas", "Horas de estudo",
    "Resultado", "Confiança", "Data do registro",
    "Data e horário da análise"
]

# Login: no Render, configure APP_USERNAME e APP_PASSWORD
USUARIO_ACESSO = os.environ.get("APP_USERNAME", "lil dhef")
SENHA_ACESSO = os.environ.get("APP_PASSWORD")

# Para testar no Colab, digite a senha quando solicitado.
if not SENHA_ACESSO:
    SENHA_ACESSO = getpass.getpass("Digite a senha de acesso do EduPredict AI: ")

if not SENHA_ACESSO:
    raise ValueError("É necessário configurar uma senha de acesso.")

# ============================================================
# 2. BANCO DE DADOS
# ============================================================

def conectar_banco():
    return sqlite3.connect(BANCO)

def criar_banco():
    with conectar_banco() as conexao:
        cursor = conexao.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historico (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                aluno TEXT,
                disciplina TEXT,
                av1 REAL,
                av2 REAL,
                av3 REAL,
                media REAL,
                faltas INTEGER,
                horas REAL,
                resultado TEXT,
                confianca REAL,
                data_registro TEXT,
                data_analise TEXT
            )
        """)

        conexao.commit()

def buscar_historico():
    with conectar_banco() as conexao:
        df = pd.read_sql_query("""
            SELECT
                aluno AS "Aluno",
                disciplina AS "Disciplina",
                av1 AS "AV1",
                av2 AS "AV2",
                av3 AS "AV3",
                media AS "Média",
                faltas AS "Faltas",
                horas AS "Horas de estudo",
                resultado AS "Resultado",
                confianca AS "Confiança",
                data_registro AS "Data do registro",
                data_analise AS "Data e horário da análise"
            FROM historico
            ORDER BY id DESC
        """, conexao)

    return df[COLUNAS_HISTORICO]

def salvar_historico(dados):
    with conectar_banco() as conexao:
        conexao.execute("""
            INSERT INTO historico (
                aluno, disciplina, av1, av2, av3,
                media, faltas, horas, resultado,
                confianca, data_registro, data_analise
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, dados)
        conexao.commit()

def limpar_historico():
    with conectar_banco() as conexao:
        conexao.execute("DELETE FROM historico")
        conexao.commit()

    vazio = pd.DataFrame(columns=COLUNAS_HISTORICO)
    return vazio, vazio, "Histórico apagado com sucesso."

criar_banco()

# ============================================================
# 3. CRIAÇÃO DOS DADOS EXPERIMENTAIS PARA O MODELO
# ============================================================

np.random.seed(42)

quantidade = 3000

dados_modelo = pd.DataFrame({
    "av1": np.random.uniform(0, 10, quantidade),
    "av2": np.random.uniform(0, 10, quantidade),
    "av3": np.random.uniform(0, 10, quantidade),
    "faltas": np.random.randint(0, 12, quantidade),
    "horas": np.random.uniform(0, 10, quantidade)
})

dados_modelo["media"] = (
    dados_modelo["av1"] +
    dados_modelo["av2"] +
    dados_modelo["av3"]
) / 3

# Regra experimental usada para gerar os exemplos:
# Média menor que 6 OU 5 faltas ou mais = REPROVADO.

dados_modelo["resultado"] = np.where(
    (dados_modelo["media"] >= MEDIA_MINIMA) &
    (dados_modelo["faltas"] < LIMITE_FALTAS),
    1,
    0
)

X = dados_modelo[
    ["av1", "av2", "av3", "faltas", "horas"]
]

y = dados_modelo["resultado"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

modelo = DecisionTreeClassifier(
    max_depth=8,
    min_samples_leaf=3,
    random_state=42
)

modelo.fit(X_treino, y_treino)

previsoes_teste = modelo.predict(X_teste)
acuracia = accuracy_score(y_teste, previsoes_teste)

# ============================================================
# 4. FUNÇÃO DE ANÁLISE DO ALUNO
# ============================================================

def analisar_aluno(
    nome,
    disciplina,
    av1,
    av2,
    av3,
    faltas,
    horas,
    data_registro
):

    try:
        nome = str(nome).strip()
        disciplina = str(disciplina).strip()

        if not nome:
            raise ValueError("Informe o nome do aluno.")

        if not disciplina:
            raise ValueError("Informe a disciplina.")

        if not data_registro:
            raise ValueError("Informe a data do registro.")

        av1 = float(av1)
        av2 = float(av2)
        av3 = float(av3)
        faltas = int(faltas)
        horas = float(horas)

        valores = [av1, av2, av3, horas]

        if not all(np.isfinite(v) for v in valores):
            raise ValueError("Digite somente números válidos.")

        if any(n < 0 or n > 10 for n in [av1, av2, av3]):
            raise ValueError("As notas devem estar entre 0 e 10.")

        if faltas < 0:
            raise ValueError("As faltas não podem ser negativas.")

        if horas < 0 or horas > 24:
            raise ValueError("As horas de estudo devem estar entre 0 e 24.")

        # Validação da data
        datetime.strptime(data_registro, "%d/%m/%Y")

        # Média das três avaliações
        media = round((av1 + av2 + av3) / 3, 2)

        # Previsão experimental do modelo
        entrada = pd.DataFrame([{
            "av1": av1,
            "av2": av2,
            "av3": av3,
            "faltas": faltas,
            "horas": horas
        }])

        probabilidades = modelo.predict_proba(entrada)[0]
        indice_aprovacao = list(modelo.classes_).index(1)
        confianca = round(float(probabilidades[indice_aprovacao]) * 100, 2)

        # Resultado final segue as regras definidas para o projeto.
        if faltas >= LIMITE_FALTAS:
            resultado = "REPROVADO"
            motivo = "Reprovado por frequência: atingiu 5 faltas ou mais."

        elif media < MEDIA_MINIMA:
            resultado = "REPROVADO"
            motivo = "Reprovado por nota: média abaixo de 6."

        else:
            resultado = "APROVADO"
            motivo = "Aprovado: média suficiente e menos de 5 faltas."

        # Horário da análise
        agora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        # Salvar no banco de dados
        dados = (
            nome, disciplina, av1, av2, av3,
            media, faltas, horas, resultado,
            confianca, data_registro, agora
        )

        salvar_historico(dados)

        # Atualizar as tabelas
        historico = buscar_historico()

        mensagem = f"""
## Resultado da análise

**Aluno:** {nome}

**Disciplina:** {disciplina}

**AV1:** {av1:.2f}

**AV2:** {av2:.2f}

**AV3:** {av3:.2f}

**Média final:** {media:.2f}

**Faltas:** {faltas}

**Horas de estudo:** {horas}

**Resultado:** {resultado}

**Motivo:** {motivo}

**Confiança estimada do modelo:** {confianca}%

*Essa confiança é uma estimativa experimental do modelo, não uma garantia de aprovação.*
"""

        return mensagem, historico, historico

    except ValueError as erro:
        return f"⚠️ {str(erro)}", buscar_historico(), buscar_historico()

    except Exception as erro:
        return (
            f"Erro durante a análise: {str(erro)}",
            buscar_historico(),
            buscar_historico()
        )

# ============================================================
# 5. HISTÓRICO
# ============================================================

def atualizar_historico():
    historico = buscar_historico()
    return historico, historico, "Histórico atualizado."

# ============================================================
# 6. MÉTRICAS DO MODELO
# ============================================================

def mostrar_metricas():
    matriz = confusion_matrix(y_teste, previsoes_teste)
    relatorio = classification_report(
        y_teste,
        previsoes_teste,
        target_names=["Reprovado", "Aprovado"],
        zero_division=0
    )

    importancia = pd.DataFrame({
        "Característica": X.columns,
        "Importância": modelo.feature_importances_
    }).sort_values("Importância", ascending=False)

    return (
        f"Acurácia experimental: {acuracia * 100:.2f}%",
        str(matriz),
        relatorio,
        importancia
    )

# ============================================================
# 7. INTERFACE GRADIO
# ============================================================

historico_inicial = buscar_historico()

css = """
.gradio-container {
    background: linear-gradient(135deg, #100d20, #21143c);
    color: white;
}
.gr-button-primary {
    background: linear-gradient(90deg, #7138d4, #a044ff) !important;
    border: none !important;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="violet"),
    css=css,
    title="EduPredict AI"
) as demo:

    gr.Markdown("""
    # 🎓 EDUPREDICT AI
    ### Classificador experimental de desempenho escolar

    Sistema de análise de notas, frequência e horas de estudo.
    """)

    with gr.Tab("📝 Analisar aluno"):

        gr.Markdown("### Dados do aluno")

        with gr.Row():
            nome = gr.Textbox(
                label="Nome do aluno",
                placeholder="Digite o nome"
            )

            disciplina = gr.Textbox(
                label="Disciplina",
                placeholder="Ex.: Matemática"
            )

        data_registro = gr.Textbox(
            label="Data do registro (DD/MM/AAAA)",
            value=datetime.now().strftime("%d/%m/%Y")
        )

        gr.Markdown("### Avaliações do IEMA")

        gr.Markdown("""
        - AV1: Primeira prova
        - AV2: Comportamento / atividade
        - AV3: Segunda prova
        """)

        with gr.Row():
            av1 = gr.Number(label="AV1 - Primeira prova", minimum=0, maximum=10)
            av2 = gr.Number(label="AV2 - Atividade", minimum=0, maximum=10)
            av3 = gr.Number(label="AV3 - Segunda prova", minimum=0, maximum=10)

        gr.Markdown("### Frequência e estudo")

        with gr.Row():
            faltas = gr.Number(
                label="Quantidade de faltas",
                value=0,
                precision=0
            )

            horas = gr.Number(
                label="Horas de estudo por dia",
                value=1,
                minimum=0,
                maximum=24
            )

        botao_analisar = gr.Button(
            "🔍 Analisar desempenho",
            variant="primary"
        )

        resultado = gr.Markdown()

        gr.Markdown("### Últimas análises")

        tabela_analise = gr.Dataframe(
            value=historico_inicial,
            headers=COLUNAS_HISTORICO,
            interactive=False
        )

    with gr.Tab("📚 Histórico completo"):

        gr.Markdown("### Histórico de alunos analisados")

        with gr.Row():
            botao_atualizar = gr.Button("🔄 Atualizar histórico")
            botao_limpar = gr.Button("🗑️ Apagar histórico")

        aviso_historico = gr.Markdown()

        tabela_historico = gr.Dataframe(
            value=historico_inicial,
            headers=COLUNAS_HISTORICO,
            interactive=False
        )

    with gr.Tab("📊 Métricas do modelo"):

        gr.Markdown("### Desempenho experimental do classificador")

        botao_metricas = gr.Button("Calcular métricas")

        texto_acuracia = gr.Markdown()
        matriz_resultado = gr.Textbox(label="Matriz de confusão")
        relatorio_resultado = gr.Textbox(label="Relatório de classificação")
        tabela_importancia = gr.Dataframe(label="Importância das características")

    with gr.Tab("ℹ️ Sobre o projeto"):

        gr.Markdown("""
        ## EduPredict AI

        Projeto escolar desenvolvido com Python e Machine Learning.

        **Tecnologias utilizadas:**
        - Python
        - Gradio
        - Pandas
        - NumPy
        - Scikit-learn
        - SQLite

        **Modelo:** Árvore de Decisão (Decision Tree).

        **Regras experimentais do projeto:**
        - Média mínima: 6,0.
        - Média máxima: 10,0.
        - Cinco faltas ou mais: reprovação por frequência.
        - Média abaixo de 6,0: reprovação por nota.

        **Atenção:** o modelo é experimental e foi treinado com dados
        gerados artificialmente. Não substitui os critérios oficiais
        da escola nem uma avaliação feita pelos professores.
        """)

    # ========================================================
    # 8. EVENTOS DOS BOTÕES
    # ========================================================

    botao_analisar.click(
        fn=analisar_aluno,
        inputs=[
            nome, disciplina, av1, av2, av3,
            faltas, horas, data_registro
        ],
        outputs=[
            resultado, tabela_analise, tabela_historico
        ]
    )

    botao_atualizar.click(
        fn=atualizar_historico,
        inputs=[],
        outputs=[
            tabela_analise, tabela_historico, aviso_historico
        ]
    )

    botao_limpar.click(
        fn=limpar_historico,
        inputs=[],
        outputs=[
            tabela_analise, tabela_historico, aviso_historico
        ]
    )

    botao_metricas.click(
        fn=mostrar_metricas,
        inputs=[],
        outputs=[
            texto_acuracia,
            matriz_resultado,
            relatorio_resultado,
            tabela_importancia
        ]
    )

# ============================================================
# 9. INICIAR SISTEMA
# ============================================================

demo.launch(
    share=True,
    auth=(USUARIO_ACESSO, SENHA_ACESSO)
)
