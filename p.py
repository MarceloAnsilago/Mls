import sqlite3
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid, GridOptionsBuilder
import yfinance as yf
import base64
import os
from datetime import datetime, timedelta
from PIL import Image
from statsmodels.tsa.stattools import coint
import numpy as np

# Configurações da Página
st.set_page_config(page_title="Gerenciamento de Ações", page_icon=":chart_with_upwards_trend:", layout="wide")
logo_image = Image.open("logos/LogoApp.jpg")

# Função para abrir a conexão com o banco de dados
def get_connection():
    conn = sqlite3.connect('cotacoes.db')
    return conn

# Função para atualizar cotações
def atualizar_cotações():
    with st.spinner('Atualizando cotações...'):
        conn = get_connection()
        cursor = conn.cursor()

        # Buscar todos os tickers no banco de dados
        cursor.execute("SELECT DISTINCT ticker FROM cotacoes")
        tickers = [row[0] for row in cursor.fetchall()]

        for ticker in tickers:
            # Verificar a data mais recente para cada ticker no banco de dados
            cursor.execute("SELECT MAX(data) FROM cotacoes WHERE ticker = ?", (ticker,))
            ultima_data = cursor.fetchone()[0]
            
            if ultima_data:
                # Se houver uma data no banco, buscar cotações a partir do dia seguinte
                start_date = datetime.strptime(ultima_data, "%Y-%m-%d") + timedelta(days=1)
            else:
                # Se não houver cotações, buscar desde um período padrão, por exemplo, 1 ano atrás
                start_date = datetime.now() - timedelta(days=365)

            # Buscar as cotações do Yahoo Finance a partir da data mais recente até hoje
            dados = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=datetime.now().strftime("%Y-%m-%d"))['Close']
            
            # Inserir novas cotações no banco de dados
            for data, fechamento in dados.items():
                cursor.execute('''
                INSERT OR IGNORE INTO cotacoes (ticker, data, fechamento) VALUES (?, ?, ?)
                ''', (ticker, data.strftime('%Y-%m-%d'), fechamento))
            
        conn.commit()
        conn.close()
    st.success("Cotações atualizadas com sucesso!")

# Adicionar um botão para atualização
if st.button("Atualizar Cotações"):
    atualizar_cotações()

# Função para carregar as cotações mais recentes
def carregar_acoes():
    conn = get_connection()
    query = """
    SELECT ticker, MAX(data) as data, fechamento 
    FROM cotacoes 
    GROUP BY ticker 
    ORDER BY ticker
    """
    cotacoes_df = pd.read_sql(query, conn)
    conn.close()
    return cotacoes_df

# Função para carregar o ícone
def carregar_icone(ticker):
    # Verifica se o quinto caractere é um número
    if len(ticker) >= 5 and ticker[4].isdigit():
        # Remove o sufixo ".SA" e o quinto caractere se for numérico
        ticker_base = ticker[:4]  # Mantém apenas os quatro primeiros caracteres
    else:
        ticker_base = ticker.replace(".SA", "")

    icon_path = f"logos/{ticker_base}.jpg"
    
    if os.path.exists(icon_path):
        try:
            with open(icon_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
                return f"data:image/jpg;base64,{encoded_string}"
        except Exception as e:
            print(f"Erro ao carregar a imagem {ticker_base}: {e}")
            return None
    else:
        print(f"Imagem não encontrada para: {ticker_base}")
        return None

# Função para encontrar pares cointegrados e calcular z-score
def find_cointegrated_pairs(data, zscore_threshold_upper, zscore_threshold_lower):
    n = data.shape[1]
    keys = data.keys()
    pairs = []
    pvalues = []
    zscores = []

    for i in range(n):
        for j in range(i + 1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            score, pvalue, _ = coint(S1, S2)
            
            if pvalue < 0.05:
                ratios = S1 / S2
                zscore = (ratios - ratios.mean()) / ratios.std()
                
                # Filtrar pelos limites de z-score
                if zscore.iloc[-1] > zscore_threshold_upper or zscore.iloc[-1] < zscore_threshold_lower:
                    pairs.append((keys[i], keys[j]))
                    pvalues.append(pvalue)
                    zscores.append(zscore.iloc[-1])

    return pairs, pvalues, zscores

# Função para carregar todas as cotações do banco de dados
def carregar_todas_cotacoes():
    conn = get_connection()
    query = """
    SELECT data, ticker, fechamento 
    FROM cotacoes 
    ORDER BY data DESC
    """
    cotacoes_df = pd.read_sql(query, conn)
    conn.close()
    return cotacoes_df

# Menu Lateral
with st.sidebar:
    st.image(logo_image, use_column_width=True)  # Exibir a imagem no menu lateral
    selected = option_menu(
        menu_title="Menu Principal",  # required
        options=["Página Inicial", "Cotações", "Análise"],  # required
        icons=["house", "currency-exchange", "graph-up-arrow"],  # ícones para cada página
        menu_icon="cast",  # ícone do menu
        default_index=0,  # seleciona a aba 'Página Inicial'
    )

# Página Inicial
if selected == "Página Inicial":
    st.title("Ações Acompanhadas")

    # Carregar as cotações mais recentes
    cotacoes_df = carregar_acoes()

    # Exibir as métricas em 5 colunas com espaçamento de 5px
    cols = st.columns(5, gap="small")

    for index, row in cotacoes_df.iterrows():
        ticker = row['ticker']
        ultimo_preco = row['fechamento']
        ultima_data = row['data']

        # Carregar o ícone da ação
        icone = carregar_icone(ticker)

        # Exibir a métrica no formato de cartão
        with cols[index % 5]:
            st.markdown(
                f"""
                <div style="border: 1px solid #ddd; border-radius: 10px; padding: 10px; text-align: center; background-color: #f9f9f9; height: 250px; margin-bottom: 15px; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <h4 style="margin: 0;">{ticker.replace(".SA", "")}</h4>
                        <hr style="border: none; border-top: 2px solid red; margin: 5px 0 10px 0;">
                    </div>
                    <div style="flex-grow: 1; display: flex; justify-content: center; align-items: center;">
                        <img src="{icone}" style="max-width: 100%; max-height: 80px; object-fit: contain;">
                    </div>
                    <div style="margin-top: 10px; text-align: center;">
                        <h6 style="font-size: 14px; color: #888; margin-bottom: 4px;">Última Cotação ({ultima_data})</h6>
                        <h3 style="margin: 0; font-size: 24px;">R$ {ultimo_preco:.2f}</h3>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# Página de Cotações
if selected == "Cotações":
    st.title("Cotações")

    # Expander com Formulário
    with st.expander("Adicionar Ação"):
        with st.form(key='add_stock_form'):
            nome_acao = st.text_input("Nome da Ação (Ticker)", help="Digite o código da ação, por exemplo, PETR4 para Petrobras.")
            periodos = st.number_input("Períodos (em dias)", min_value=1, max_value=365, value=150, help="Número de dias para baixar cotações históricas.")
            submit_button = st.form_submit_button(label="Adicionar Ação e Baixar Cotações")

        if submit_button:
            if nome_acao:
                nome_acao = nome_acao.upper()
                if not nome_acao.endswith(".SA"):
                    nome_acao += ".SA"
                
                if nome_acao in [acao[0] for acao in st.session_state.get('acoes_adicionadas', [])]:
                    st.warning(f"Ação {nome_acao} já foi adicionada anteriormente.")
                else:
                    try:
                        st.write(f"Baixando cotações para {nome_acao}...")
                        dados = yf.download(nome_acao, period=f"{periodos}d")['Close']
                        if dados.empty:
                            st.error(f"Erro: Nenhum dado encontrado para {nome_acao}. Verifique o ticker.")
                        else:
                            dados.name = nome_acao

                            # Salvar as cotações no banco de dados
                            conn = get_connection()
                            for data, fechamento in dados.items():
                                cursor = conn.cursor()
                                cursor.execute('''
                                INSERT OR IGNORE INTO cotacoes (ticker, data, fechamento) VALUES (?, ?, ?)
                                ''', (nome_acao, data.strftime('%Y-%m-%d'), fechamento))
                            conn.commit()
                            conn.close()

                            st.session_state.setdefault('acoes_adicionadas', []).append((nome_acao, periodos))
                            st.success(f"Ação {nome_acao} adicionada com {periodos} períodos.")
                    except Exception as e:
                        st.error(f"Erro ao baixar cotações para {nome_acao}: {e}")
            else:
                st.error("O nome da ação é obrigatório!")

    # Exibição das Cotações em uma Grade (Grid)
    st.subheader("Preços de Fechamento das Ações")
    conn = get_connection()
    query = "SELECT data, ticker, fechamento FROM cotacoes ORDER BY data DESC"
    cotacoes_df = pd.read_sql(query, conn)
    conn.close()

    if not cotacoes_df.empty:
        cotacoes_pivot = cotacoes_df.pivot(index='data', columns='ticker', values='fechamento')

        # Configurando a grade (grid) com st-aggrid
        gb = GridOptionsBuilder.from_dataframe(cotacoes_pivot.reset_index())
        gb.configure_pagination(paginationAutoPageSize=True)  # Habilitar paginação
        gb.configure_side_bar()  # Adicionar barra lateral para filtros
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, editable=True)

        gridOptions = gb.build()
        AgGrid(cotacoes_pivot.reset_index(), gridOptions=gridOptions, enable_enterprise_modules=True)
    else:
        st.write("Nenhuma cotação disponível ainda.")

# Página de Análise
if selected == "Análise":
    st.title("Análise de Cointegração de Ações")

    # Seleção de parâmetros para análise
    with st.form(key='analysis_form'):
        numero_periodos = st.number_input("Número de Períodos para Análise", min_value=1, value=120, help="Número de períodos (mais recentes) para considerar na análise de cointegração.")
        zscore_threshold_upper = st.number_input("Limite Superior do Z-Score", value=2.0)
        zscore_threshold_lower = st.number_input("Limite Inferior do Z-Score", value=-2.0)
        submit_button = st.form_submit_button(label="Analisar Pares Cointegrados")

    if submit_button:
        cotacoes_df = carregar_todas_cotacoes()

        # Transformar os dados em um formato adequado para a cointegração
        cotacoes_pivot = cotacoes_df.pivot(index='data', columns='ticker', values='fechamento')

        # Selecionar os últimos `numero_periodos` (mais recentes)
        cotacoes_pivot = cotacoes_pivot.tail(numero_periodos)

        # Verificar o número de períodos que realmente foram selecionados
        numero_de_periodos_selecionados = cotacoes_pivot.shape[0]
        st.write(f"Número de períodos selecionados para análise: {numero_de_periodos_selecionados}")

        # Encontrar os pares cointegrados e calcular z-scores
        pairs, pvalues, zscores = find_cointegrated_pairs(cotacoes_pivot, zscore_threshold_upper, zscore_threshold_lower)

        if pairs:
            # Criar DataFrame para exibir os resultados
            resultados_df = pd.DataFrame({
                'Pair': [f"{pair[0]} - {pair[1]}" for pair in pairs],
                'p-value': pvalues,
                'Z-Score': zscores,
                'Selecionar': [False] * len(pairs)  # Checkbox inicializando como False
            })

            st.subheader("Pares Cointegrados Encontrados (Z-Score fora dos limites):")
            
            # Configurando a grade (grid) com st-aggrid
            gb = GridOptionsBuilder.from_dataframe(resultados_df)
            gb.configure_pagination(paginationAutoPageSize=True)
            gb.configure_side_bar()
            gb.configure_selection(selection_mode="multiple", use_checkbox=True)
            gridOptions = gb.build()
            
            grid_response = AgGrid(
                resultados_df,
                gridOptions=gridOptions,
                enable_enterprise_modules=True,
                update_mode="MODEL_CHANGED",
                allow_unsafe_jscode=True,
                fit_columns_on_grid_load=True,
            )

            # Obter as linhas selecionadas
            selected_rows = grid_response['selected_rows']
            if selected_rows:
                st.write("Pares Selecionados para Análise:")
                for row in selected_rows:
                    st.write(f"Par: {row['Pair']} | p-value: {row['p-value']:.5f} | Z-Score: {row['Z-Score']:.2f}")
            else:
                st.write("Nenhum par selecionado.")
        else:
            st.write("Nenhum par cointegrado encontrado.")
