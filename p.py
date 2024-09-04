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
import statsmodels.api as sm
from hurst import compute_Hc
import matplotlib.pyplot as plt

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

# Função para calcular Half-Life
def half_life_calc(ts):
    lagged = ts.shift(1).fillna(method="bfill")
    delta = ts - lagged
    X = sm.add_constant(lagged.values)
    ar_res = sm.OLS(delta, X).fit()
    half_life = -1 * np.log(2) / ar_res.params[1]
    return half_life

# Função para calcular Hurst exponent
def hurst_exponent(ts):
    H, c, data = compute_Hc(ts, kind='price', simplified=True)
    return H

# Função para calcular Beta Rotation (ang. cof)
def beta_rotation(series_x, series_y, window=40):
    beta_list = []
    try:
        for i in range(0, len(series_x) - window):
            slice_x = series_x[i:i + window]
            slice_y = series_y[i:i + window]
            X = sm.add_constant(slice_x.values)
            mod = sm.OLS(slice_y, X)
            results = mod.fit()
            beta = results.params[1]
            beta_list.append(beta)
    except Exception as e:
        st.error(f"Erro ao calcular beta rotation: {e}")
        raise

    return beta_list[-1]  # Return the most recent beta value

# Função para encontrar pares cointegrados e calcular z-score, half-life, Hurst, ang. cof
def find_cointegrated_pairs(data, zscore_threshold_upper, zscore_threshold_lower):
    n = data.shape[1]
    keys = data.keys()
    pairs = []
    pvalues = []
    zscores = []
    half_lives = []
    hursts = []
    beta_rotations = []

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
                    half_lives.append(half_life_calc(ratios))
                    hursts.append(hurst_exponent(ratios))
                    beta_rotations.append(beta_rotation(S1, S2))

    return pairs, pvalues, zscores, half_lives, hursts, beta_rotations

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
        numero_periodos = st.number_input(
            "Número de Períodos para Análise",
            min_value=1,
            value=120,
            help="Número de períodos (mais recentes) para considerar na análise de cointegração."
        )
        zscore_threshold_upper = st.number_input("Limite Superior do Z-Score", value=2.0)
        zscore_threshold_lower = st.number_input("Limite Inferior do Z-Score", value=-2.0)
        submit_button = st.form_submit_button(label="Analisar Pares Cointegrados")

    if submit_button or 'cotacoes_pivot' in st.session_state:
        if submit_button:
            cotacoes_df = carregar_todas_cotacoes()

            # Transformar os dados em um formato adequado para a cointegração
            cotacoes_pivot = cotacoes_df.pivot(index='data', columns='ticker', values='fechamento')

            # Selecionar os últimos `numero_periodos` (mais recentes)
            cotacoes_pivot = cotacoes_pivot.tail(numero_periodos)

            # Armazenar no session state
            st.session_state['cotacoes_pivot'] = cotacoes_pivot

        # Pegar do session state se existir
        cotacoes_pivot = st.session_state['cotacoes_pivot']

        # Verificar o número de períodos que realmente foram selecionados
        numero_de_periodos_selecionados = cotacoes_pivot.shape[0]
        st.write(f"Número de períodos selecionados para análise: {numero_de_periodos_selecionados}")

        # Adiciona um separador e o título "Pares Encontrados"
        st.markdown("---")  # Separador
        st.subheader("Pares Encontrados")

        # Encontrar os pares cointegrados e calcular z-scores, half-lives, hurst, beta rotations
        pairs, pvalues, zscores, half_lives, hursts, beta_rotations = find_cointegrated_pairs(
            cotacoes_pivot, zscore_threshold_upper, zscore_threshold_lower
        )

        if pairs:
            # Criar uma lista de pares com todas as métricas (Z-Score, P-Value, Hurst, Beta, Half-Life)
            for idx, (pair, zscore, pvalue, hurst, beta, half_life) in enumerate(zip(pairs, zscores, pvalues, hursts, beta_rotations, half_lives)):
                par_str = f"{pair[0]} - {pair[1]}"
                metricas_str = f"Z-Score: {zscore:.2f} | P-Value: {pvalue:.4f} | Hurst: {hurst:.4f} | Beta: {beta:.4f} | Half-Life: {half_life:.2f}"

                # Botão para exibir todas as métricas com o par
                if st.button(f"{par_str} | {metricas_str}", key=f"btn_{idx}"):
                    st.session_state['par_selecionado'] = pair

        # Exibe o gráfico apenas se houver um par selecionado
        if 'par_selecionado' in st.session_state:
            pair_selected = st.session_state['par_selecionado']
            par_str = f"{pair_selected[0]} - {pair_selected[1]}"
            metricas_str = f"Z-Score: {zscores[pairs.index(pair_selected)]:.2f} | P-Value: {pvalues[pairs.index(pair_selected)]:.4f} | Hurst: {hursts[pairs.index(pair_selected)]:.4f} | Beta: {beta_rotations[pairs.index(pair_selected)]:.4f} | Half-Life: {half_lives[pairs.index(pair_selected)]:.2f}"

            # Exibir o par escolhido, suas métricas e o gráfico do Z-Score
            st.markdown(f"<h4>{par_str} | {metricas_str}</h4>", unsafe_allow_html=True)

            # Exibir o gráfico do z-score para o par selecionado
            S1 = cotacoes_pivot[pair_selected[0]]
            S2 = cotacoes_pivot[pair_selected[1]]
            ratios = S1 / S2
            zscore_series = (ratios - ratios.mean()) / ratios.std()

            # st.subheader(f"Gráfico do Z-Score para o par:")
            plt.figure(figsize=(10, 5))
            plt.plot(zscore_series, label='Z-Score')
            plt.axhline(0, color='black', linestyle='--')
            plt.axhline(2, color='red', linestyle='--')
            plt.axhline(-2, color='green', linestyle='--')
            plt.legend(loc='best')
            plt.xlabel('Data')
            plt.ylabel('Z-Score')
            st.pyplot(plt)

        else:
            st.write("Nenhum par cointegrado encontrado.")
