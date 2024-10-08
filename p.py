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
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LinearRegression
import time
import mplfinance as mpf

# Configurações da Página
st.set_page_config(page_title="Gerenciamento de Ações", page_icon=":chart_with_upwards_trend:", layout="wide")
logo_image = Image.open("logos/LogoApp.jpg")

# Função para abrir a conexão com o banco de dados
def get_connection():
    conn = sqlite3.connect('cotacoes.db')
    return conn

# Função para atualizar cotações
def atualizar_cotacoes():
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
    atualizar_cotacoes()

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

# Função para calcular o beta móvel em uma janela deslizante
def calcular_beta_movel(S1, S2, window=40):
    returns_S1 = np.log(S1 / S1.shift(1)).dropna()
    returns_S2 = np.log(S2 / S2.shift(1)).dropna()

    betas = []
    index_values = returns_S1.index[window-1:]  # Ajustar para a janela

    for i in range(window, len(returns_S1) + 1):
        reg = LinearRegression().fit(returns_S2[i-window:i].values.reshape(-1, 1), returns_S1[i-window:i].values)
        betas.append(reg.coef_[0])

    return pd.Series(betas, index=index_values)

# Exibir o gráfico de beta móvel
def plotar_beta_movel(S1, S2, window=40):
    beta_movel = calcular_beta_movel(S1, S2, window)

    plt.figure(figsize=(10, 5))
    plt.plot(beta_movel, label=f'Beta Móvel ({window} períodos)')
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f'Beta Móvel ({window} períodos')
    plt.xlabel('Data')
    plt.ylabel('Beta')
    plt.legend()

    # Ajustar as datas para exibição transversal e diminuir a fonte
    plt.xticks(rotation=45, fontsize=6)
    
    # Reduzir a frequência dos rótulos exibidos no eixo X
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))

    plt.grid(True)
    st.pyplot(plt)

# Exibir o gráfico de dispersão entre os dois ativos
def plotar_grafico_dispersao(S1, S2):
    plt.figure(figsize=(10, 5))
    plt.scatter(S1, S2)
    plt.title(f'Dispersão entre {S1.name} e {S2.name}')
    plt.xlabel(f'{S1.name}')
    plt.ylabel(f'{S2.name}')
    plt.grid(True)
    st.pyplot(plt)

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
# Função para obter o preço atual de uma ação usando yfinance

def obter_preco_atual(ticker):
    dados = yf.download(ticker, period="1d")  # Baixar o dado mais recente
    if not dados.empty:
        return dados['Close'].iloc[-1]  # Retornar o preço de fechamento mais recente
    else:
        return None
# Função para plotar o gráfico do Z-Score
def plotar_grafico_zscore(S1, S2):
    ratios = S1 / S2
    zscore_series = (ratios - ratios.mean()) / ratios.std()

    plt.figure(figsize=(10, 5))
    plt.plot(zscore_series, label='Z-Score')
    plt.axhline(0, color='black', linestyle='--')
    plt.axhline(2, color='red', linestyle='--')
    plt.axhline(-2, color='green', linestyle='--')
    plt.legend(loc='best')
    plt.xlabel('Data')
    plt.ylabel('Z-Score')
    plt.xticks(rotation=45, fontsize=6)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
    st.pyplot(plt)

    # Função para plotar o gráfico dos preços das ações
def plotar_grafico_precos(S1, S2, ticker1, ticker2):
    plt.figure(figsize=(10, 5))
    plt.plot(S1, label=ticker1)
    plt.plot(S2, label=ticker2)
    plt.legend(loc='best')
    plt.xlabel('Data')
    plt.ylabel('Preço de Fechamento')
    plt.xticks(rotation=45, fontsize=6)
    st.pyplot(plt)


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
            S1 = data[keys[i]].dropna()  # Remover NaNs de S1
            S2 = data[keys[j]].dropna()  # Remover NaNs de S2

            # Garantir que ambas as séries tenham o mesmo comprimento após a remoção dos NaNs
            combined = pd.concat([S1, S2], axis=1).dropna()
            if len(combined) < 2:  # Verificar se ainda há dados suficientes
                continue

            S1 = combined.iloc[:, 0]
            S2 = combined.iloc[:, 1]

            try:
                score, pvalue, _ = coint(S1, S2)
                if pvalue < 0.05:
                    ratios = S1 / S2
                    zscore = (ratios - ratios.mean()) / ratios.std()

                    if zscore.iloc[-1] > zscore_threshold_upper or zscore.iloc[-1] < zscore_threshold_lower:
                        pairs.append((keys[i], keys[j]))
                        pvalues.append(pvalue)
                        zscores.append(zscore.iloc[-1])
                        half_lives.append(half_life_calc(ratios))
                        hursts.append(hurst_exponent(ratios))
                        beta_rotations.append(beta_rotation(S1, S2))

            except Exception as e:
                print(f"Erro ao calcular cointegração para {keys[i]} e {keys[j]}: {e}")
                continue

    return pairs, pvalues, zscores, half_lives, hursts, beta_rotations
# Função para criar cards explicativos das métricas
def criar_card_metrica(nome_metrica, valor_metrica, descricao):
    st.markdown(
        f"""
        <div style="border: 1px solid #ddd; border-radius: 10px; padding: 10px; text-align: center; background-color: #f9f9f9; height: 250px; margin-bottom: 15px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <h4 style="margin: 0;">{nome_metrica}</h4>
                <hr style="border: none; border-top: 2px solid red; margin: 5px 0 10px 0;">
            </div>
            <div style="flex-grow: 1; display: flex; justify-content: center; align-items: center;">
                <h2 style="margin: 0; font-size: 24px;">{valor_metrica}</h2>
            </div>
            <div style="margin-top: 10px; text-align: center;">
                <p style="font-size: 14px; color: #888; margin-bottom: 4px;">{descricao}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
# Função para obter o preço atual de uma ação usando yfinance
def obter_preco_atual(ticker):
    dados = yf.download(ticker, period="1d")  # Baixar o dado mais recente
    if not dados.empty:
        return dados['Close'].iloc[-1]  # Retornar o preço de fechamento mais recente
    else:
        return None

# Função para plotar o gráfico do Z-Score
def plotar_grafico_zscore(S1, S2):
    ratios = S1 / S2
    zscore_series = (ratios - ratios.mean()) / ratios.std()

    plt.figure(figsize=(10, 5))
    plt.plot(zscore_series, label='Z-Score')
    plt.axhline(0, color='black', linestyle='--')
    plt.axhline(2, color='red', linestyle='--')
    plt.axhline(-2, color='green', linestyle='--')
    plt.legend(loc='best')
    plt.xlabel('Data')
    plt.ylabel('Z-Score')
    plt.xticks(rotation=45, fontsize=6)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
    st.pyplot(plt)

# Função para exibir a métrica no formato de cartão
def exibir_metrica_cartao(ticker, ultimo_preco, ultima_data, icone=None):
    st.markdown(
        f"""
        <div style="border: 1px solid #ddd; border-radius: 10px; padding: 10px; text-align: center; background-color: #f9f9f9; height: 250px; margin-bottom: 15px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <h4 style="margin: 0;">{ticker.replace(".SA", "")}</h4>
                <hr style="border: none; border-top: 2px solid red; margin: 5px 0 10px 0;">
            </div>
            <div style="flex-grow: 1; display: flex; justify-content: center; align-items: center;">
                {f'<img src="{icone}" style="max-width: 100%; max-height: 80px; object-fit: contain;">' if icone else ''}
            </div>
            <div style="margin-top: 10px; text-align: center;">
                <h6 style="font-size: 14px; color: #888; margin-bottom: 4px;">Última Cotação ({ultima_data})</h6>
                <h3 style="margin: 0; font-size: 24px;">R$ {ultimo_preco:.2f}</h3>
            </div>
        </div>
        """, unsafe_allow_html=True
    )

# Menu Lateral
with st.sidebar:
    st.image(logo_image, use_column_width=True)  # Exibir a imagem no menu lateral
    selected = option_menu(
        menu_title="Menu Principal",  # required
        options=["Página Inicial", "Cotações", "Análise", "Operações"],  # required
        icons=["house", "currency-exchange", "graph-up-arrow", "briefcase"],  # ícones para cada página
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

        # Exibir a métrica no formato de cartão usando a função reutilizável
        with cols[index % 5]:
            exibir_metrica_cartao(ticker, ultimo_preco, ultima_data, icone)


# Página de Cotações
if selected == "Cotações":
    st.title("Cotações")
    # Expander com Formulário
    with st.expander("Adicionar Ação"):
        with st.form(key='add_stock_form'):
            nome_acao = st.text_input("Nome da Ação (Ticker)", help="Digite o código da ação, por exemplo, PETR4 para Petrobras.")
            periodos = st.number_input("Períodos (em dias)", min_value=1, max_value=365, value=200, help="Número de dias para baixar cotações históricas.")
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
            # Selecionar os últimos numero_periodos (mais recentes)
            cotacoes_pivot = cotacoes_pivot.tail(numero_periodos)
            # Armazenar no session state
            st.session_state['cotacoes_pivot'] = cotacoes_pivot
        # Pegar do session state se existir
        cotacoes_pivot = st.session_state['cotacoes_pivot']
        # Verificar o número de períodos que realmente foram selecionados
        numero_de_periodos_selecionados = cotacoes_pivot.shape[0]
        st.write(f"Número de períodos selecionados para análise: {numero_de_periodos_selecionados}")
        # Adiciona um separador e o título "Pares Encontrados"
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
            st.markdown("---")  # Separador
            if 'par_selecionado' in st.session_state:
                pair_selected = st.session_state['par_selecionado']
                par_str = f"{pair_selected[0]} - {pair_selected[1]}"
                metricas_str = f"Z-Score: {zscores[pairs.index(pair_selected)]:.2f} | P-Value: {pvalues[pairs.index(pair_selected)]:.4f} | Hurst: {hursts[pairs.index(pair_selected)]:.4f} | Beta: {beta_rotations[pairs.index(pair_selected)]:.4f} | Half-Life: {half_lives[pairs.index(pair_selected)]:.2f}"
                # Centralizar o título usando HTML
                st.markdown(f"<h4 style='text-align: center;'>{par_str} | {metricas_str}</h4>", unsafe_allow_html=True)
                # Configurando as colunas
                col1, col2 = st.columns(2)
                with col1:
                    # Gráfico do Z-Score
                    S1 = cotacoes_pivot[pair_selected[0]]
                    S2 = cotacoes_pivot[pair_selected[1]]
                    ratios = S1 / S2
                    zscore_series = (ratios - ratios.mean()) / ratios.std()
                    plt.figure(figsize=(10, 5))  # Tamanho ajustado
                    plt.plot(zscore_series, label='Z-Score')
                    plt.axhline(0, color='black', linestyle='--')
                    plt.axhline(2, color='red', linestyle='--')
                    plt.axhline(-2, color='green', linestyle='--')
                    plt.legend(loc='best')
                    plt.xlabel('Data')
                    plt.ylabel('Z-Score')
                    plt.xticks(rotation=45, fontsize=6)
                    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
                    st.pyplot(plt)
                    # Gráfico de Beta Móvel logo abaixo do Z-Score
                    st.subheader(f"Beta Móvel para {pair_selected[0]} e {pair_selected[1]}")
                    plotar_beta_movel(S1, S2, window=40)
                with col2:
                    # Gráfico de paridades (cotação dos dois ativos)
                    plt.figure(figsize=(10, 5))
                    plt.plot(S1 / S1.iloc[0], label=f"{pair_selected[0]}")
                    plt.plot(S2 / S2.iloc[0], label=f"{pair_selected[1]}")
                    plt.legend(loc='best')
                    plt.xlabel('Data')
                    plt.xticks(rotation=45, fontsize=6)
                    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
                    # Ajustar diretamente o índice de datas no eixo X
                    plt.gca().set_xticks(range(0, len(cotacoes_pivot.index), max(1, len(cotacoes_pivot.index) // 6)))
                    plt.gca().set_xticklabels([pd.to_datetime(date).strftime('%Y-%m-%d') for date in cotacoes_pivot.index[::max(1, len(cotacoes_pivot.index) // 6)]], rotation=45)
                    st.pyplot(plt)
                    # Gráfico de Dispersão logo abaixo do gráfico de paridade
                    st.subheader(f"Dispersão entre {pair_selected[0]} e {pair_selected[1]}")
                    plotar_grafico_dispersao(S1, S2)


                # Adicionar o botão "Salvar Par para Operação"
                if st.button("Salvar Par para Operação"):
                    conn = get_connection()
                    cursor = conn.cursor()

                    # Criar a tabela no banco de dados, se ainda não existir, com as novas colunas
                    cursor.execute('''
                    CREATE TABLE IF NOT EXISTS operacoes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        par TEXT,
                        zscore REAL,
                        pvalue REAL,
                        hurst REAL,
                        beta REAL,
                        half_life REAL,
                        preco_inicial_acao1 REAL,
                        preco_inicial_acao2 REAL,
                        preco_final_acao1 REAL,
                        preco_final_acao2 REAL,
                        qtd_acoes1 INTEGER,
                        qtd_acoes2 INTEGER,
                        data_inicio TEXT,
                        data_encerramento TEXT,
                        resultado REAL,
                        status TEXT,
                        data TEXT
                    )
                    ''')

                    # Inserir os dados do par atual na tabela "operacoes" com status "analise" e data atual
                    data_atual = datetime.now().strftime('%Y-%m-%d')
                    preco_inicial_acao1 = obter_preco_atual(pair_selected[0])
                    preco_inicial_acao2 = obter_preco_atual(pair_selected[1])
                    
                    # Adicionar inputs para capturar a quantidade de ações
                    qtd_acoes1 = 0
                    qtd_acoes2 = 0

                    cursor.execute('''
                    INSERT INTO operacoes (par, zscore, pvalue, hurst, beta, half_life, preco_inicial_acao1, preco_inicial_acao2, qtd_acoes1, qtd_acoes2, status, data_inicio, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (par_str, zscore, pvalue, hurst, beta, half_life, preco_inicial_acao1, preco_inicial_acao2, qtd_acoes1, qtd_acoes2, 'analise', data_atual, data_atual))

                    conn.commit()
                    conn.close()

                    st.success(f"Par {par_str} salvo com sucesso para operação na data {data_atual}!")
                    

                # Adicionar um separador entre os gráficos e as métricas
                st.markdown("---") 

                # Agora adicionamos os 5 cards com as métricas logo abaixo dos gráficos
                st.subheader("Métricas Explicativas")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    criar_card_metrica(
                        "Z-Score", 
                        f"{zscores[pairs.index(pair_selected)]:.2f}", 
                        "O Z-Score mede quantos desvios padrão o ativo está de sua média histórica."
                    )
                with col2:
                    criar_card_metrica(
                        "P-Value", 
                        f"{pvalues[pairs.index(pair_selected)]:.4f}", 
                        "O P-Value indica a probabilidade de a relação entre os ativos ocorrer por acaso."
                    )
                with col3:
                    criar_card_metrica(
                        "Hurst", 
                        f"{hursts[pairs.index(pair_selected)]:.4f}", 
                        "O Exponente de Hurst avalia a tendência de reversão à média."
                    )
                with col4:
                    criar_card_metrica(
                        "Beta", 
                        f"{beta_rotations[pairs.index(pair_selected)]:.4f}", 
                        "O Beta mede a sensibilidade de um ativo em relação a outro."
                    )
                with col5:
                    criar_card_metrica(
                        "Half-Life", 
                        f"{half_lives[pairs.index(pair_selected)]:.2f}", 
                        "O Half-Life é o tempo estimado para que a diferença entre dois ativos cointegrados reverta à média."
                    )

            else:
                st.write("Nenhum par cointegrado encontrado.")












if selected == "Operações":
    st.title("Operações")

    # Opções de status (analise, aberta, fechada)
    status_opcoes = ["Analise", "Aberta", "Fechada"]
    status_selecionado = st.radio("Selecione o status da operação:", status_opcoes, key="unique_status_radio_operacoes")
    status_mapeado = status_selecionado.lower()

    # Consulta ao banco de dados para buscar pares com o status selecionado
    conn = get_connection()
    query = """
    SELECT par, zscore, pvalue, hurst, beta, half_life, qtd_acoes1, qtd_acoes2, preco_inicial_acao1, preco_inicial_acao2, data_inicio 
    FROM operacoes
    WHERE status = ?
    """
    operacoes_df = pd.read_sql(query, conn, params=[status_mapeado])
    conn.close()

    # Se houver pares disponíveis para o status selecionado
    if not operacoes_df.empty:
        # Criar um selectbox para escolher o par
        pares_disponiveis = operacoes_df['par'].tolist()
        par_selecionado = st.selectbox(f"Selecione um par com status '{status_selecionado}':", pares_disponiveis, key="unique_par_selectbox")

        # Buscar os detalhes do par selecionado
        par_detalhes = operacoes_df[operacoes_df['par'] == par_selecionado].iloc[0]
        ticker1, ticker2 = par_selecionado.split(" - ")

        # Recuperar numero_periodos da análise ou definir um padrão
        if 'numero_periodos' in st.session_state:
            numero_periodos = st.session_state['numero_periodos']
        else:
            numero_periodos = 120  # Valor padrão se não estiver no session_state

        # Baixar dados atuais para os dois tickers, garantindo que tenham o mesmo número de períodos
        periodo = f"{numero_periodos}d"
        S1 = yf.download(ticker1, period=periodo)['Close']
        S2 = yf.download(ticker2, period=periodo)['Close']

        # Verificar se os dados foram baixados corretamente
        if S1.empty or S2.empty:
            st.error("Erro ao baixar os dados dos tickers selecionados.")
            st.stop()

        # Exibir os gráficos e permitir iniciar a operação se for status "Analise"
        if status_selecionado == "Analise":
            # Gráfico do Z-Score
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"### Gráfico do Z-Score: {par_selecionado}")
                ratios = S1 / S2
                zscore_series = (ratios - ratios.mean()) / ratios.std()
                plt.figure(figsize=(10, 5))  # Tamanho ajustado
                plt.plot(zscore_series, label='Z-Score')
                plt.axhline(0, color='black', linestyle='--')
                plt.axhline(2, color='red', linestyle='--')
                plt.axhline(-2, color='green', linestyle='--')
                plt.legend(loc='best')
                plt.xlabel('Data')
                plt.ylabel('Z-Score')
                plt.xticks(rotation=45, fontsize=6)
                st.pyplot(plt)

            # Gráfico de Preços Normalizados
            with col2:
                st.markdown(f"### Gráfico de Preços Normalizados: {ticker1} e {ticker2}")
                S1_normalizado = S1 / S1.iloc[0]
                S2_normalizado = S2 / S2.iloc[0]
                plt.figure(figsize=(10, 5))
                plt.plot(S1_normalizado, label=ticker1)
                plt.plot(S2_normalizado, label=ticker2)
                plt.legend(loc='best')
                plt.xlabel('Data')
                plt.ylabel('Preço Normalizado')
                plt.xticks(rotation=45, fontsize=6)
                st.pyplot(plt)

            # Expander: Calculadora de Operação
            with st.expander("Calculadora de Operação", expanded=True):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("### Escolher Lotes")
                    lotes_vendidos = st.number_input(f"Quantidade de Ações Vendidas ({ticker1}):", min_value=1, step=1, value=100, key="lotes_vendidos")
                    lotes_comprados = st.number_input(f"Quantidade de Ações Compradas ({ticker2}):", min_value=1, step=1, value=100, key="lotes_comprados")

                with col2:
                    # Calcula o valor da venda e compra
                    preco_venda = obter_preco_atual(ticker1) * lotes_vendidos
                    preco_compra = obter_preco_atual(ticker2) * lotes_comprados
                    saldo = preco_venda - preco_compra
                    st.markdown(f"**Valor Vendido ({ticker1}):** R$ {preco_venda:.2f}")
                    st.markdown(f"**Valor Comprado ({ticker2}):** R$ {preco_compra:.2f}")
                    st.markdown(f"**Saldo:** R$ {saldo:.2f}")

                with col3:
                    # Formulário para iniciar operação na terceira coluna
                    with st.form("iniciar_operacao"):
                        st.markdown(f"### Iniciar Operação")
                        st.markdown(f"**Preço Atual de {ticker1}:** R$ {preco_venda / lotes_vendidos:.2f}")
                        st.markdown(f"**Preço Atual de {ticker2}:** R$ {preco_compra / lotes_comprados:.2f}")

                        iniciar_button = st.form_submit_button("Iniciar Operação")
                        if iniciar_button:
                            # Atualizar operação no banco de dados
                            conn = get_connection()
                            cursor = conn.cursor()
                            cursor.execute('''
                            UPDATE operacoes
                            SET qtd_acoes1 = ?, qtd_acoes2 = ?, preco_inicial_acao1 = ?, preco_inicial_acao2 = ?, status = ?, data_inicio = ?
                            WHERE par = ?
                            ''', (lotes_vendidos, lotes_comprados, preco_venda / lotes_vendidos, preco_compra / lotes_comprados, 'aberta', datetime.now().strftime('%Y-%m-%d'), par_selecionado))
                            conn.commit()
                            conn.close()
                            st.success(f"Operação iniciada com sucesso para o par {par_selecionado}!")

                # Botão para cancelar a análise (fora do formulário)
                cancelar_button = st.button("Cancelar Análise")
                if cancelar_button:
                    conn = get_connection()
                    cursor = conn.cursor()
                    # Excluir a operação do banco de dados
                    cursor.execute("DELETE FROM operacoes WHERE par = ?", (par_selecionado,))
                    conn.commit()
                    conn.close()
                    st.success(f"A operação para o par {par_selecionado} foi cancelada e removida do banco de dados.")

        elif status_selecionado == "Aberta":
            st.markdown(f"### Operação Aberta: {par_selecionado}")
            st.markdown(f"**Data de Início:** {par_detalhes['data_inicio']}")

            # Colunas para exibir as informações de cada ação
            col1, col2 = st.columns(2)

            with col1:
                # Exibir detalhes da ação 1
                st.markdown(f"### {ticker1}")
                st.markdown(f"**Quantidade de Ações Vendidas:** {par_detalhes['qtd_acoes1']}")
                st.markdown(f"**Preço Inicial de {ticker1}:** R$ {par_detalhes['preco_inicial_acao1']:.2f}")
                valor_inicial_acao1 = par_detalhes['preco_inicial_acao1'] * par_detalhes['qtd_acoes1']
                st.markdown(f"**Valor Inicial ({ticker1}):** R$ {valor_inicial_acao1:.2f}")

                # Obter e exibir o preço atual
                preco_atual_acao1 = obter_preco_atual(ticker1)
                st.markdown(f"**Preço Atual de {ticker1}:** R$ {preco_atual_acao1:.2f}")
                valor_atual_acao1 = preco_atual_acao1 * par_detalhes['qtd_acoes1']
                st.markdown(f"**Valor Atual ({ticker1}):** R$ {valor_atual_acao1:.2f}")

                # Diferença de valor para a ação 1
                diferenca_acao1 = valor_atual_acao1 - valor_inicial_acao1
                st.markdown(f"**Diferença de Valor ({ticker1}):** R$ {diferenca_acao1:.2f}")

            with col2:
                # Exibir detalhes da ação 2
                st.markdown(f"### {ticker2}")
                st.markdown(f"**Quantidade de Ações Compradas:** {par_detalhes['qtd_acoes2']}")
                st.markdown(f"**Preço Inicial de {ticker2}:** R$ {par_detalhes['preco_inicial_acao2']:.2f}")
                valor_inicial_acao2 = par_detalhes['preco_inicial_acao2'] * par_detalhes['qtd_acoes2']
                st.markdown(f"**Valor Inicial ({ticker2}):** R$ {valor_inicial_acao2:.2f}")

                # Obter e exibir o preço atual
                preco_atual_acao2 = obter_preco_atual(ticker2)
                st.markdown(f"**Preço Atual de {ticker2}:** R$ {preco_atual_acao2:.2f}")
                valor_atual_acao2 = preco_atual_acao2 * par_detalhes['qtd_acoes2']
                st.markdown(f"**Valor Atual ({ticker2}):** R$ {valor_atual_acao2:.2f}")

                # Diferença de valor para a ação 2
                diferenca_acao2 = valor_atual_acao2 - valor_inicial_acao2
                st.markdown(f"**Diferença de Valor ({ticker2}):** R$ {diferenca_acao2:.2f}")

            # Calcular o resultado total da operação
            valor_total_inicial = valor_inicial_acao1 + valor_inicial_acao2
            valor_total_atual = valor_atual_acao1 + valor_atual_acao2
            resultado_total = valor_total_atual - valor_total_inicial

            # Exibir o resultado total
            st.markdown(f"### Resultado Total da Operação")
            st.markdown(f"**Valor Total Inicial:** R$ {valor_total_inicial:.2f}")
            st.markdown(f"**Valor Total Atual:** R$ {valor_total_atual:.2f}")
            st.markdown(f"**Resultado da Operação:** R$ {resultado_total:.2f}")

            # Gráfico do Z-Score Atualizado
            st.markdown(f"### Acompanhamento do Z-Score Atual: {par_selecionado}")
            
            # Baixar dados atuais dos dois tickers para acompanhamento
            S1 = yf.download(ticker1, period='120d')['Close']
            S2 = yf.download(ticker2, period='120d')['Close']
            
            if S1.empty or S2.empty:
                st.error("Erro ao baixar os dados para acompanhamento do Z-Score.")
            else:
                # Calcular o Z-Score com dados atualizados
                ratios = S1 / S2
                zscore_series = (ratios - ratios.mean()) / ratios.std()

                # Plotar o gráfico do Z-Score
                plt.figure(figsize=(10, 3))  # Tamanho ajustado para 10x3
                plt.plot(zscore_series, label='Z-Score Atual')
                plt.axhline(0, color='black', linestyle='--')
                plt.axhline(2, color='red', linestyle='--')
                plt.axhline(-2, color='green', linestyle='--')
                plt.legend(loc='best')
                plt.xlabel('Data')
                plt.ylabel('Z-Score')
                plt.xticks(rotation=45, fontsize=6)
                st.pyplot(plt)

                # Gráfico de Preços Normalizados logo abaixo do Z-Score
                st.markdown(f"### Gráfico de Preços Normalizados: {ticker1} e {ticker2}")
                S1_normalizado = S1 / S1.iloc[0]
                S2_normalizado = S2 / S2.iloc[0]
                plt.figure(figsize=(10, 3))  # Tamanho ajustado para 10x3
                plt.plot(S1_normalizado, label=ticker1)
                plt.plot(S2_normalizado, label=ticker2)
                plt.legend(loc='best')
                plt.xlabel('Data')
                plt.ylabel('Preço Normalizado')
                plt.xticks(rotation=45, fontsize=6)
                st.pyplot(plt)

                

             

        # Adicionando a divisão em colunas
        st.markdown(f"### Gráficos Candlestick dos Últimos 10 Períodos")

        col1, col2 = st.columns(2)

        # Adicionando o botão para gerar os gráficos de candlestick
        gerar_graficos = st.button("Gerar Gráfico de Candlestick")

        # Só gera os gráficos se o botão for clicado
        if gerar_graficos:
            
            # Função para verificar se os dados possuem as colunas necessárias para candlestick
            def verificar_dados_candlestick(df):
                return all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])

            # Gráfico de Candlestick para o ticker1
            with col1:
                st.markdown(f"#### Candlestick: {ticker1}")
                # Baixando dados com colunas necessárias para o candlestick
                S1_candles = yf.download(ticker1, period='10d', interval='1d')
                
                # Garantir que os dados possuem colunas de Candlestick
                if verificar_dados_candlestick(S1_candles):
                    fig, (ax, ax_volume) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 6))  # Criar dois eixos
                    mpf.plot(S1_candles, type='candle', style='charles', ax=ax, volume=ax_volume)  # Passar ambos os eixos
                    st.pyplot(fig)  # Exibir o gráfico no Streamlit
                else:
                    st.error(f"Erro ao baixar dados de candlestick para {ticker1}. Verifique se o ativo possui dados suficientes.")

            # Gráfico de Candlestick para o ticker2
            with col2:
                st.markdown(f"#### Candlestick: {ticker2}")
                # Baixando dados com colunas necessárias para o candlestick
                S2_candles = yf.download(ticker2, period='10d', interval='1d')

                # Garantir que os dados possuem colunas de Candlestick
                if verificar_dados_candlestick(S2_candles):
                    fig, (ax, ax_volume) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 6))  # Criar dois eixos
                    mpf.plot(S2_candles, type='candle', style='charles', ax=ax, volume=ax_volume)  # Passar ambos os eixos
                    st.pyplot(fig)  # Exibir o gráfico no Streamlit
                else:
                    st.error(f"Erro ao baixar dados de candlestick para {ticker2}. Verifique se o ativo possui dados suficientes.")

                        # Botão para encerrar a operação
            encerrar_button = st.button("Encerrar Operação")
            if encerrar_button:
                conn = get_connection()
                cursor = conn.cursor()
                # Atualizar o status da operação para 'fechada' e registrar a data de encerramento
                data_encerramento = datetime.now().strftime('%Y-%m-%d')
                cursor.execute('''
                UPDATE operacoes
                SET status = ?, data_encerramento = ?
                WHERE par = ?
                ''', ('fechada', data_encerramento, par_selecionado))
                conn.commit()
                conn.close()
                st.success(f"A operação para o par {par_selecionado} foi encerrada com sucesso!")