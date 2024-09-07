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

if selected == "Operações":
    st.title("Operações")

    # Opções de status (analise, aberta, fechada)
    status_opcoes = ["Analise", "Aberta", "Fechada"]
    status_selecionado = st.radio("Selecione o status da operação:", status_opcoes)

    # Mapeamento do valor selecionado para o status no banco de dados
    status_mapeado = status_selecionado.lower()

    # Consulta ao banco de dados para buscar pares com o status selecionado
    conn = get_connection()
    query = f"""
    SELECT par, zscore, pvalue, hurst, beta, half_life, data 
    FROM operacoes
    WHERE status = ?
    """
    operacoes_df = pd.read_sql(query, conn, params=[status_mapeado])
    conn.close()

    # Se houver pares disponíveis para o status selecionado
    if not operacoes_df.empty:
        # Criar um selectbox para escolher o par
        pares_disponiveis = operacoes_df['par'].tolist()
        par_selecionado = st.selectbox(f"Selecione um par com status '{status_selecionado}':", pares_disponiveis)

        # Buscar os detalhes do par selecionado
        par_detalhes = operacoes_df[operacoes_df['par'] == par_selecionado].iloc[0]

        # Dividir os tickers do par selecionado
        ticker1, ticker2 = par_selecionado.split(" - ")

        # Obter o preço atual de cada ação diretamente do Yahoo Finance
        preco_acao1 = obter_preco_atual(ticker1)
        preco_acao2 = obter_preco_atual(ticker2)

        # Primeira linha: gráficos do Z-Score e dos preços
        col1, col2 = st.columns(2)

        with col1:
            # Baixar os dados históricos das ações para calcular o Z-Score
            S1 = yf.download(ticker1, period="1y")['Close']
            S2 = yf.download(ticker2, period="1y")['Close']

            # Plotar o gráfico do Z-Score
            plotar_grafico_zscore(S1, S2)

        with col2:
            # Exibir as métricas detalhadas do par
            st.markdown(f"### Detalhes do Par: {par_selecionado}")
            st.markdown(
                f"""
                **Data:** {par_detalhes['data']}  
                **Z-Score:** {par_detalhes['zscore']:.2f}  
                **P-Value:** {par_detalhes['pvalue']:.4f}  
                **Hurst:** {par_detalhes['hurst']:.4f}  
                **Beta:** {par_detalhes['beta']:.4f}  
                **Half-Life:** {par_detalhes['half_life']:.2f}  
                """
            )

        # Segunda linha: gráficos de preços e cotações recentes
        col3, col4 = st.columns(2)

        with col3:
            # Plotar o gráfico dos preços das duas ações
            plotar_grafico_precos(S1, S2, ticker1, ticker2)

        with col4:
            # Exibir as cotações mais recentes de cada ação (sem cards, apenas markdown)
            st.markdown(f"**Preço Atual de {ticker1}:** R$ {preco_acao1:.2f}")
            st.markdown(f"**Preço Atual de {ticker2}:** R$ {preco_acao2:.2f}")

        st.markdown("---")

        # Aplicando a cor ao expander
        st.markdown(
            """
            <style>
            .streamlit-expanderHeader {
                background-color: #f0f8ff;  /* Azul claro */
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .streamlit-expanderContent {
                background-color: #e6f7ff;  /* Azul claro para o conteúdo */
            }
            </style>
            """, unsafe_allow_html=True
        )

        with st.expander("Calculadora de Operação", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### Escolher Lotes")

                # Input para definir a quantidade de lotes vendidos
                lotes_vendidos = st.number_input(f"Quantidade de Lotes Vendidos ({ticker1}):", min_value=100, step=100, value=100)
                valor_venda = preco_acao1 * lotes_vendidos

                # Input para definir a quantidade de lotes comprados
                lotes_comprados = st.number_input(f"Quantidade de Lotes Comprados ({ticker2}):", min_value=100, step=100, value=100)
                valor_compra = preco_acao2 * lotes_comprados

                # Calcula o saldo remanescente
                saldo_remanescente = valor_venda - valor_compra

            with col2:
                # Alterar o título da seção após clicar em "Cancelar Análise"
                st.markdown("### Resultados da Operação")
                st.markdown(f"""
                - **Valor Vendido ({ticker1}):** R$ {valor_venda:.2f}  
                - **Valor Comprado ({ticker2}):** R$ {valor_compra:.2f}  
                - **Saldo Remanescente:** R$ {saldo_remanescente:.2f}
                """)

                # Adiciona o botão "Cancelar Análise"
                if st.button("Cancelar Análise"):
                    st.markdown("### Lotes Escolhidos")

            with col3:
                # Formulário para iniciar operação
                with st.form("iniciar_operacao"):
                    st.markdown("### Iniciar Operação")
                    st.markdown(f"**Preço Atual de {ticker1}:** R$ {preco_acao1:.2f}")
                    st.markdown(f"**Preço Atual de {ticker2}:** R$ {preco_acao2:.2f}")
                    st.markdown(f"**Quantidade de Lotes Vendidos ({ticker1}):** {lotes_vendidos}")
                    st.markdown(f"**Quantidade de Lotes Comprados ({ticker2}):** {lotes_comprados}")
                    
                    # Botão para iniciar a operação
                    iniciar_button = st.form_submit_button("Iniciar Operação")

                    if iniciar_button:
                        st.success("Operação Iniciada com Sucesso!")
    else:
        st.write(f"Não há pares com status '{status_selecionado}' no momento.")
