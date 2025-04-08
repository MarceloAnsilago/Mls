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
from io import BytesIO
import base64                

# Inicializando o estado global para as cotações
if "global_cotacoes" not in st.session_state:
    st.session_state["global_cotacoes"] = pd.DataFrame()


st.set_page_config(page_title="Gerenciamento de Ações", page_icon=":chart_with_upwards_trend:", layout="wide")
logo_image = Image.open("logos/LogoApp.png")

# Função para carregar o ícone
def carregar_icone(ticker):
    # Verifica o formato do ticker e ajusta o nome do arquivo
    if len(ticker) >= 5 and ticker[4].isdigit():
        ticker_base = ticker[:4]
    else:
        ticker_base = ticker.replace(".SA", "")

    # Caminho do arquivo do ícone
    icon_path = f"logos/{ticker_base}.jpg"

    if os.path.exists(icon_path):
        try:
            with open(icon_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
                return f"data:image/jpg;base64,{encoded_string}"
        except Exception as e:
            return None
    else:
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
    try:
        returns_S1 = np.log(S1 / S1.shift(1)).dropna()
        returns_S2 = np.log(S2 / S2.shift(1)).dropna()

        betas = []
        index_values = returns_S1.index[window - 1:]  # Ajustar para a janela

        for i in range(window, len(returns_S1) + 1):
            reg = LinearRegression().fit(
                returns_S2[i - window:i].values.reshape(-1, 1),
                returns_S1[i - window:i].values
            )
            betas.append(reg.coef_[0])

        beta_movel = pd.Series(betas, index=index_values)

        # Plotar o gráfico de beta móvel
        plt.figure(figsize=(10, 5))
        plt.plot(beta_movel, label=f'Beta Móvel ({window} períodos)')
        plt.axhline(0, color='black', linestyle='--')
        plt.title(f'Beta Móvel ({window} períodos)')
        plt.xlabel('Data')
        plt.ylabel('Beta')
        plt.legend()
        plt.xticks(rotation=45, fontsize=6)
        plt.grid(True)

        # Reduzir a quantidade de rótulos no eixo X
        ax = plt.gca()             # Pega o axis atual
        ticks = ax.get_xticks()    # Pega os ticks atuais
        ax.set_xticks(ticks[::5])  # Exibe somente 1 a cada 5

        st.pyplot(plt)
    except Exception as e:
        st.error(f"Erro ao calcular ou plotar o beta móvel: {e}")


# Exibir o gráfico de dispersão entre os dois ativos
def plotar_grafico_dispersao(S1, S2):
    plt.figure(figsize=(10, 5))
    plt.scatter(S1, S2)
    plt.title(f'Dispersão entre {S1.name} e {S2.name}')
    plt.xlabel(f'{S1.name}')
    plt.ylabel(f'{S2.name}')
    plt.grid(True)
    st.pyplot(plt)


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


def exibir_metrica_cartao(ticker, ultimo_preco, ultima_data, icone=None):
    icone_html = (
        f'<img src="{icone}" style="max-width: 100px; max-height: 100px; object-fit: contain;">'
        if icone else '<p style="color: red;">Sem Ícone</p>'
    )

    st.markdown(
        f"""
        <div style="border: 1px solid #ddd; border-radius: 10px; padding: 10px; text-align: center; background-color: #f9f9f9; height: 300px; margin-bottom: 15px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <h2 style="margin: 0;">{ticker}</h2> <!-- Aumentei a fonte do título -->
                <hr style="border: none; border-top: 2px solid red; margin: 5px 0 10px 0;">
            </div>
            <div style="flex-grow: 1; display: flex; justify-content: center; align-items: center;">
                {icone_html} <!-- Ícone com altura ajustada -->
            </div>
            <div style="margin-top: 10px; text-align: center;">
                <h6 style="font-size: 14px; color: #888; margin-bottom: 4px;">Última Cotação ({ultima_data})</h6>
                <h3 style="margin: 0; font-size: 24px;">R$ {ultimo_preco:.2f}</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# Menu Lateral
with st.sidebar:
    st.image(logo_image, use_column_width=True)  # Exibir a imagem no menu lateral
    selected = option_menu(
        menu_title="Menu Principal",  # required
        options=["Página Inicial", "Cotações", "Análise", "Operações", "Backtesting"],  # required
        icons=["house", "currency-exchange", "graph-up-arrow", "briefcase", "clock-history"],  # ícones para cada página
        menu_icon="cast",  # ícone do menu
        default_index=0,  # seleciona a aba 'Página Inicial'
    )

# Aba "Ações Acompanhadas"
if selected == "Página Inicial":
    # st.title("Ações Acompanhadas")
    
    def get_base64(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()

    image_base64 = get_base64("logos/Ações_Acompanhadas.png")

    st.markdown(
        f"""
        <div style="display: flex; align-items: center;">
            <img src="data:image/png;base64,{image_base64}" alt="Logo" style="height: 130px; margin-right: 10px;">
            <h1 style="margin: 0;">Ações Acompanhadas</h1>
        </div>
        """,
        unsafe_allow_html=True
    )


    # Verificar se o DataFrame global tem dados
    if "global_cotacoes" in st.session_state and not st.session_state["global_cotacoes"].empty:
        # DataFrame global com as cotações
        cotacoes_df = st.session_state["global_cotacoes"].copy()

        # Verificar se o índice é chamado 'Date' e transformá-lo em coluna
        if "Date" not in cotacoes_df.columns:
            cotacoes_df.reset_index(inplace=True)

        # Garantir que a coluna "Date" exista
        if "Date" in cotacoes_df.columns:
            # Preparar os dados para exibição (última cotação de cada ticker)
            cotacoes_df = cotacoes_df.melt(id_vars=["Date"], var_name="ticker", value_name="fechamento")
            cotacoes_df = cotacoes_df.dropna().sort_values(by=["ticker", "Date"], ascending=[True, False])
            cotacoes_df = cotacoes_df.groupby("ticker").first().reset_index()

            # Exibir as métricas em 5 colunas com espaçamento de 5px
            cols = st.columns(5, gap="small")

            for index, row in cotacoes_df.iterrows():
                ticker = row['ticker']
                ultimo_preco = row['fechamento']
                ultima_data = row['Date']

                # **Carregar ícone do ticker**
                icone = carregar_icone(ticker)

                # Exibir a métrica no formato de cartão
                with cols[index % 5]:
                    exibir_metrica_cartao(ticker, ultimo_preco, ultima_data, icone)
        else:
            st.error("A coluna de datas ('Date') não foi encontrada no DataFrame.")
    else:
        st.warning("Nenhuma cotação carregada. Por favor, carregue as cotações na aba 'Cotações'.")





# Aba de Cotações
if selected == "Cotações":
    def get_base64(file_path):
     with open(os.path.abspath(file_path), "rb") as f:
        return base64.b64encode(f.read()).decode()

    image_base64 = get_base64("logos/cotacoes.png") 
    
    st.markdown(
        f"""
        <div style="display: flex; align-items: center;">
            <img src="data:image/png;base64,{image_base64}" alt="Cotação" style="height: 130px; margin-right: 10px;">
            <h1 style="margin: 0;">Cotações de Ações</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Upload do arquivo TXT com tickers
    st.markdown("### Upload do Arquivo com Tickers")
    uploaded_file = st.file_uploader(
        "Envie um arquivo TXT com os tickers das ações (um por linha):", 
        type="txt", 
        key="file_uploader_cotacoes"
    )

    # Entrada do número de períodos para buscar cotações
    num_periodos = st.number_input(
        "Número de Períodos (em dias) para Buscar Cotações",
        min_value=1,
        max_value=365,
        value=200,
        step=1,
        key="number_input_periodos"
    )

    # Botão para processar e exibir as cotações
    if st.button("Carregar Cotações", key="botao_carregar_cotacoes"):
        if uploaded_file:
            # Ler os tickers do arquivo TXT
            tickers = uploaded_file.read().decode('utf-8').splitlines()
            tickers = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
            
            if tickers:
                # Criar um DataFrame temporário para armazenar novas cotações
                new_cotacoes = pd.DataFrame()

                progress_bar = st.progress(0)  # Barra de progresso
                status_text = st.empty()  # Espaço para texto dinâmico

                for idx, ticker in enumerate(tickers):
                    try:
                        status_text.text(f"Baixando cotações para {ticker}...")
                        dados = yf.download(ticker, period=f"{num_periodos}d")['Close']
                        dados.name = ticker  # Renomear a série com o ticker
                        new_cotacoes = pd.concat([new_cotacoes, dados], axis=1)
                    except Exception as e:
                        st.error(f"Erro ao buscar cotações para {ticker}: {e}")

                    # Atualizar barra de progresso
                    progress_bar.progress((idx + 1) / len(tickers))
                
                # Remover status e barra após conclusão
                status_text.empty()
                progress_bar.empty()

                # Atualizar o DataFrame global no session_state
                if not new_cotacoes.empty:
                    # Formatar o índice de datas antes de atualizar o DataFrame global
                    if isinstance(new_cotacoes.index, pd.DatetimeIndex):
                        new_cotacoes.index = new_cotacoes.index.strftime("%Y-%m-%d")
                    new_cotacoes.index.name = "Date"  # Nomear o índice como 'Date'

                    # Concatenar os dados ao DataFrame global, removendo duplicatas
                    st.session_state["global_cotacoes"] = pd.concat(
                        [st.session_state["global_cotacoes"], new_cotacoes], axis=1
                    ).loc[:, ~pd.concat(
                        [st.session_state["global_cotacoes"], new_cotacoes], axis=1
                    ).columns.duplicated()]
                    st.success("Cotações carregadas com sucesso!")
            else:
                st.error("O arquivo está vazio ou não contém tickers válidos.")
        else:
            st.error("Por favor, envie um arquivo TXT com os tickers das ações.")

    # Exibir o DataFrame global no Streamlit, se não estiver vazio
    if not st.session_state["global_cotacoes"].empty:
        df_para_exibir = st.session_state["global_cotacoes"].reset_index()
        st.markdown("### Cotações Carregadas")
        st.dataframe(df_para_exibir)
    else:
        st.warning("Nenhuma cotação foi carregada ainda.")

def mostrar_fluxo_liquido(venda_total: float, compra_total: float) -> float:
    """
    Exibe o fluxo financeiro líquido da operação de long & short.

    Retorna o valor do fluxo líquido para uso posterior.
    """
    resultado_total = venda_total - compra_total

    mensagem = f"**Fluxo Líquido da Operação: R$ {resultado_total:.2f}**"
    explicacao = (
        "Este valor representa o fluxo financeiro inicial da operação de long & short. "
        "Se for positivo, você recebe esse montante ao montar a estratégia. "
        "Se for negativo, você precisa investir esse valor para abrir as posições."
    )

    if resultado_total >= 0:
        st.success(mensagem)
    else:
        st.error(mensagem)

    st.markdown(f"<p style='font-size: 14px; color: #666;'>{explicacao}</p>", unsafe_allow_html=True)

    return resultado_total



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
            if "global_cotacoes" not in st.session_state or st.session_state["global_cotacoes"].empty:
                st.error("Nenhuma cotação carregada. Por favor, carregue as cotações antes de realizar a análise.")
                st.stop()

            cotacoes_df = st.session_state["global_cotacoes"]
            if "Date" in cotacoes_df.columns:
                cotacoes_df.set_index("Date", inplace=True)

            cotacoes_pivot = cotacoes_df.tail(numero_periodos)
            st.session_state['cotacoes_pivot'] = cotacoes_pivot

        cotacoes_pivot = st.session_state['cotacoes_pivot']
        st.write(f"Número de períodos selecionados para análise: {cotacoes_pivot.shape[0]}")

        st.subheader("Pares Encontrados")

        # Encontrar pares cointegrados e calcular métricas
        # Spinner enquanto os pares são analisados
        with st.spinner("🔍 Analisando cointegração entre os ativos..."):
            pairs, pvalues, zscores, half_lives, hursts, beta_rotations = find_cointegrated_pairs(
                cotacoes_pivot, zscore_threshold_upper, zscore_threshold_lower
            )

        if pairs:
            for idx, (pair, zscore, pvalue, hurst, beta, half_life) in enumerate(zip(pairs, zscores, pvalues, hursts, beta_rotations, half_lives)):
                par_str = f"{pair[0]} - {pair[1]}"
                metricas_str = f"Z-Score: {zscore:.2f} | P-Value: {pvalue:.4f} | Hurst: {hurst:.4f} | Beta: {beta:.4f} | Half-Life: {half_life:.2f}"
                
                if st.button(f"{par_str} | {metricas_str}", key=f"btn_{idx}"):
                    st.session_state['par_selecionado'] = pair

            if 'par_selecionado' in st.session_state:
                pair_selected = st.session_state['par_selecionado']
                S1 = cotacoes_pivot[pair_selected[0]]
                S2 = cotacoes_pivot[pair_selected[1]]
                ratios = S1 / S2
                zscore_series = (ratios - ratios.mean()) / ratios.std()

                st.markdown("---")
                st.markdown(f"<h4 style='text-align: center;'>{pair_selected[0]} - {pair_selected[1]}</h4>", unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Z-Score do Par")
                    # Cria a figura e o eixo
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(zscore_series, label='Z-Score')
                    ticks = ax.get_xticks()  
                    ax.set_xticks(ticks[::5])
                    ax.axhline(0, color='black', linestyle='--')
                    ax.axhline(2, color='red', linestyle='--')
                    ax.axhline(-2, color='green', linestyle='--')
                    ax.axhline(3, color='orange', linestyle='--', label='+3 Desvio (Stop)')
                    ax.axhline(-3, color='orange', linestyle='--', label='-3 Desvio (Stop)')
                    ax.legend(loc='best')
                    ax.set_xlabel('Data')
                    ax.set_ylabel('Z-Score')

                    # Rotaciona e diminui o tamanho das labels do eixo X
                    plt.xticks(rotation=45, fontsize=6)

                    ax.grid(True)

                    # Ajusta automaticamente o layout para evitar sobreposições
                    fig.tight_layout()

                    # Exibe no Streamlit
                    st.pyplot(fig)

                with col2:
                    st.subheader("Cotação Normalizada")
                    fig, ax = plt.subplots(figsize=(10, 5))

                    # Plota os dois ativos normalizados
                    ax.plot(S1 / S1.iloc[0], label=f"{pair_selected[0]}")
                    ax.plot(S2 / S2.iloc[0], label=f"{pair_selected[1]}")

                    # Ajusta legendas e eixos
                    ax.legend(loc='best')
                    ax.set_xlabel('Data')
                    ax.set_ylabel('Cotação Normalizada')
                    ax.grid(True)

                    # Rotaciona e diminui a fonte das datas
                    plt.xticks(rotation=45, fontsize=6)

                    # Reduz a quantidade de rótulos no eixo X
                    ticks = ax.get_xticks()      # Pega os ticks atuais
                    ax.set_xticks(ticks[::5])    # Exibe somente 1 a cada 5

                    fig.tight_layout()
                    st.pyplot(fig)

                col3, col4 = st.columns(2)

                with col3:
                    st.subheader(f"Beta Móvel para {pair_selected[0]} e {pair_selected[1]}")
                    plotar_beta_movel(S1, S2, window=40)

                with col4:
                    st.subheader(f"Dispersão entre {pair_selected[0]} e {pair_selected[1]}")
                    plotar_grafico_dispersao(S1, S2)


               # === Tabs adicionais antes do Expander ===
               # Determina os ativos antes de usar nas abas
                current_zscore = zscores[pairs.index(pair_selected)]
                if current_zscore > 0:
                    stock_to_sell = pair_selected[0]
                    stock_to_buy = pair_selected[1]
                else:
                    stock_to_sell = pair_selected[1]
                    stock_to_buy = pair_selected[0]




                tab1, tab2 = st.tabs(["📐 Calcular Proporção", "📎 Outra Ação"])
              
                with tab1:
                        with st.expander("📐 Cálculo da Proporção entre os Ativos", expanded=False):
                            col1, col2 = st.columns(2)

                            preco_venda = S1.iloc[-1]
                            preco_compra = S2.iloc[-1]

                            ativo_venda = stock_to_sell
                            ativo_compra = stock_to_buy

                            with col1:
                                st.markdown(f"### 🔻 Vender (Short): `{ativo_venda}`")
                                st.write(f"Preço atual de **{ativo_venda}**: R$ {preco_venda:.2f}")
                                capital_maximo = st.number_input(
                                    "Capital Total para Venda (R$)", 
                                    min_value=100.0, 
                                    value=25000.0, 
                                    step=100.0
                                )

                            with col2:
                                st.markdown(f"### 🔺 Comprar (Long): `{ativo_compra}`")
                                st.write(f"Preço atual de **{ativo_compra}**: R$ {preco_compra:.2f}")

                            st.markdown("---")
                            st.subheader("📊 Melhor Proporção com Base no Limite de Venda")

                            melhor_resultado = None

                            # Calcula o máximo de lotes de venda dentro do capital informado
                            max_lotes_venda = int(capital_maximo // (100 * preco_venda))

                            for lotes_venda in range(1, max_lotes_venda + 1):
                                total_venda = lotes_venda * 100 * preco_venda

                                for lotes_compra in range(1, 100):
                                    total_compra = lotes_compra * 100 * preco_compra
                                    residuo = abs(total_venda - total_compra)

                                    if melhor_resultado is None or residuo < melhor_resultado["residuo"]:
                                        melhor_resultado = {
                                            "lotes_venda": lotes_venda,
                                            "lotes_compra": lotes_compra,
                                            "total_venda": total_venda,
                                            "total_compra": total_compra,
                                            "residuo": residuo,
                                            "fluxo_liquido": total_venda - total_compra
                                        }

                            if melhor_resultado:
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.markdown(f"### 🔻 Vender (Short) {ativo_venda}")
                                    st.write(f"- Lotes de 100: **{melhor_resultado['lotes_venda']}**")
                                    st.write(f"- Quantidade: **{melhor_resultado['lotes_venda'] * 100} ações**")
                                    st.write(f"- Total Venda: R$ {melhor_resultado['total_venda']:.2f}")

                                with col2:
                                    st.markdown(f"### 🔺 Comprar (Long): `{ativo_compra}`")
                                    st.write(f"- Lotes de 100: **{melhor_resultado['lotes_compra']}**")
                                    st.write(f"- Quantidade: **{melhor_resultado['lotes_compra'] * 100} ações**")
                                    st.write(f"- Total Compra: R$ {melhor_resultado['total_compra']:.2f}")

                                st.markdown("---")
                                fluxo = melhor_resultado['fluxo_liquido']
                                if fluxo >= 0:
                                    st.success(f"💰 Fluxo Inicial da Operação: R$ {fluxo:.2f}")
                                else:
                                    st.error(f"📉 Fluxo Inicial da Operação: R$ {fluxo:.2f}")

                                st.markdown(f"📎 Resíduo Absoluto entre os valores: R$ {melhor_resultado['residuo']:.2f}")
                            else:
                                st.warning("❗ Nenhuma combinação de lotes encontrada.")




                with tab2:
                            st.subheader("📎 Correlação inversa")
                            st.info("Conteúdo alternativo aqui se desejar adicionar algo.")





                with st.expander("Configurar Operação", expanded=True):
                    current_zscore = zscores[pairs.index(pair_selected)]

                    if current_zscore > 0:
                        st.markdown(
                            f"**Legenda de Operação:** Com o Z-Score positivo ({current_zscore:.2f}), recomenda-se **VENDER {pair_selected[0]}** (ativo sobrevalorizado) e **COMPRAR {pair_selected[1]}** (ativo subvalorizado)."
                        )
                        stock_to_sell = pair_selected[0]
                        stock_to_buy = pair_selected[1]
                        sell_price = S1.iloc[-1]
                        buy_price = S2.iloc[-1]
                    else:
                        st.markdown(
                            f"**Legenda de Operação:** Com o Z-Score negativo ({current_zscore:.2f}), recomenda-se **VENDER {pair_selected[1]}** (ativo sobrevalorizado) e **COMPRAR {pair_selected[0]}** (ativo subvalorizado)."
                        )
                        stock_to_sell = pair_selected[1]
                        stock_to_buy = pair_selected[0]
                        sell_price = S2.iloc[-1]
                        buy_price = S1.iloc[-1]
                  
                    col1, col2 = st.columns(2)

                    # =========================
                    # Coluna 1 - Ação Vendida
                    # =========================
                    with col1:
                        st.subheader(f"Vender Ação: {stock_to_sell}")
                        venda_quantidade = st.number_input("Quantidade para Vender", min_value=100, step=100, value=100, key="venda_quantidade")
                        venda_preco_atual = sell_price
                        venda_total = venda_quantidade * venda_preco_atual

                        st.write(f"Preço Atual: R$ {venda_preco_atual:.2f}")
                        st.write(f"Total Venda: R$ {venda_total:.2f}")
                        
                        slider_venda = st.slider("Movimento (%)", min_value=0, max_value=25, value=5, step=1, key="slider_venda")
                        st.write(f"Simulação de {slider_venda}%")

                        novo_preco_caindo = venda_preco_atual * (1 - slider_venda / 100)
                        lucro_short = (venda_preco_atual - novo_preco_caindo) * venda_quantidade

                        novo_preco_subindo = venda_preco_atual * (1 + slider_venda / 100)
                        preju_short = (venda_preco_atual - novo_preco_subindo) * venda_quantidade

                        st.metric(f"Queda de {slider_venda}% (Lucro Short)", f"R$ {novo_preco_caindo:.2f}", delta=round(lucro_short, 2))
                        st.metric(f"Alta de {slider_venda}% (Prejuízo Short)", f"R$ {novo_preco_subindo:.2f}", delta=round(preju_short, 2))

                    # =========================
                    # Coluna 2 - Ação Comprada
                    # =========================
                    with col2:
                        st.subheader(f"Comprar Ação: {stock_to_buy}")
                        compra_quantidade = st.number_input("Quantidade para Comprar", min_value=100, step=100, value=100, key="compra_quantidade")
                        compra_preco_atual = buy_price
                        compra_total = compra_quantidade * compra_preco_atual

                        st.write(f"Preço Atual: R$ {compra_preco_atual:.2f}")
                        st.write(f"Total Compra: R$ {compra_total:.2f}")

                        slider_compra = st.slider("Movimento (%)", min_value=0, max_value=25, value=5, step=1, key="slider_compra")
                        st.write(f"Simulação de {slider_compra}%")

                        novo_preco_subindo_long = compra_preco_atual * (1 + slider_compra / 100)
                        lucro_long = (novo_preco_subindo_long - compra_preco_atual) * compra_quantidade

                        novo_preco_caindo_long = compra_preco_atual * (1 - slider_compra / 100)
                        preju_long = (novo_preco_caindo_long - compra_preco_atual) * compra_quantidade

                        st.metric(f"Alta de {slider_compra}% (Lucro Long)", f"R$ {novo_preco_subindo_long:.2f}", delta=round(lucro_long, 2))
                        st.metric(f"Queda de {slider_compra}% (Prejuízo Long)", f"R$ {novo_preco_caindo_long:.2f}", delta=round(preju_long, 2))
                    resultado_total = mostrar_fluxo_liquido(venda_total, compra_total)
                    st.markdown("---")
                    preju_total = preju_short + preju_long
                    st.metric("STOP (Soma dos Prejuízos)", f"R$ {preju_total:.2f}", delta=round(preju_total, 2))

         

 
                st.markdown("---")
                if st.button("Salvar Operação como Excel"):
                    operacao_data = {
                        "Ativo Vendido": [stock_to_sell],
                        "Ativo Comprado": [stock_to_buy],
                        "Preço Venda": [sell_price],
                        "Preço Compra": [buy_price],
                        "Quantidade Vendida": [venda_quantidade],
                        "Quantidade Comprada": [compra_quantidade],
                        "Resultado Total": [resultado_total],
                        "Data Operação": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                    }

                    df_operacao = pd.DataFrame(operacao_data)
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df_operacao.to_excel(writer, index=False, sheet_name="Operacao")
                    output.seek(0)

                    st.download_button("Baixar Operação em Excel", data=output, file_name=f"operacao_{pair_selected[0]}_{pair_selected[1]}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                # Exibir métricas no rodapé
                st.markdown("---")
                st.markdown("### 📊 Métricas do Par Selecionado e Recomendações")

                col1, col2, col3, col4, col5 = st.columns(5)

                # Z-Score
                col1.metric("Z-Score", f"{zscores[pairs.index(pair_selected)]:.2f}")
                col1.caption("📌 Mede o desvio da média. Valores acima de ±2 indicam oportunidade de reversão.")

                # P-Value
                col2.metric("P-Value", f"{pvalues[pairs.index(pair_selected)]:.4f}")
                col2.caption("📌 Probabilidade de cointegração ser aleatória. Valores abaixo de 0.05 são desejáveis.")

                # Hurst Exponent
                col3.metric("Hurst", f"{hursts[pairs.index(pair_selected)]:.4f}")
                col3.caption("📌 Mede a tendência de reversão ou persistência. Valores próximos de 0.5 são ideais.")

                # Beta Rotation
                col4.metric("Beta", f"{beta_rotations[pairs.index(pair_selected)]:.4f}")
                col4.caption("📌 Sensibilidade do ativo em relação ao outro. Quanto menor, melhor para pares estáveis.")

                # Half-Life
                col5.metric("Half-Life", f"{half_lives[pairs.index(pair_selected)]:.2f}")
                col5.caption("📌 Tempo esperado para metade da reversão ao valor médio. Quanto menor, melhor.")




if selected == "Operações":
    st.title("Operações")

    # Upload do arquivo Excel de operações
    uploaded_file = st.file_uploader("Carregar Arquivo de Operação (Excel)", type=["xlsx"])

    if uploaded_file is not None:
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.linear_model import LinearRegression
        import numpy as np

        # Função para plotar Beta Móvel
        def plotar_beta_movel(S1, S2, window=40):
            try:
                returns_S1 = np.log(S1 / S1.shift(1)).dropna()
                returns_S2 = np.log(S2 / S2.shift(1)).dropna()

                betas = []
                index_values = returns_S1.index[window - 1:]  # Ajustar para a janela

                for i in range(window, len(returns_S1) + 1):
                    reg = LinearRegression().fit(
                        returns_S2[i - window:i].values.reshape(-1, 1),
                        returns_S1[i - window:i].values
                    )
                    betas.append(reg.coef_[0])

                beta_movel = pd.Series(betas, index=index_values)

                # Plotar o gráfico de beta móvel
                plt.figure(figsize=(10, 5))
                plt.plot(beta_movel, label=f'Beta Móvel ({window} períodos)')
                plt.axhline(0, color='black', linestyle='--')
                plt.title(f'Beta Móvel ({window} períodos)')
                plt.xlabel('Data')
                plt.ylabel('Beta')
                plt.legend()
                plt.grid(True)
                st.pyplot(plt)
            except Exception as e:
                st.error(f"Erro ao calcular ou plotar o beta móvel: {e}")

        try:
            # Ler o arquivo Excel
            df_operacoes = pd.read_excel(uploaded_file)

            # 1) Converter a coluna de datas para datetime
            df_operacoes["Data Operação"] = pd.to_datetime(df_operacoes["Data Operação"], errors="coerce")

            # Avisar caso tenha datas inválidas
            if df_operacoes["Data Operação"].isnull().any():
                st.warning("Há datas inválidas em 'Data Operação'. Verifique o arquivo Excel.")

            # Verificar se "global_cotacoes" está carregado
            if "global_cotacoes" not in st.session_state or st.session_state["global_cotacoes"].empty:
                st.error("Nenhuma cotação carregada no sistema. Por favor, carregue as cotações na aba 'Cotações'.")
                st.stop()

            # 2) Converter o índice do DataFrame de cotações para DateTimeIndex
            cotacoes_df = st.session_state["global_cotacoes"]
            if not pd.api.types.is_datetime64_any_dtype(cotacoes_df.index):
                # Tenta converter o índice atual para datetime
                cotacoes_df.index = pd.to_datetime(cotacoes_df.index, errors='coerce')
                # Remover possíveis linhas com índice inválido (NaT)
                cotacoes_df = cotacoes_df[~cotacoes_df.index.isnull()]
                # Sobrescreve no session_state, se desejar reutilizar o DF corrigido
                st.session_state["global_cotacoes"] = cotacoes_df

            # Preparar a tabela final
            tabela_final = []

            for index, row in df_operacoes.iterrows():
                # Verifica se a data é válida (não é NaT)
                if pd.isnull(row["Data Operação"]):
                    st.error(f"Linha {index}: Data da Operação inválida. Operação ignorada.")
                    continue

                ativo_venda = row["Ativo Vendido"]
                ativo_compra = row["Ativo Comprado"]
                data_operacao = row["Data Operação"]
                quantidade_venda = row["Quantidade Vendida"]
                quantidade_compra = row["Quantidade Comprada"]
                valor_inicial_venda = row["Preço Venda"]
                valor_inicial_compra = row["Preço Compra"]
                numero_periodos = 120  # Número de períodos padrão

                # Verificar se os ativos estão disponíveis nas cotações
                if ativo_venda not in cotacoes_df.columns or ativo_compra not in cotacoes_df.columns:
                    st.error(f"Os ativos {ativo_venda} ou {ativo_compra} não estão disponíveis nas cotações globais.")
                    continue

                # Obter as séries históricas dos ativos, limitadas a 120 períodos
                series_venda = cotacoes_df[ativo_venda].tail(numero_periodos)
                series_compra = cotacoes_df[ativo_compra].tail(numero_periodos)

                # Calcular o Z-Score
                ratios = series_venda / series_compra
                zscore_series = (ratios - ratios.mean()) / ratios.std()

                # Calcular valor total e saldo
                valor_total_venda = quantidade_venda * series_venda.iloc[-1]
                valor_total_compra = quantidade_compra * series_compra.iloc[-1]
                saldo = valor_total_venda - valor_total_compra

                # Adicionar dados à tabela final
                tabela_final.append({
                    "Data Operação": data_operacao,
                    "Ativo": ativo_venda,
                    "Quantidade": quantidade_venda,
                    "Tipo": "Venda",
                    "Valor Inicial": f"R$ {valor_inicial_venda:.2f}",
                    "Valor Total": f"R$ {valor_total_venda:.2f}",
                    "Saldo": f"R$ {saldo:.2f}"
                })

                tabela_final.append({
                    "Data Operação": data_operacao,
                    "Ativo": ativo_compra,
                    "Quantidade": quantidade_compra,
                    "Tipo": "Compra",
                    "Valor Inicial": f"R$ {valor_inicial_compra:.2f}",
                    "Valor Total": f"R$ {valor_total_compra:.2f}",
                    "Saldo": f"R$ {saldo:.2f}"
                })

                # Exibir informações e gráficos
                st.markdown("---")  # Separador
                st.write(f"**Ativo Vendido:** {ativo_venda}")
                st.write(f"**Ativo Comprado:** {ativo_compra}")
                st.write(f"**Data da Operação:** {data_operacao}")
                st.write(f"**Média das Razões:** {ratios.mean():.4f}")
                st.write(f"**Desvio Padrão das Razões:** {ratios.std():.4f}")

                # ======= Gráfico do Z-Score =======
                plt.figure(figsize=(12, 3))
                plt.plot(zscore_series, label="Z-Score")
                plt.axhline(0, color='black', linestyle='--')
                plt.axhline(2, color='red', linestyle='--', label="+2 Desvio")
                plt.axhline(-2, color='green', linestyle='--', label="-2 Desvio")
                plt.axhline(3, color='orange', linestyle='--', label="+3 Desvio (Stop)")
                plt.axhline(-3, color='orange', linestyle='--', label="-3 Desvio (Stop)")
                plt.legend(loc='best')
                plt.xlabel("Períodos")
                plt.ylabel("Z-Score")
                plt.title(f"Z-Score: {ativo_venda} vs {ativo_compra}")
                plt.grid(True)
                st.pyplot(plt)

                # ======= Gráfico de Paridade Normalizada =======
                plt.figure(figsize=(12, 3))
                plt.plot(series_venda / series_venda.iloc[0], label=f"{ativo_venda}")
                plt.plot(series_compra / series_compra.iloc[0], label=f"{ativo_compra}")
                plt.legend(loc='best')
                plt.xlabel("Períodos")
                plt.ylabel("Cotação Normalizada")
                plt.title(f"Paridade Normalizada: {ativo_venda} vs {ativo_compra}")
                plt.grid(True)
                st.pyplot(plt)

                # ======= Gráficos de Beta Móvel e Dispersão =======
                col3, col4 = st.columns(2)

                with col3:
                    plotar_beta_movel(series_venda, series_compra, window=40)

                with col4:
                    plt.figure(figsize=(10, 5))
                    plt.scatter(series_venda, series_compra, alpha=0.7)
                    plt.xlabel(f"{ativo_venda}")
                    plt.ylabel(f"{ativo_compra}")
                    plt.title(f"Dispersão: {ativo_venda} vs {ativo_compra}")
                    plt.grid(True)
                    st.pyplot(plt)

                # ======= Novos Gráficos de Desempenho (em duas colunas) =======
                # ======= Novos Gráficos de Desempenho (em duas colunas) =======
                data_inicial = data_operacao - pd.Timedelta(days=5)

                # Pegar série completa sem limitar a 120 períodos
                serie_completa_vendida = cotacoes_df[ativo_venda]
                serie_completa_comprada = cotacoes_df[ativo_compra]

                # Filtrar a partir de data_inicial até a última cotação
                desempenho_vendido = serie_completa_vendida.loc[data_inicial:]
                desempenho_comprado = serie_completa_comprada.loc[data_inicial:]

                col5, col6 = st.columns(2)

                # ======= Novos Gráficos de Desempenho (em duas colunas) =======
                data_inicial = data_operacao - pd.Timedelta(days=5)

                # Pegar série completa sem limitar a 120 períodos
                serie_completa_vendida = cotacoes_df[ativo_venda]
                serie_completa_comprada = cotacoes_df[ativo_compra]

                # Filtrar a partir de data_inicial até a última cotação
                desempenho_vendido = serie_completa_vendida.loc[data_inicial:]
                desempenho_comprado = serie_completa_comprada.loc[data_inicial:]

                col5, col6 = st.columns(2)

                # Gráfico 1 (Ativo Vendido)
                with col5:
                    plt.figure(figsize=(10, 4))
                    plt.plot(desempenho_vendido.index, desempenho_vendido.values, color='blue')
                    plt.title(
                        f"Desempenho do Ativo Vendido ({ativo_venda})\n"
                        f"(5 dias antes do início da operação até a última cotação) – Par: {ativo_compra}"
                    )
                    plt.xlabel("Data")
                    plt.ylabel("Preço")
                    plt.grid(False)  # remove a grade de fundo
                    st.pyplot(plt)

                # Gráfico 2 (Ativo Comprado)
                with col6:
                    plt.figure(figsize=(10, 4))
                    plt.plot(desempenho_comprado.index, desempenho_comprado.values, color='red')
                    plt.title(
                        f"Desempenho do Ativo Comprado ({ativo_compra})\n"
                        f"(5 dias antes do início da operação até a última cotação) – Par: {ativo_venda}"
                    )
                    plt.xlabel("Data")
                    plt.ylabel("Preço")
                    plt.grid(False)  # remove a grade de fundo
                    st.pyplot(plt)



            # Exibir tabela consolidada
            st.markdown("---")
            st.markdown("### Tabela Consolidada de Operações")
            tabela_df = pd.DataFrame(tabela_final)
            st.dataframe(tabela_df)

            # Separador com Posição Atual
            st.markdown("---")
            st.markdown("### Posição Atual")

            # DataFrame consolidado para posição atual
            posicao_atual = []

            for index, row in df_operacoes.iterrows():
                ativo_venda = row["Ativo Vendido"]
                ativo_compra = row["Ativo Comprado"]
                quantidade_venda = row["Quantidade Vendida"]
                quantidade_compra = row["Quantidade Comprada"]
                preco_inicial_venda = row["Preço Venda"]
                preco_inicial_compra = row["Preço Compra"]

                # Obter preços atuais dos ativos
                preco_atual_venda = cotacoes_df[ativo_venda].iloc[-1] if ativo_venda in cotacoes_df.columns else None
                preco_atual_compra = cotacoes_df[ativo_compra].iloc[-1] if ativo_compra in cotacoes_df.columns else None

                if preco_atual_venda is not None:
                    lucro_venda = (preco_inicial_venda - preco_atual_venda) * quantidade_venda
                    posicao_atual.append({
                        "Ativo": ativo_venda,
                        "Tipo": "Venda",
                        "Quantidade": quantidade_venda,
                        "Preço Inicial": f"R$ {preco_inicial_venda:.2f}",
                        "Preço Atual": f"R$ {preco_atual_venda:.2f}",
                        "Lucro/Prejuízo": f"R$ {lucro_venda:.2f}"
                    })

                if preco_atual_compra is not None:
                    lucro_compra = (preco_atual_compra - preco_inicial_compra) * quantidade_compra
                    posicao_atual.append({
                        "Ativo": ativo_compra,
                        "Tipo": "Compra",
                        "Quantidade": quantidade_compra,
                        "Preço Inicial": f"R$ {preco_inicial_compra:.2f}",
                        "Preço Atual": f"R$ {preco_atual_compra:.2f}",
                        "Lucro/Prejuízo": f"R$ {lucro_compra:.2f}"
                    })

            # Exibir a tabela de posição atual
            if posicao_atual:
                posicao_atual_df = pd.DataFrame(posicao_atual)
                st.dataframe(posicao_atual_df)
            else:
                st.warning("Nenhum dado disponível para a posição atual.")

            # Calcular o saldo final
            saldo_final = sum([
                (float(row["Lucro/Prejuízo"].replace('R$', '').replace(',', '').strip()))
                for row in posicao_atual
            ])

            # Exibir o saldo final
            st.markdown("### Saldo Final Consolidado")
            st.markdown(f"<h3 style='text-align: center; color: blue;'>R$ {saldo_final:.2f}</h3>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")