import streamlit as st
import pandas as pd
import plotly.express as px
from functools import lru_cache
import joblib
import os



# Constantes
MODEL_PATH = '../output/'
MODEL_FILE = 'RandomForestClassifier_model.pkl'
SCALER_FILE = 'scaler.pkl'
COLUMNS_FILE = 'columns.pkl'

## funções de pré-processamento

# Colunas datetime para processamento
DATETIME_COLUMNS = [
    'order_purchase_timestamp', 
    'order_approved_at', 
    'order_delivered_carrier_date', 
    'order_delivered_customer_date', 
    'order_estimated_delivery_date', 
    'shipping_limit_date'
]

@lru_cache(maxsize=1)
def load_model_files():
    """Carrega e cacheia os arquivos do modelo"""
    try:
        if not os.path.exists(MODEL_PATH + MODEL_FILE):
            raise FileNotFoundError(f"Modelo não encontrado em: {MODEL_PATH + MODEL_FILE}")
        
        model = joblib.load(MODEL_PATH + MODEL_FILE)
        scaler = joblib.load(MODEL_PATH + SCALER_FILE)
        columns = joblib.load(MODEL_PATH + COLUMNS_FILE)
        
        return model, scaler, columns
    except Exception as e:
        raise Exception(f"Erro ao carregar arquivos do modelo: {str(e)}")

def process_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Processa features baseadas em datetime"""
    for col in DATETIME_COLUMNS:
        df[col] = pd.to_datetime(df[col])
    
    base_date = df['order_purchase_timestamp']
    
    # Calcular deltas de uma vez
    deltas = {
        'delta_approved': 'order_approved_at',
        'delta_estimated_delivery': 'order_estimated_delivery_date',
        'delta_shipping_limit': 'shipping_limit_date',
        'delta_delivered_customer': 'order_delivered_customer_date',
        'delta_delivered_carrier': 'order_delivered_carrier_date'
    }
    
    for new_col, date_col in deltas.items():
        df[new_col] = (df[date_col] - base_date).dt.days
    
    # Extrair componentes de data
    df['purchase_year'] = base_date.dt.year
    df['purchase_month'] = base_date.dt.month
    df['purchase_day'] = base_date.dt.day

    return df.select_dtypes(exclude=['datetime64[ns]'])

def calculate_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features relacionadas ao produto e frete"""
    df['product_cubic_volume'] = df['product_length_cm'] * df['product_height_cm'] * df['product_width_cm']
    df = df.drop(columns=['product_length_cm', 'product_height_cm', 'product_width_cm'])
    
    # Cálculos financeiros
    df['freight_percentage'] = df.freight_value / df.price
    df['net_revenue'] = df.price - df.freight_value
    df['revenue_per_order'] = df.price + df.freight_value
    df['is_high_freight'] = df.freight_percentage > df.freight_percentage.median()
    
    return df

def create_categorical_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Cria features categóricas"""
    categorical_columns = {
        'payment_type': 'payment_type_',
        'customer_state': 'customer_state_',
        'seller_state': 'seller_state_',
        'product_category_name': 'product_category_name_'
    }
    
    df_categorical = pd.DataFrame(columns=columns[21:], index=[0], data=False)
    
    for col, prefix in categorical_columns.items():
        value = df[col].iloc[0]
        if pd.notna(value):
            col_name = f"{prefix}{value}"
            if col_name in df_categorical.columns:
                df_categorical[col_name] = True
    
    return df_categorical

######################

# Configuração da página
st.set_page_config(
    page_title="Dashboard Predição de Satisfação",
    page_icon="📊",
    layout="wide"
)

# Título do dashboard
st.title("📊 Dashboard Predição de Satisfação")

# Sidebar com upload e informações
with st.sidebar:
    st.header("Configurações")
    
    # Upload do arquivo CSV
    uploaded_file = st.file_uploader("Escolha um arquivo CSV com dados dos pedidos", type="csv")
    
    if uploaded_file is not None:
        # Carregar dados
        df_ = pd.read_csv(uploaded_file)

        selected_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "payment_sequential",
        "payment_type",
        "payment_installments",
        "payment_value",
        "customer_state",
        "order_item_id",
        "shipping_limit_date",
        "price",
        "freight_value",
        "product_category_name",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
        "seller_state"
        ]
        df_teste = df_[selected_columns]

        st.header("Informações dos Pedidos")
        st.write(f"Total de Pedidos: {df_teste.shape[0]}")
        st.write(f"Total de Features: {df_teste.shape[1]}")
        
        # Seletor de colunas
        selected_columns = st.multiselect(
            "Selecione as colunas para visualizar",
            options=df_teste.columns,
            default=['price', 'payment_value', 'freight_value', 'customer_state']
        )

# Conteúdo principal
if uploaded_file is not None:
    # Layout principal
    tab1, tab2, tab3 = st.tabs(["📋 Dados", "📊 Gráficos", "📈 Predição"])
    
    with tab1:
        # Visualização dos dados
        st.subheader("Visualização dos Dados")
        st.dataframe(df_teste[selected_columns], use_container_width=True)
        
    with tab2:
        st.subheader("Visualização Gráfica")
        
        # Seletor de tipo de gráfico
        chart_type = st.selectbox(
            "Escolha o tipo de gráfico",
            ["Histograma", "Scatter Plot", "Box Plot"]
        )
        
        # Colunas numéricas
        numeric_cols = df_teste.select_dtypes(include=['float64', 'int64']).columns
        
        if chart_type == "Histograma":
            col = st.selectbox("Selecione a coluna", numeric_cols)
            fig = px.histogram(df_teste, x=col)
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Scatter Plot":
            col_x = st.selectbox("Selecione a coluna X", numeric_cols)
            col_y = st.selectbox("Selecione a coluna Y", numeric_cols)
            fig = px.scatter(df_teste, x=col_x, y=col_y)
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Box Plot":
            col = st.selectbox("Selecione a coluna", numeric_cols)
            fig = px.box(df_teste, y=col)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Predição de Satisfação dos pedidos")       
        # Estatísticas descritivas
        st.write("Resultados a partir do modelo que avalia se o cliente irá atribuir uma nota ao pedido.")
        st.write("Sendo maior ou igual a 4: Satisfeito, menor que 4: Insatisfeito.")
        
        # 1. Primeiro carregamos o modelo e fazemos a predição
        model, scaler, columns = load_model_files()
        # Pipeline de processamento
        df_temp = process_datetime_features(df_teste)
        df_temp = calculate_product_features(df_temp)
        df_categorical = create_categorical_features(df_temp, columns)
        # Combinar features e aplicar scaler
        df_final = pd.concat([df_temp, df_categorical], axis=1)[columns]
        df_scaled = pd.DataFrame(scaler.transform(df_final), columns=columns)

        # 2. Fazemos a predição
        prediction = model.predict(df_scaled)
        df_teste['prediction'] = prediction
        df_teste['prediction_label'] = df_teste['prediction'].apply(lambda x: 'Satisfeito' if x == 1 else 'Insatisfeito')
        
        # 3. Agora criamos os gráficos
        # Gráfico de barras para distribuição geral
        fig_bar = px.bar(df_teste['prediction_label'].value_counts(), 
                        title='Distribuição das Predições de Satisfação',
                        labels={'value': 'Quantidade', 'index': 'Status'},
                        color_discrete_sequence=['#636EFA', '#EF553B'])
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # 4. Análise dos clientes insatisfeitos
        df_insatisfeitos = df_teste[df_teste['prediction_label'] == 'Insatisfeito']
        
        # Gráfico de análise por estado para clientes insatisfeitos
        fig_insatisfeitos = px.bar(df_insatisfeitos['customer_state'].value_counts(),
                                  title='Distribuição Prevista de Clientes Insatisfeitos por Estado',
                                  labels={'value': 'Quantidade', 'index': 'Estado'},
                                  color_discrete_sequence=['#EF553B'])
        st.plotly_chart(fig_insatisfeitos, use_container_width=True)
        
        # 5. Métricas finais
        st.subheader("Análise dos Clientes Insatisfeitos previstos")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Insatisfeitos", len(df_insatisfeitos))
        with col2:
            st.metric("Porcentagem Insatisfeitos", f"{len(df_insatisfeitos) / len(df_teste) * 100:.2f}%")
        with col3:
            st.metric("Média de Preço dos Insatisfeitos", f"R$ {df_insatisfeitos['price'].mean():.2f}")
        
        st.dataframe(df_teste[['price', 'payment_value', 'freight_value', 'customer_state', 'prediction_label']], use_container_width=True)



else:
    st.info("👈 Por favor, faça upload de um arquivo CSV na barra lateral para começar.")



