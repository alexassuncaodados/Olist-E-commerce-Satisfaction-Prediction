from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from typing import Optional
import joblib
import os
from functools import lru_cache

# Constantes
MODEL_PATH = '../output/'
MODEL_FILE = 'RandomForestClassifier_model.pkl'
SCALER_FILE = 'scaler.pkl'
COLUMNS_FILE = 'columns.pkl'

# Colunas datetime para processamento
DATETIME_COLUMNS = [
    'order_purchase_timestamp', 
    'order_approved_at', 
    'order_delivered_carrier_date', 
    'order_delivered_customer_date', 
    'order_estimated_delivery_date', 
    'shipping_limit_date'
]

class OrderInput(BaseModel):
    order_purchase_timestamp: object
    order_approved_at: object
    order_delivered_carrier_date: object
    order_delivered_customer_date: object
    order_estimated_delivery_date: object
    payment_sequential: Optional[int] = None
    payment_type: object
    payment_installments: Optional[int] = None
    payment_value: float
    customer_state: object
    order_item_id: Optional[int] = None
    shipping_limit_date: object
    price: float
    freight_value: float
    product_category_name: object
    product_photos_qty: Optional[int] = None
    product_weight_g: float
    product_length_cm: float
    product_height_cm: float
    product_width_cm: float
    seller_state: object

    class Config:
        from_attributes = True

@lru_cache(maxsize=1)
def load_model_files():
    """Carrega e cacheia os arquivos do modelo"""
    try:
        if not os.path.exists(MODEL_PATH + MODEL_FILE):
            raise FileNotFoundError(f"Modelo não encontrado em: {MODEL_PATH + MODEL_FILE}")
        
        model = joblib.load(MODEL_PATH + MODEL_FILE)
        scaler = joblib.load(MODEL_PATH + SCALER_FILE)
        columns = joblib.load(MODEL_PATH + 'columns.pkl')
        
        return model, scaler, columns
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao carregar arquivos do modelo: {str(e)}")

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

app = FastAPI()

@app.get('/')
def health_check():
    model, scaler, _ = load_model_files()
    return {
        'status': 'healthy',
        'message': 'API do Modelo de Classificação Olist E-Commerce',
        'model_type': type(model).__name__,
        'scaler_type': type(scaler).__name__
    }

@app.post('/predict')
def predict(order_input: OrderInput):
    # Carregar arquivos necessários
    model, scaler, columns = load_model_files()
    
    # Converter input para DataFrame
    df = pd.DataFrame([order_input.model_dump()])
    
    # Pipeline de processamento
    df = process_datetime_features(df)
    df = calculate_product_features(df)
    df_categorical = create_categorical_features(df, columns)
    
    # Combinar features e aplicar scaler
    df_final = pd.concat([df, df_categorical], axis=1)[columns]
    df_scaled = pd.DataFrame(scaler.transform(df_final), columns=columns)
    
    # Fazer predição
    prediction = model.predict(df_scaled)
    
    return {
        'prediction': int(prediction[0]),
        'prediction_label': 'Satisfeito' if prediction[0] == 1 else 'Insatisfeito'
    }