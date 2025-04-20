# Olist Customer Satisfaction API

API REST para predição de satisfação de clientes Olist usando modelo de machine learning.

## 🚀 Estrutura do Projeto

```
api/
├── main.py           # Aplicação FastAPI principal
├── teste.ipynb       # Notebook para testes da API
├── Dockerfile        # Configuração do container
├── docker-compose.yml# Configuração do ambiente Docker
└── requirements.txt  # Dependências do projeto
```

## 📋 Pré-requisitos

- Python 3.12+
- Docker (opcional)
- Arquivos do modelo em `/output`:
  - RandomForestClassifier_model.pkl
    - Modelo de classificação treinado, devido ao tamanho do arquivo não é possível incluir no repositório, sendo necessário treinar o modelo novamente, modelo de treino disponível em [Olist Customer Satisfaction Prediction](https://github.com/alexassuncaodados/Olist-E-commerce-Satisfaction-Prediction/blob/main/Olist%20E-commerce%20Customer%20Satisfaction%20Prediction%20Project.ipynb).
  - scaler.pkl
    - Escalador de dados treinado na base de treino
  - columns.pkl
    - Lista de colunas utilizadas no modelo

Obs.: Os arquivos são utilizados pela API para tratamento dos dados brutos submetidos.

## 🔧 Instalação

### Usando Python local

```bash
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt

# Iniciar servidor para localhost
uvicorn main:app --host 0.0.0.0 --port 8001
```

### Usando Docker

```bash
# Construir e iniciar container
docker-compose up --build
```

## 🔍 Endpoints

### GET /
Health check e informações da API

**Resposta**:
```json
{
    "status": "healthy",
    "message": "API do Modelo de Classificação Olist E-Commerce",
    "model_type": "RandomForestClassifier",
    "scaler_type": "StandardScaler"
}
```

### POST /predict
Predição de satisfação do cliente

**Dados principais para tratamento e predição**:
```json
{
    "order_purchase_timestamp": "2017-01-01 00:00:00",
    "order_approved_at": "2017-01-01 00:10:00",
    "order_delivered_carrier_date": "2017-01-02 00:00:00",
    "order_delivered_customer_date": "2017-01-05 00:00:00",
    "order_estimated_delivery_date": "2017-01-10 00:00:00",
    "payment_sequential": 1,
    "payment_type": "credit_card",
    "payment_installments": 1,
    "payment_value": 100.0,
    "customer_state": "SP",
    "shipping_limit_date": "2017-01-02 00:00:00",
    "price": 90.0,
    "freight_value": 10.0,
    "product_category_name": "electronics",
    "product_photos_qty": 3,
    "product_weight_g": 200,
    "product_length_cm": 30,
    "product_height_cm": 10,
    "product_width_cm": 20,
    "seller_state": "SP"
}
```

**Resposta**:
```json
{
    "prediction": 1,
    "prediction_label": "Satisfeito"
}
```

## 🧪 Testes

Use o notebook `teste.ipynb` para:
- Testar endpoints
- Processar múltiplas predições em paralelo
- Validar resultados

## 🛠️ Tecnologias

- FastAPI
- Scikit-learn
- Pandas
- Docker



## 📝 Notas

- Todas as datas devem ser enviadas no formato "YYYY-MM-DD HH:MM:SS"
- O processamento inclui feature engineering dos dados de entrada não sendo necessário enviar os dados já tratados, a base de teste utilizada permanece com os dados brutos.