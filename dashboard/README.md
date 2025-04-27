# Dashboard de Predição de Satisfação Olist

## 📊 Visão Geral
Dashboard interativo desenvolvido com Streamlit para análise e predição de satisfação de clientes da Olist. O dashboard permite carregar dados de pedidos via CSV e oferece visualizações dos dados e predições de satisfação usando um modelo de machine learning pré-treinado.

## 📺 Demonstração em Vídeo

Um breve vídeo demonstrando a funcionalidade do dashboard:


https://github.com/user-attachments/assets/aaa44c2c-319d-4552-916e-2d6c47d658f9




## 🚀 Funcionalidades

### 1. Upload de Dados
- Suporte para arquivos CSV
- Validação automática das colunas necessárias
- Visualização prévia dos dados carregados

### 2. Visualização de Dados 
- Tabela interativa com dados dos pedidos
- Seleção personalizada de colunas
- Informações sobre total de pedidos e features

### 3. Análise Gráfica (Em Desenvolvimento)
- Tipos de gráficos disponíveis:
  - Histograma
  - Scatter Plot
  - Box Plot
- Seleção flexível de variáveis
- Visualizações interativas com Plotly

### 4. Predição de Satisfação
- Predição automática usando modelo RandomForest
- Visualizações:
  - Distribuição geral de satisfação
  - Análise por estado dos clientes insatisfeitos
- Métricas importantes:
  - Total de clientes insatisfeitos
  - Porcentagem de insatisfação
  - Média de preço dos pedidos insatisfeitos

## 🛠️ Requisitos Técnicos

```python
streamlit==1.28.0
pandas==2.1.1
plotly==5.17.0
scikit-learn==1.4.2
joblib==1.3.2
```

## 📋 Estrutura de Dados Necessária
O arquivo CSV deve conter as seguintes colunas:
```
- order_purchase_timestamp
- order_approved_at
- order_delivered_carrier_date
- order_delivered_customer_date
- order_estimated_delivery_date
- payment_sequential
- payment_type
- payment_installments
- payment_value
- customer_state
- order_item_id
- shipping_limit_date
- price
- freight_value
- product_category_name
- product_photos_qty
- product_weight_g
- product_length_cm
- product_height_cm
- product_width_cm
- seller_state
```

## 🚀 Como Executar

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Execute o dashboard:
```bash
streamlit run dashboard/app.py
```

3. Acesse via navegador:
```
http://localhost:8501
```

## 🔍 Modelo de Predição
- Utiliza RandomForestClassifier pré-treinado
- Arquivos necessários em '../output/':
  - RandomForestClassifier_model.pkl
  - scaler.pkl
  - columns.pkl

## 📈 Features Processadas
- Features temporais (deltas entre datas)
- Features do produto (volume, peso)
- Features financeiras (frete, preço)
- Features categóricas (estado, tipo pagamento)

## 👤 Autor
Alex Silva de Assunção
- [LinkedIn](https://www.linkedin.com/in/alexassuncaodata/)
- [GitHub](https://github.com/alexassuncaodados)

## 📫 Contato
- Email: alexassuncao.dados@email.com
