# IBOVESPA Predictor - Tech Challenge
# Autor: JP Lucchi

# Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configurações de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# %% [markdown]
# # 1. Carregamento e Preparação dos Dados

# %%
# Função para carregar e preparar os dados do IBOVESPA
def load_ibovespa_data(file_path):
    """
    Carrega e prepara os dados do IBOVESPA a partir de um arquivo CSV.
    
    Args:
        file_path (str): Caminho para o arquivo CSV
        
    Returns:
        DataFrame: DataFrame pandas com os dados preparados
    """
    # Carregar os dados do CSV
    df = pd.read_csv(file_path)
    
    # Converter o formato da data (DD.MM.YYYY para YYYY-MM-DD)
    df['Data'] = pd.to_datetime(df['Data'], format='%d.%m.%Y')
    
    # Renomear colunas para inglês para facilitar
    df = df.rename(columns={
        'Data': 'date',
        'Último': 'close',
        'Abertura': 'open',
        'Máxima': 'high',
        'Mínima': 'low',
        'Vol.': 'volume',
        'Var%': 'pct_change'
    })
    
    # Limpar a coluna de variação percentual
    df['pct_change'] = df['pct_change'].str.replace('%', '').str.replace(',', '.').astype(float)
    
    # Limpar a coluna de volume
    df['volume'] = df['volume'].str.replace(',', '.').str.replace('B', 'e9').str.replace('M', 'e6').str.replace('K', 'e3')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    
    # Ordenar por data (mais antiga para mais recente)
    df = df.sort_values('date')
    
    # Definir a data como índice
    df.set_index('date', inplace=True)
    
    print(f"Dados carregados com sucesso. Período: {df.index.min().date()} até {df.index.max().date()}")
    print(f"Total de registros: {len(df)}")
    
    return df

# Carregar os dados
ibovespa = load_ibovespa_data('ibovespa_raw_data.csv')

# Exibir as primeiras linhas
ibovespa.head()

# %% [markdown]
# # 2. Análise Exploratória de Dados (EDA)

# %%
# Estatísticas descritivas
print("Estatísticas Descritivas do IBOVESPA:")
ibovespa.describe()

# %%
# Visualizar a série temporal do preço de fechamento
plt.figure(figsize=(16, 8))
plt.plot(ibovespa.index, ibovespa['close'], linewidth=1.5)
plt.title('Evolução do IBOVESPA', fontsize=16)
plt.xlabel('Data')
plt.ylabel('Pontos')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Visualizar retornos diários
plt.figure(figsize=(16, 8))
plt.plot(ibovespa.index, ibovespa['pct_change'], linewidth=0.8, color='blue', alpha=0.7)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title('Retornos Diários do IBOVESPA (%)', fontsize=16)
plt.xlabel('Data')
plt.ylabel('Retorno (%)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Histograma dos retornos diários
plt.figure(figsize=(12, 6))
sns.histplot(ibovespa['pct_change'], bins=50, kde=True)
plt.title('Distribuição dos Retornos Diários', fontsize=16)
plt.xlabel('Retorno (%)')
plt.ylabel('Frequência')
plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Análise de Autocorrelação
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Função de Autocorrelação (ACF)
lags = 30
acf_values = acf(ibovespa['close'], nlags=lags)
ax1.stem(range(lags+1), acf_values, linefmt='b-', markerfmt='bo', basefmt='r-')
ax1.set_title('Função de Autocorrelação (ACF) do Preço de Fechamento', fontsize=14)
ax1.set_xlabel('Lags')
ax1.set_ylabel('Autocorrelação')
ax1.grid(True, alpha=0.3)

# Função de Autocorrelação Parcial (PACF)
pacf_values = pacf(ibovespa['close'], nlags=lags)
ax2.stem(range(lags+1), pacf_values, linefmt='b-', markerfmt='bo', basefmt='r-')
ax2.set_title('Função de Autocorrelação Parcial (PACF) do Preço de Fechamento', fontsize=14)
ax2.set_xlabel('Lags')
ax2.set_ylabel('Autocorrelação Parcial')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Teste de Estacionariedade (Augmented Dickey-Fuller)
def test_stationarity(timeseries, window=12, title=''):
    """
    Realiza o teste de estacionariedade na série temporal.
    
    Args:
        timeseries: Série temporal para testar
        window: Tamanho da janela móvel para o gráfico
        title: Título adicional para o gráfico
    
    Returns:
        results: Resultados do teste ADF
    """
    # Calcular estatísticas móveis
    rolling_mean = timeseries.rolling(window=window).mean()
    rolling_std = timeseries.rolling(window=window).std()
    
    # Plotar as estatísticas móveis
    plt.figure(figsize=(14, 7))
    plt.subplot(211)
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolling_mean, color='red', label=f'Média Móvel (janela={window})')
    plt.plot(rolling_std, color='green', label=f'Desvio Padrão Móvel (janela={window})')
    plt.title(f'Análise de Estacionariedade: {title}', fontsize=16)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Realizar teste Augmented Dickey-Fuller
    result = adfuller(timeseries.dropna())
    
    # Exibir resultados do teste
    plt.subplot(212)
    plt.text(0.01, 0.8, f'Estatística do Teste ADF: {result[0]:.4f}', transform=plt.gca().transAxes)
    plt.text(0.01, 0.7, f'Valor-p: {result[1]:.4f}', transform=plt.gca().transAxes)
    plt.text(0.01, 0.6, f'Valores Críticos:', transform=plt.gca().transAxes)
    for key, value in result[4].items():
        plt.text(0.01, 0.5 - 0.1 * list(result[4].keys()).index(key), 
                 f'   {key}: {value:.4f}', transform=plt.gca().transAxes)
    
    interpretation = "A série é estacionária" if result[1] < 0.05 else "A série não é estacionária"
    plt.text(0.01, 0.1, f'Interpretação: {interpretation}', transform=plt.gca().transAxes, 
             fontweight='bold', bbox=dict(facecolor='yellow', alpha=0.3))
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return result

# Testar estacionariedade do preço de fechamento
adf_result = test_stationarity(ibovespa['close'], title='Preço de Fechamento')

# Testar estacionariedade da primeira diferença
ibovespa['close_diff'] = ibovespa['close'].diff().dropna()
adf_result_diff = test_stationarity(ibovespa['close_diff'].dropna(), title='Primeira Diferença do Preço de Fechamento')

# %% [markdown]
# # 3. Engenharia de Features

# %%
def create_features(df):
    """
    Cria features adicionais para o modelo preditivo.
    
    Args:
        df: DataFrame com os dados do IBOVESPA
        
    Returns:
        DataFrame: DataFrame com as novas features
    """
    # Criar uma cópia para não modificar o original
    data = df.copy()
    
    # Médias Móveis
    for window in [5, 10, 20, 30, 60]:
        data[f'sma_{window}'] = data['close'].rolling(window=window).mean()
    
    # Momentum
    for window in [5, 10, 20]:
        data[f'momentum_{window}'] = data['close'] - data['close'].shift(window)
    
    # Retorno
    for window in [1, 5, 10, 20]:
        data[f'return_{window}'] = data['close'].pct_change(periods=window) * 100
    
    # Volatilidade
    for window in [5, 10, 20, 30]:
        data[f'volatility_{window}'] = data['pct_change'].rolling(window=window).std()
    
    # Indicadores de amplitude
    data['daily_range'] = data['high'] - data['low']
    data['daily_range_pct'] = data['daily_range'] / data['open'] * 100
    
    # Indicadores de dias da semana
    data['day_of_week'] = data.index.dayofweek
    for i in range(5):  # 0 = Segunda, 4 = Sexta
        data[f'is_day_{i}'] = (data['day_of_week'] == i).astype(int)
    
    # Lags (valores defasados)
    for lag in [1, 2, 3, 5, 10]:
        data[f'lag_{lag}_close'] = data['close'].shift(lag)
        data[f'lag_{lag}_return'] = data['pct_change'].shift(lag)
    
    # Remover linhas com valores NaN
    data = data.dropna()
    
    print(f"Features criadas. Nova dimensão: {data.shape}")
    
    return data

# Criar features
ibovespa_features = create_features(ibovespa)

# Exibir as novas colunas
print("\nNovas colunas:")
new_columns = [col for col in ibovespa_features.columns if col not in ibovespa.columns]
print(new_columns)

# Exibir as primeiras linhas com as novas features
ibovespa_features.head()

# %% [markdown]
# # 4. Preparação para Modelagem

# %%
# Definir a variável alvo e as features
target_column = 'close'
feature_columns = [col for col in ibovespa_features.columns if col != target_column and col != 'pct_change' and col != 'close_diff']

X = ibovespa_features[feature_columns]
y = ibovespa_features[target_column]

# Dividir os dados em conjuntos de treino e teste
train_size = int(len(X) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

print(f"Conjunto de treino: {X_train.shape}")
print(f"Conjunto de teste: {X_test.shape}")

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Converter de volta para DataFrame para manter os nomes das colunas
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

# %% [markdown]
# # 5. Implementação dos Modelos

# %% [markdown]
# ## 5.1 Modelo Baseline - Média Móvel Simples (SMA)

# %%
class SimpleMovingAverage:
    def __init__(self, window=5):
        self.window = window
    
    def predict(self, data):
        return data.rolling(window=self.window).mean().values
    
    def evaluate(self, true_values, predictions):
        # Ignorar NaNs (que ocorrem no início devido à janela)
        valid_indices = ~np.isnan(predictions)
        true_values = true_values[valid_indices]
        predictions = predictions[valid_indices]
        
        mae = mean_absolute_error(true_values, predictions)
        mse = mean_squared_error(true_values, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_values, predictions)
        
        # Calcular a acurácia direcional
        direction_accuracy = self._calculate_direction_accuracy(true_values, predictions)
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'Direction Accuracy': direction_accuracy
        }
    
    def _calculate_direction_accuracy(self, true_values, predictions):
        true_direction = np.diff(true_values) > 0
        pred_direction = np.diff(predictions) > 0
        
        correct_direction = np.sum(true_direction == pred_direction)
        total_days = len(true_direction)
        
        return correct_direction / total_days * 100

# Implementar e avaliar o modelo SMA
sma_model = SimpleMovingAverage(window=5)

# Preparar série histórica completa para SMA
historical_closes = ibovespa['close'].values

# Previsões do SMA (deslocadas 1 dia para frente simulando previsões reais)
sma_predictions = np.roll(sma_model.predict(ibovespa['close']), 1)

# Avaliar o modelo apenas no período de teste
test_indices = y_test.index
sma_test_true = ibovespa.loc[test_indices, 'close'].values
sma_test_pred = sma_predictions[-len(test_indices):]

# Calcular métricas
sma_metrics = sma_model.evaluate(sma_test_true, sma_test_pred)
print("Métricas do modelo SMA:")
for metric, value in sma_metrics.items():
    print(f"{metric}: {value:.4f}")

# %% [markdown]
# ## 5.2 Modelo de Regressão Linear Múltipla

# %%
# Implementar modelo de regressão linear múltipla
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Fazer previsões
lr_train_pred = lr_model.predict(X_train_scaled)
lr_test_pred = lr_model.predict(X_test_scaled)

# Avaliar o modelo
def evaluate_model(y_true, y_pred, model_name="Modelo"):
    """
    Avalia o desempenho do modelo.
    
    Args:
        y_true: Valores reais
        y_pred: Valores previstos pelo modelo
        model_name: Nome do modelo para exibição
        
    Returns:
        dict: Dicionário com as métricas calculadas
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Calcular a acurácia direcional
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    
    correct_direction = np.sum(true_direction == pred_direction)
    total_days = len(true_direction)
    
    direction_accuracy = correct_direction / total_days * 100
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Direction Accuracy': direction_accuracy
    }
    
    print(f"Métricas do {model_name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

# Avaliar o modelo de regressão linear
lr_metrics = evaluate_model(y_test, lr_test_pred, "Modelo de Regressão Linear")

# Visualizar as previsões vs valores reais
plt.figure(figsize=(16, 8))
plt.plot(y_test.index, y_test.values, label='Valores Reais', color='blue')
plt.plot(y_test.index, lr_test_pred, label='Previsões LR', color='red', linestyle='--')
plt.title('Valores Reais vs. Previsões (Regressão Linear)', fontsize=16)
plt.xlabel('Data')
plt.ylabel('Pontos do IBOVESPA')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Analisar os coeficientes mais importantes
coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': lr_model.coef_
})
coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)

plt.figure(figsize=(14, 10))
sns.barplot(x='Abs_Coefficient', y='Feature', data=coef_df.head(15))
plt.title('15 Features Mais Importantes (Regressão Linear)', fontsize=16)
plt.xlabel('Importância (Valor Absoluto do Coeficiente)')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5.3 Modelo Auto-Regressivo (AR)

# %%
from statsmodels.tsa.ar_model import AutoReg

# Implementar modelo auto-regressivo
def train_ar_model(data, lags=5):
    """
    Treina um modelo auto-regressivo.
    
    Args:
        data: Série temporal para treinar
        lags: Número de lags a considerar
        
    Returns:
        model_fit: Modelo AR treinado
    """
    model = AutoReg(data, lags=lags)
    model_fit = model.fit()
    return model_fit

# Treinar modelo AR com os dados de treino
ar_model = train_ar_model(y_train, lags=5)

# Fazer previsões in-sample e out-of-sample
ar_train_pred = ar_model.predict(start=5, end=len(y_train)-1)
ar_test_pred = ar_model.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)

# Avaliar o modelo AR
# Precisamos alinhar os índices para avaliação adequada
ar_test_metrics = evaluate_model(y_test, ar_test_pred, "Modelo Auto-Regressivo (AR)")

# Visualizar as previsões vs valores reais
plt.figure(figsize=(16, 8))
plt.plot(y_test.index, y_test.values, label='Valores Reais', color='blue')
plt.plot(y_test.index, ar_test_pred, label='Previsões AR', color='green', linestyle='--')
plt.title('Valores Reais vs. Previsões (Auto-Regressivo)', fontsize=16)
plt.xlabel('Data')
plt.ylabel('Pontos do IBOVESPA')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5.4 Modelo Ensemble

# %%
# Criar um modelo ensemble (média ponderada dos modelos individuais)
def ensemble_predictions(predictions_list, weights=None):
    """
    Combina as previsões de vários modelos usando média ponderada.
    
    Args:
        predictions_list: Lista de arrays com as previsões de cada modelo
        weights: Pesos para cada modelo (se None, usa pesos iguais)
        
    Returns:
        array: Previsões combinadas
    """
    if weights is None:
        weights = [1/len(predictions_list)] * len(predictions_list)
    
    # Verificar se as previsões têm o mesmo tamanho
    lengths = [len(pred) for pred in predictions_list]
    if len(set(lengths)) > 1:
        raise ValueError("Todas as previsões devem ter o mesmo tamanho")
    
    # Calcular a média ponderada
    ensemble_pred = np.zeros(lengths[0])
    for i, pred in enumerate(predictions_list):
        ensemble_pred += weights[i] * pred
        
    return ensemble_pred

# Combinar as previsões dos modelos individuais
# Pesos: 0.2 para SMA, 0.5 para LR, 0.3 para AR
ensemble_weights = [0.2, 0.5, 0.3]
ensemble_predictions_test = ensemble_predictions(
    [sma_test_pred, lr_test_pred, ar_test_pred],
    weights=ensemble_weights
)

# Avaliar o modelo ensemble
ensemble_metrics = evaluate_model(y_test, ensemble_predictions_test, "Modelo Ensemble")

# Visualizar as previsões de todos os modelos juntos
plt.figure(figsize=(16, 8))
plt.plot(y_test.index, y_test.values, label='Valores Reais', color='blue', linewidth=2)
plt.plot(y_test.index, sma_test_pred, label='SMA', color='orange', linestyle='--', alpha=0.7)
plt.plot(y_test.index, lr_test_pred, label='Regressão Linear', color='red', linestyle='--', alpha=0.7)
plt.plot(y_test.index, ar_test_pred, label='Auto-Regressivo', color='green', linestyle='--', alpha=0.7)
plt.plot(y_test.index, ensemble_predictions_test, label='Ensemble', color='purple', linewidth=1.5)
plt.title('Comparação das Previsões de Todos os Modelos', fontsize=16)
plt.xlabel('Data')
plt.ylabel('Pontos do IBOVESPA')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# # 6. Comparação dos Modelos

# %%
# Reunir as métricas de todos os modelos para comparação
models_metrics = {
    'SMA': sma_metrics,
    'Regressão Linear': lr_metrics,
    'Auto-Regressivo': ar_test_metrics,
    'Ensemble': ensemble_metrics
}

# Criar DataFrame para visualizar a comparação
metrics_df = pd.DataFrame(models_metrics).T
metrics_df = metrics_df.round(4)

# Visualizar comparação
print("Comparação das Métricas de Todos os Modelos:")
print(metrics_df)

# Visualizar graficamente
metrics_to_plot = ['MAE', 'RMSE', 'R2', 'Direction Accuracy']
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, metric in enumerate(metrics_to_plot):
    ax = axes[i]
    
    if metric == 'R2':
        # R2 pode ser negativo, então precisamos de uma abordagem diferente
        colors = ['green' if val >= 0 else 'red' for val in metrics_df[metric]]
        sns.barplot(x=metrics_df.index, y=metrics_df[metric], ax=ax, palette=colors)
    else:
        # Para outras métricas, menor é melhor (exceto acurácia direcional)
        if metric != 'Direction Accuracy':
            # Inverter a ordem para que menores valores apareçam maiores no gráfico
            sns.barplot(x=metrics_df.index, y=metrics_df[metric], ax=ax, 
                       palette='viridis')
        else:
            # Acurácia direcional: maior é melhor
            sns.barplot(x=metrics_df.index, y=metrics_df[metric], ax=ax, 
                       palette='viridis')
    
    ax.set_title(f'Comparação de {metric}', fontsize=14)
    ax.set_xlabel('Modelo')
    ax.set_ylabel(metric)
    
    # Adicionar valores nas barras
    for j, val in enumerate(metrics_df[metric]):
        ax.text(j, val, f'{val:.2f}', ha='center', va='bottom')
    
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# # 7. Fazer Previsão para o Próximo Dia

# %%
# Função para fazer previsão para o próximo dia
def predict_next_day(model_type, data, feature_data=None, ar_model=None, lr_model=None):
    """
    Faz previsão para o próximo dia com base no modelo especificado.
    
    Args:
        model_type: Tipo de modelo ('sma', 'lr', 'ar' ou 'ensemble')
        data: Série temporal completa
        feature_data: DataFrame com as features (para modelo LR)
        ar_model: Modelo AR treinado
        lr_model: Modelo LR treinado
        
    Returns:
        float: Previsão para o próximo dia
    """
    if model_type == 'sma':
        # Previsão usando SMA
        last_values = data.iloc[-5:].values
        prediction = np.mean(last_values)
        
    elif model_type == 'lr':
        # Previsão usando Regressão Linear
        latest_features = feature_data.iloc[-1:].values
        latest_features_scaled = scaler.transform(latest_features)
        prediction = lr_model.predict(latest_features_scaled)[0]
        
    elif model_type == 'ar':
        # Previsão usando modelo AR
        prediction = ar_model.predict(start=len(data), end=len(data))[0]
        
    elif model_type == 'ensemble':
        # Previsão usando todos os modelos combinados
        sma_pred = predict_next_day('sma', data)
        lr_pred = predict_next_day('lr', data, feature_data, ar_model, lr_model)
        ar_pred = predict_next_day('ar', data, feature_data, ar_model, lr_model)
        
        prediction = ensemble_weights[0] * sma_pred + ensemble_weights[1] * lr_pred + ensemble_weights[2] * ar_pred
        
    else:
        raise ValueError(f"Tipo de modelo não reconhecido: {model_type}")
        
    return prediction

# Fazer previsões para o próximo dia usando todos os modelos
next_day = ibovespa.index[-1] + pd.Timedelta(days=1)

sma_prediction = predict_next_day('sma', ibovespa['close'])
lr_prediction = predict_next_day('lr', ibovespa['close'], X, ar_model, lr_model)
ar_prediction = predict_next_day('ar', ibovespa['close'], X, ar_model, lr_model)
ensemble_prediction = predict_next_day('ensemble', ibovespa['close'], X, ar_model, lr_model)

print(f"\nPrevisões para o próximo dia de negociação ({next_day.date()}):")
print(f"SMA: {sma_prediction:.2f}")
print(f"Regressão Linear: {lr_prediction:.2f}")
print(f"Auto-Regressivo: {ar_prediction:.2f}")
print(f"Ensemble: {ensemble_prediction:.2f}")

# %% [markdown]
# # 8. Conclusões

# %% [markdown]
# Com base nas análises realizadas, podemos concluir que:
# 
# 1. A série temporal do IBOVESPA apresenta características não-estacionárias, com tendências e padrões complexos.
# 
# 2. Os modelos implementados apresentaram diferentes