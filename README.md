# Previsão do IBOVESPA - Modelo Preditivo para Séries Temporais

## Descrição do Projeto

Este projeto foi desenvolvido como parte do Tech Challenge para criar um modelo preditivo capaz de prever diariamente o valor de fechamento do IBOVESPA, o principal índice da bolsa de valores brasileira.

O projeto inclui todo o processo de Data Science, desde a captura e limpeza dos dados, análise exploratória, engenharia de features, até o desenvolvimento e validação de vários modelos preditivos.

## Objetivos

1. Desenvolver um modelo com storytelling completo, desde a captura do dado até a entrega
2. Justificar a técnica utilizada
3. Atingir uma acuracidade adequada (acima de 70%)

## Estrutura do Projeto

- `README.md`: Documentação completa do projeto
- `notebook.py`: Código Python com a análise e implementação dos modelos
- `ibovespa_raw_data.csv`: Dataset do IBOVESPA utilizado para análise
- `dashboard.html`: Dashboard interativo para visualização dos resultados
- `requirements.txt`: Lista de dependências necessárias para executar o código

## Metodologia

### 1. Obtenção e Preparação dos Dados

Obtive os dados históricos do IBOVESPA através do site da Investing.com, cobrindo o período de março de 2019 a fevereiro de 2025. O dataset contém informações diárias sobre:

- Data
- Preço de abertura
- Preço de fechamento
- Preço máximo
- Preço mínimo
- Volume de negociação
- Variação percentual

Após a coleta, realizei a limpeza dos dados, convertendo o formato das datas, tratando valores ausentes e normalizando os campos numéricos.

### 2. Análise Exploratória de Dados (EDA)

Realizei uma análise exploratória para entender o comportamento da série temporal do IBOVESPA:

- Estatísticas descritivas do preço de fechamento
- Análise de tendências e sazonalidade
- Verificação de estacionariedade da série
- Cálculo de autocorrelação para identificar padrões temporais
- Análise de retornos diários e sua distribuição

Descobri que:
- A série apresenta alta correlação serial (autocorrelação)
- É necessário diferenciação para tornar a série estacionária
- Existem padrões sazonais semanais
- A volatilidade varia ao longo do tempo

### 3. Engenharia de Features

Para melhorar o desempenho dos modelos, criei as seguintes features:

- Médias móveis (SMA) de diferentes períodos (5, 10, 20 dias)
- Indicadores de momentum
- Volatilidade em diferentes janelas de tempo
- Valores defasados (lags) do preço de fechamento
- Retornos percentuais em diferentes horizontes temporais
- Amplitudes diárias (high-low)

### 4. Modelagem

Implementei e comparei vários modelos:

1. **Modelo de Média Móvel Simples (SMA)**: Utilizado como baseline
2. **Regressão Linear Múltipla**: Modelo que utiliza múltiplas variáveis para prever o preço
3. **Modelo Auto-Regressivo (AR)**: Baseado exclusivamente nos valores passados da série
4. **Modelo Ensemble**: Combinação ponderada dos modelos anteriores

### 5. Avaliação dos Modelos

Métricas utilizadas para avaliar o desempenho dos modelos:
- MAE (Erro Médio Absoluto)
- RMSE (Raiz do Erro Quadrático Médio)
- R² (Coeficiente de Determinação)
- Acurácia Direcional (capacidade de prever a direção do movimento)

## Resultados

Os modelos apresentaram os seguintes resultados no conjunto de teste:

| Modelo             | MAE     | RMSE    | R²      | Acurácia Direcional |
|--------------------|---------|---------|---------|---------------------|
| SMA (5 dias)       | 1.18    | 1.47    | 0.867   | 47.96%              |
| Regressão Linear   | 2.82    | 2.83    | 0.509   | 97.28%              |
| Auto-Regressivo    | 3.69    | 4.53    | -0.265  | 46.26%              |
| **Ensemble**       | **1.75**| **2.06**| **0.738**| **82.31%**          |

O modelo ensemble se mostrou o mais equilibrado, combinando as vantagens de cada abordagem individual. Embora o modelo de Regressão Linear tenha a maior acurácia direcional, o modelo ensemble apresenta melhor equilíbrio entre precisão de valor e direção.

## Features Mais Importantes

As features que mais contribuíram para o modelo foram:
1. Média móvel de 5 dias
2. Valores defasados (lag) de 1, 2 e 3 dias
3. Momentum de curto prazo
4. Preços máximo e mínimo

## Previsão para o Próximo Dia

Com base no modelo ensemble, a previsão para o próximo dia de negociação é:

- **Data**: 2025-03-01
- **Previsão de Fechamento**: 124.626

## Conclusões

1. A combinação de diferentes técnicas de modelagem (ensemble) produziu os melhores resultados
2. A série temporal do IBOVESPA apresenta padrões complexos que exigem abordagens sofisticadas
3. As features de curto prazo (5 dias) têm maior influência nas previsões
4. O modelo conseguiu atingir a meta de acurácia direcional superior a 70% (82.31%)

## Tecnologias Utilizadas

- Python
- Pandas & NumPy para manipulação de dados
- Matplotlib & Seaborn para visualizações
- Scikit-learn para modelagem e avaliação
- Statsmodels para análise de séries temporais

## Autor

JP Lucchi
