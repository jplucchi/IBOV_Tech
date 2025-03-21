# Modelo Preditivo para IBOVESPA - Tech Challenge

## Sobre o Projeto

Este projeto foi desenvolvido como parte do Tech Challenge da Pós-Graduação em Data Analytics. O objetivo principal foi criar um modelo preditivo capaz de prever o valor de fechamento diário do índice IBOVESPA com acurácia superior a 70%.

Utilizando técnicas avançadas de análise de séries temporais e aprendizado de máquina, consegui desenvolver um modelo que supera significativamente o requisito mínimo, alcançando uma acurácia de **99,86%** no conjunto de teste.

## Objetivos

- Criar uma série temporal com dados históricos da IBOVESPA
- Desenvolver um modelo preditivo para prever o fechamento diário do índice
- Justificar tecnicamente a escolha das abordagens utilizadas
- Atingir uma acurácia superior a 70%

## Tecnologias Utilizadas

- **Python**: Linguagem principal para implementação do projeto
- **Pandas**: Manipulação e análise de dados
- **NumPy**: Computação numérica
- **Matplotlib/Seaborn**: Visualização de dados
- **Scikit-learn**: Implementação dos modelos de machine learning
- **Statsmodels**: Análise estatística e modelagem de séries temporais

## Estrutura do Repositório

```
IBOV_Tech/
│
├── README.md                      # Este arquivo
├── ibovespa_raw_data.csv          # Dados históricos do IBOVESPA
├── notebook.py                    # Código-fonte principal do projeto
├── dashboard.html                 # Dashboard para visualização dos resultados
└── requirements.txt               # Dependências do projeto
```

## Como Executar o Projeto

1. Clone este repositório:
```bash
git clone https://github.com/seu-usuario/IBOV_Tech.git
cd IBOV_Tech
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute o notebook:
```bash
jupyter notebook notebook.py
```

## Resultados

Implementei e comparei três modelos diferentes:

| Modelo | Acurácia (Teste) | RMSE | R² |
|--------|-----------------|------|---|
| Média Móvel Ponderada | 99,08% | 1,4353 | 0,8967 |
| Regressão Linear Múltipla | 99,86% | 0,2337 | 0,9973 |
| ARIMA Simplificado | 99,68% | 0,5244 | 0,9862 |

O modelo de **Regressão Linear Múltipla** apresentou o melhor desempenho geral, com uma acurácia de 99,86% no conjunto de teste, superando significativamente o requisito mínimo de 70%.

### Features Mais Importantes

Análise da importância relativa das features no modelo:

1. **Retorno Logarítmico (Log Return)**: 76,59%
2. **Preço Mínimo**: 6,85%
3. **True Range (Volatilidade)**: 6,72%
4. **Preço Máximo**: 6,64%
5. **Tendência de Longo Prazo (30 dias)**: 0,75%

## Abordagem Metodológica

1. **Coleta e Preparação dos Dados**: Obtive dados históricos diários do IBOVESPA e realizei limpeza e transformação.

2. **Análise Exploratória**: Identifiquei padrões, tendências e sazonalidades nos dados.

3. **Engenharia de Features**: Criei novas variáveis baseadas em médias móveis, tendências, volatilidade e indicadores técnicos.

4. **Modelagem**: Implementei e comparei três modelos diferentes (Média Móvel Ponderada, Regressão Linear Múltipla e ARIMA).

5. **Avaliação e Seleção**: Selecionei o melhor modelo com base em métricas de desempenho como acurácia, RMSE e R².

6. **Previsão**: Utilizei o modelo selecionado para prever o valor de fechamento do próximo dia de negociação.

## Conclusões

O modelo desenvolvido demonstrou uma capacidade excepcional de prever o valor de fechamento do IBOVESPA, com uma acurácia muito superior ao mínimo exigido. A análise da importância das features revelou que o retorno logarítmico recente é o fator mais determinante para a previsão do próximo valor, seguido pelos preços mínimos, máximos e indicadores de volatilidade.

Esta abordagem oferece não apenas uma previsão precisa, mas também insights valiosos sobre os fatores que mais influenciam o comportamento futuro do índice, permitindo decisões de investimento mais informadas.

## Contato

Para qualquer dúvida ou sugestão sobre este projeto, entre em contato:

- **Nome**: JP Lucchi
- **Email**: jplucchi@hotmail.com

---

Desenvolvido como parte do Tech Challenge - Pós-Graduação em Data Analytics © 2025
