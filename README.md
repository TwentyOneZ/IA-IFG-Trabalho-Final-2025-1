Os dados analisados foram das criptomoedas BTC, ETH, LTC, XRP, BCH, XMR, DASH, ETC, BAT, ZRX, EOS, LSK, usando o histórico do preço de fechamento diário na Poloniex.

# Instruções de Uso do `main.py`

Este README descreve passo a passo como executar cada comando disponível no `main.py`, quais pastas são geradas e o que cada argumento faz.

## Comandos Disponíveis

### 1. analyze
Gera gráficos estatísticos da criptomoeda selecionada.

**Uso:**
```
python main.py analyze --crypto <SYMBOL> [--show] [--save]
```
- `--crypto`: símbolo da criptomoeda (ex: BTC, ETH).
- `--show`: exibe os gráficos na tela.
- `--save`: salva os gráficos em `figures/` dentro de subpastas (`boxplot/`, `hist/`, `lineplot/`).

Os gráficos gerados usam os dados de fechamento diário da criptomoeda, e são gerados: boxplot, histograma e evolução do preço com média, mediana e moda. Todos os gráficos gerados estão nas pastas `figures/` e as respectivas subpastas (`boxplot/`, `hist/`, `lineplot/`).

### 2. simulate
Simula estratégia de investimento baseado em previsão e buy&hold.

**Uso:**
```
python main.py simulate --crypto <SYMBOL> --model <MODEL> [--start-date YYYY-MM-DD] [--save]
```
- `--crypto`: símbolo da criptomoeda.
- `--model`: modelo a ser utilizado (`linear`, `mlp`, `poly_deg2`, etc.).
- `--start-date`: data inicial para iniciar a simulação.
- `--save`: salva gráfico em `figures/lucro/` e planilha CSV.

### 3. train
Treina modelos (linear, MLP, polinomial) com K-Fold cross validation.

**Uso:**
```
python main.py train --crypto <SYMBOL> [--kfolds N]
```
- `--crypto`: símbolo da criptomoeda.
- `--kfolds`: número de folds (padrão 5).

### 4. compare
Compara todos os modelos, gera scatter, evolução de capital e salva detalhes.

**Uso:**
```
python main.py compare --crypto <SYMBOL>
```
- `--crypto`: símbolo da criptomoeda.

Gera:
- `figures/compare/results_<SYMBOL>.txt`
- `figures/compare/scatter_<SYMBOL>.png`
- `figures/compare/profit_<SYMBOL>.png`

### 5. hypothesis
Executa teste de hipótese H0: retorno diário ≥ limiar.

**Uso:**
```
python main.py hypothesis --threshold <VALOR_PCT> [--save]
```
- `--threshold`: valor percentual a ser testado (ex: 1.5 para 1.5%).
- `--save`: salva resultados em `results/`.

### 6. anova
Realiza ANOVA one-way entre retornos médios diários de todas as criptomoedas.

**Uso:**
```
python main.py anova --alpha <NÍVEL_SIGNIFICÂNCIA> [--save]
```
- `--alpha`: nível de significância (padrão 0.05).
- `--save`: salva resultados em `results/`.

### 7. anova-groups
Agrupa criptomoedas por métrica e realiza ANOVA/Tukey HSD.

**Uso:**
```
python main.py anova-groups --metric <volatility|mean_return|volume> --alpha <NÍVEL> [--save]
```
- `--metric`: métrica de agrupamento.
- `--alpha`: nível de significância.
- `--save`: salva resultados e atribuições em `results/`.

## Estrutura de Pastas de Saída

```
figures/
├── boxplot/
│   └── boxplot_<SYMBOL>.png
├── hist/
│   └── hist_<SYMBOL>.png
├── lineplot/
│   └── lineplot_<SYMBOL>.png
├── lucro/
│   ├── lucro_<SYMBOL>_<MODEL>.png
│   └── lucro_<SYMBOL>_<MODEL>.csv
└── compare/
    ├── scatter_<SYMBOL>.png
    ├── scatter_test_<SYMBOL>.png
    ├── profit_<SYMBOL>.png
    ├── profit_test_<SYMBOL>.png
    ├── results_<SYMBOL>.txt
    └── results_<SYMBOL>.csv

results/
├── anova.csv
├── anova_groups.csv
├── tukey_groups.csv
├── hypothesis_results.csv
└── group_assignments.csv

tests/
└── test_*.py  (suite de testes automatizados)