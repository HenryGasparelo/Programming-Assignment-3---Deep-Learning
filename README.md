# Deep Learning: Programming Assignment 3 — Poucos Rótulos: Fauna em Câmeras Trap com Dados Não Rotulados

**Disciplina:** Aprendizado Profundo  
**Professor:** Dario Oliveira  
**Monitor:** Erick Brito  
**Integrantes:** Henrique Gabriel Gasparelo & José Thevez Gomes Guedes  

---

## 1. O Problema e Desafios

Câmeras trap capturam imagens da fauna silvestre em condições remotas e desafiadoras. No monitoramento real, há abundância de dados não rotulados e extrema escassez de anotações humanas qualificadas.

Principais desafios do dataset **ENA24-detection** (21 classes):
1. **Razão de Ocupação:** Em ~25% das capturas, o animal ocupa menos de 2% da área do quadro.
2. **Ambiente e Iluminação:** Muitas capturas são noturnas, em infravermelho (monocromáticas/ruidosas).
3. **Atalhos Espúrios de Fundo:** A mesma câmera fotografa repetidamente o mesmo cenário, induzindo a rede a aprender o cenário em vez do espécime.
4. **Desbalanceamento Severo:** Classes abundantes (ex: urso com 690 imagens) vs. classes raras (ex: cavalo com 45 imagens no treino).
5. **Métrica Principal:** **F1 Macro** (pondera igualmente classes raras e frequentes; a acurácia isolada mascara o desempenho em classes minoritárias).

---

## 2. Regras de Engenharia Obrigatórias

* **Teto de Parâmetros:** Máximo de **12 milhões de parâmetros** (`sum(p.numel() for p in model.parameters()) <= 12_000_000`). Arquiteturas ideais: ResNet-18 (~11.7M), EfficientNet-B0 (~5.3M) ou MobileNetV3.
* **Ambiente de Inferência:** Execução estritamente em **CPU**, sem internet, classificando o conjunto `test_private` em menos de 10 minutos.
* **Orçamento Principal:** 5 rótulos por classe (105 imagens no total). Orçamentos de 1 e 20 rótulos são explorados na curva da Parte 2.
* **Proibições:** 
  * Não rotular imagens manualmente.
  * Proibido uso de APIs comerciais fechadas (OpenAI, Gemini API, Claude) para anotação/classificação.
  * Proibido baixar o dataset ENA24 original ou fontes externas não autorizadas.
  * Não realizar treino ou sintonia de hiperparâmetros/limiares sobre o `test_public`.

---

## 3. Planejamento da Estrutura de Arquivos em Árvore

A estrutura do repositório foi planejada seguindo o padrão modular, reprodutível e robusto consolidado no PAA2:

```text
PAA3/
├── checkpoints/                        # Pesos salvos dos modelos treinados
│   ├── .gitkeep
│   └── best_model.pt                   # [Planejado] Pesos do modelo final entregue (<= 12M params)
├── data/                               # Dados do ENA24 (descompactados do pa3_dados.zip)
│   ├── .gitkeep
│   ├── classes.csv                     # Mapeamento oficial de 21 classes
│   ├── rotulos_1.csv                   # Orçamento de 1 rótulo por classe (21 imagens)
│   ├── rotulos_5.csv                   # Orçamento padrão de 5 rótulos por classe (105 imagens)
│   ├── rotulos_20.csv                  # Orçamento de 20 rótulos por classe (420 imagens)
│   ├── val.csv                         # Validação (202 imagens, seleção de modelo)
│   ├── test_public.csv                 # Teste público para reporte (811 imagens)
│   └── images/                         # Imagens brutas (512px no maior lado)
├── docs/                               # Enunciado, documentação e figuras da apresentação
│   ├── PA3.pdf                         # Enunciado oficial
│   └── figures/                        # [Planejado] Gráficos de saliência e curvas para os slides
├── results/                            # Saídas consolidadas e artefatos de avaliação
│   ├── .gitkeep
│   └── curva_orcamento.csv             # [Planejado] Tabela oficial da Parte 2 (método, orçamento, seed, F1, acurácia)
├── src/                                # Código-fonte modular reutilizável
│   ├── __init__.py
│   ├── data.py                         # [Planejado] DataLoaders, stratificação e aumentações (Albumentations)
│   ├── models.py                       # [Planejado] Backbones (ResNet-18, EfficientNet-B0), verificação de <=12M params
│   ├── training.py                     # [Planejado] Loop de treino supervisionado, early stopping no val e scheduler
│   ├── unlabeled.py                    # [Planejado] Implementação da trilha de dados não rotulados (A, B ou C)
│   ├── metrics.py                      # [Planejado] F1 Macro, acurácia, matriz de confusão e métricas por classe
│   ├── saliency.py                     # [Planejado] Grad-CAM / Mapas de oclusão para galeria de falhas e atalhos
│   └── utils.py                        # [Planejado] Fixação de seeds, gerenciamento de checkpoints e logs
├── main.ipynb                          # Notebook principal de condução e experimentação (Partes 0 a 3)
├── predict.py                          # [Planejado] Script de inferência em CPU exigido para avaliação no test_private
├── requirements.txt                    # Dependências do projeto
├── AI_LOG.md                           # Relatório de utilização e decisões assistidas por IA
├── .gitignore                          # Regras de exclusão de artefatos temporários e dados brutos
└── README.md                           # Documentação central do projeto
```

---

## 4. Planejamento das Etapas e Experimentos

### Parte 0 — O Piso (Baseline Supervisionado)
* **Objetivo:** Estabelecer a referência inferior de desempenho treinando apenas com os 105 exemplos de `rotulos_5.csv`.
* **Versões:**
  * *(a)* CNN pequena treinada do zero.
  * *(b)* Backbone pré-treinado no ImageNet (ex: ResNet-18 ou EfficientNet-B0) com fine-tuning cuidadoso.
* **Métricas:** F1 Macro e acurácia no `test_public`, matriz de confusão da versão *(b)* e curvas de treino/validação identificando o início de memorização (*overfitting* severo pelo número diminuto de amostras).
* **Teto de Referência:** F1 Macro de **0,686** (ResNet-18 treinada com todos os 6.505 exemplos do `train`).

### Parte 1 — Explorando os Dados Não Rotulados (6.505 imagens)
Escolha e implementação de uma das três trilhas com o mesmo backbone:
* **Trilha A (Auto-supervisionado):** Continuação do pré-treino no domínio com SimCLR/SwAV sobre as 6.505 imagens + avaliação linear e fine-tuning com 105 rótulos. Investigação sobre o impacto de *RandomResizedCrop* em imagens com animais pequenos (< 2% do quadro).
* **Trilha B (Semi-supervisionado):** FixMatch utilizando os 105 rótulos e as 6.505 imagens não rotuladas, com estudo de sensibilidade do limiar $\tau$ e correção de viés para as classes raras.
* **Trilha C (Professor & Aluno):** Destilação de conhecimento a partir de um *foundation model* aberto (ex: BioCLIP/CLIP) gerando pseudo-rótulos para treinar o aluno leve (<= 12M).

### Parte 2 — A Curva de Orçamento
* Treinamento do baseline e do método escolhido nos orçamentos de **1, 5 e 20 rótulos por classe**.
* Execução em **3 seeds** no orçamento de 5 rótulos para reporte de média $\pm$ desvio padrão.
* Gráfico semi-log: F1 Macro no `test_public` vs. número de rótulos com a linha do teto ($0,686$).
* Geração do arquivo `results/curva_orcamento.csv` e análise de ganhos por classe.

### Parte 3 — Galeria de Falhas e Atalhos (Saliência e Grad-CAM)
* **3 Falhas:** Imagem, rótulo real, classe predita, confiança, mapa de Grad-CAM/oclusão e diagnóstico de atalho de fundo vs. escala do espécime.
* **5 Acertos:** Análise se a decisão correta ocorreu em virtude das feições do animal ou do contexto do cenário.

---

## 5. Download e Preparação dos Dados

Para baixar o conjunto de dados oficial do PA3:

1. Baixe o pacote compactado pelo link oficial do LILA BC / Google Drive:
   * **Arquivo:** `pa3_dados.zip` (~415 MB)
   * **URL:** [Google Drive - pa3_dados.zip](https://drive.google.com/file/d/1g7FGqtXOk_ziNKBwQAk8BxjPOaPCnbui/view?usp=sharing)
2. Descompacte os arquivos na pasta `data/`:
   ```bash
   unzip pa3_dados.zip -d data/
   ```
3. A pasta `data/` deverá conter as imagens e os arquivos CSV (`rotulos_1.csv`, `rotulos_5.csv`, `rotulos_20.csv`, `val.csv`, `test_public.csv`, `classes.csv`).

---

## 6. Configuração do Ambiente

```bash
# Criação do ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalação das dependências
pip install -r requirements.txt
```

---

## 7. Como Executar

* **Pipeline Interativo:** Abrir e executar `main.ipynb` para inspeção visual, gráficos de saliência, treinamento e geração dos artefatos da apresentação.
* **Script de Inferência Oficial (`predict.py`):**
  ```bash
  python predict.py --imagens data/images --saida preds.csv
  ```
  *(O script roda em CPU, sem conexão de rede, exibe a contagem de parâmetros e gera o CSV formatado com `image_id,label`)*.
