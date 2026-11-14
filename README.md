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

* **Teto de Parâmetros:** Máximo de **12 milhões de parâmetros** (`sum(p.numel() for p in model.parameters()) <= 12_000_000`). Backbone principal adotado: **ResNet-18** (`11.187.285` parâmetros).
* **Ambiente de Inferência:** Execução estritamente em **CPU**, sem internet, classificando o conjunto `test_private` em menos de 10 minutos.
* **Orçamento Principal:** 5 rótulos por classe (105 imagens no total). Orçamentos de 1 e 20 rótulos são explorados na curva da Parte 2.
* **Proibições:** 
  * Não rotular imagens manualmente.
  * Proibido uso de APIs comerciais fechadas (OpenAI, Gemini API, Claude) para anotação/classificação.
  * Proibido baixar o dataset ENA24 original ou fontes externas não autorizadas.
  * Não realizar treino ou sintonia de hiperparâmetros/limiares sobre o `test_public`.

---

## 3. Estrutura de Arquivos em Árvore

```text
PAA3/
├── checkpoints/                        # Pesos salvos dos modelos treinados
│   ├── part0a_small_cnn_b5_s42.pt      # Pesos da Parte 0 (a): SmallCNN treinada do zero
│   ├── part0b_resnet18_b5_s42.pt       # Pesos da Parte 0 (b): ResNet-18 (ImageNet Fine-Tuning)
│   └── best_model.pt                   # [Planejado] Pesos do modelo final entregue (<= 12M params)
├── data/                               # Dados do ENA24 (descompactados de pa3_dados.zip)
│   ├── LEIAME.md                       # Informações oficiais do pacote de dados
│   ├── classes.csv                     # Mapeamento oficial de 21 classes (idx, label)
│   ├── labels/                         # Anotações em CSV (image_id, label)
│   │   ├── rotulos_1.csv               # Orçamento de 1 rótulo por classe (21 imagens)
│   │   ├── rotulos_5.csv               # Orçamento padrão de 5 rótulos por classe (105 imagens)
│   │   ├── rotulos_20.csv              # Orçamento de 20 rótulos por classe (420 imagens)
│   │   ├── val.csv                     # Validação (202 imagens, até 10 por classe)
│   │   └── test_public.csv             # Teste público para reporte (811 imagens)
│   └── images/                         # Imagens brutas (512px no maior lado)
│       ├── train/                      # 6.505 imagens do pool de treino
│       ├── val/                        # 202 imagens do conjunto de validação
│       └── test_public/                # 811 imagens do teste público
├── docs/                               # Enunciado, documentação e figuras da apresentação
│   ├── PA3.pdf                         # Enunciado oficial
│   └── figures/                        # Gráficos e figuras geradas
│       ├── part0_curves_resnet18.png   # Curva de treino e validação da Parte 0 (b)
│       ├── part0_comparison_a_vs_b.png # Comparativo Parte 0 (a) vs Parte 0 (b)
│       └── part0_confusion_matrix_resnet18.png # Matriz de confusão da Parte 0 (b) no test_public
├── results/                            # Saídas consolidadas e artefatos de avaliação
│   ├── part0_summary.json              # Resumo de métricas e diagnóstico de memorização (Parte 0)
│   ├── part0_per_class_f1.csv          # F1 por classe no test_public (Parte 0a vs 0b)
│   └── curva_orcamento.csv             # [Planejado] Tabela oficial da Parte 2 (método, orçamento, seed, F1, acurácia)
├── src/                                # Código-fonte modular reutilizável
│   ├── __init__.py
│   ├── data.py                         # Dataset, mapeamento de classes e DataLoaders
│   ├── models.py                       # SmallCNN e ResNet-18 com verificação estrita de <= 12M params
│   ├── training.py                     # Loop de treino supervisionado com seleção de época pelo val F1 Macro
│   ├── metrics.py                      # F1 Macro, acurácia, matriz de confusão e curvas de overfitting
│   ├── part0.py                        # Pipeline completo e reprodutível da Parte 0 (O Piso)
│   ├── unlabeled.py                    # [Planejado] Implementação da Trilha C (Professor & Aluno)
│   ├── saliency.py                     # [Planejado] Grad-CAM / Mapas de oclusão para galeria de falhas e atalhos
│   └── utils.py                        # Fixação de seeds, gerenciamento de checkpoints e utilitários
├── main.ipynb                          # Notebook principal de condução e experimentação (Partes 0 a 3)
├── predict.py                          # [Planejado] Script de inferência em CPU exigido para avaliação no test_private
├── requirements.txt                    # Dependências do projeto
├── AI_LOG.md                           # Relatório de utilização e decisões assistidas por IA
├── .gitignore                          # Regras de exclusão de artefatos temporários e dados brutos
└── README.md                           # Documentação central do projeto
```

---

## 4. Etapas, Experimentos e Resultados

### Parte 0 — O Piso (Baseline Supervisionado — Concluída)
Treinamento supervisionado exclusivamente com os **105 exemplos** de `rotulos_5.csv` (5 rótulos por classe), selecionando a melhor época pelo **F1 Macro no `val` (202 imagens)**:

| Modelo / Versão | Parâmetros | Melhor Época (`val`) | Val F1 Macro | Test Public F1 Macro | Test Public Acurácia |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Parte 0 (a)** — `SmallCNN` (Treinada do Zero) | `424.501` | 40 | `0,1929` | **`0,1536`** | `0,1665` (16,65%) |
| **Parte 0 (b)** — `ResNet-18` (ImageNet Fine-Tuning) | `11.187.285` | 21 | `0,4246` | **`0,4651`** | `0,5068` (50,68%) |
| **Teto de Referência** — `ResNet-18` (6.505 rótulos) | `11.187.285` | - | - | **`0,6860`** | - |

* **Diagnóstico de Memorização (*Overfitting*) na Versão (b):** Com apenas 105 imagens, a `ResNet-18` começa a decorar o conjunto de treino já na **Época 5** (quando ultrapassa `90%` de F1 e acurácia no treino, atingindo `97%–99%` entre as épocas 7 e 10). A *Loss* de validação atinge seu mínimo na **Época 12** (`2,2021`) e o melhor F1 Macro de validação ocorre na **Época 21** (`0,4246`).
* **Análise por Classe (Matriz de Confusão):** Espécies bem representadas na ImageNet alcançam alto F1 mesmo com 5 rótulos (`eastern_chipmunk`: `0,967`; `american_crow`: `0,796`; `virginia_opossum`: `0,796`; `dog`: `0,760`), enquanto `bird` (`0,000`, confundido com `chicken` e `eastern_gray_squirrel`) e `eastern_cottontail` (`0,000`, confundido com `coyote` em capturas noturnas) colapsam.

### Parte 1 — Usando o que Não Tem Rótulo: Trilha C (Professor e Aluno)
* **Trilha Escolhida — Trilha C (Professor & Aluno):** Utilização de um *foundation model* aberto (ex: BioCLIP / SigLIP / CLIP) como **Professor** para rotular as 6.505 imagens de `train`, avaliando seu F1 *zero-shot* no `test_public`, diagnosticando sua matriz de confusão (ruído sistemático dos pseudo-rótulos) e aplicando tratamento de ruído (filtragem por confiança / temperatura / combinação com `rotulos_5.csv`) para treinar o modelo **Aluno (`ResNet-18`, <= 12M parâmetros)**.

### Parte 2 — A Curva de Orçamento
* Treinamento do piso (`0b`) e do método da Trilha C nos orçamentos de **1, 5 e 20 rótulos por classe**.
* Execução em **3 seeds** no orçamento de 5 rótulos para reporte de média $\pm$ desvio padrão.
* Gráfico semi-log: F1 Macro no `test_public` vs. número de rótulos com a linha do teto ($0,686$) e do Professor.
* Geração do arquivo `results/curva_orcamento.csv` e análise de ganhos por classe.

### Parte 3 — Galeria de Falhas e Atalhos (Saliência e Grad-CAM)
* **3 Falhas:** Imagem, rótulo real, classe predita, confiança, mapa de Grad-CAM/oclusão e diagnóstico de atalho de fundo vs. escala do espécime.
* **5 Acertos:** Análise se a decisão correta ocorreu em virtude das feições do animal ou do contexto do cenário.

---

## 5. Configuração do Ambiente e Execução

```bash
# 1. Criação e ativação do ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# 2. Instalação das dependências
pip install -r requirements.txt

# 3. Comando para treinar e avaliar a Parte 0 (O Piso)
python -m src.part0

# 4. Script de Inferência Oficial em CPU (Parte final)
python predict.py --imagens data/images/test_public --saida preds.csv
```
