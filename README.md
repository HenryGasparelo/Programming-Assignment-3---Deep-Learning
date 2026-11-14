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
│   ├── part0b_resnet18_b1_s42.pt       # Piso Supervisionado B=1 (21 rótulos)
│   ├── part0b_resnet18_b5_s42.pt       # Piso Supervisionado B=5 (105 rótulos, seed 42)
│   ├── part0b_resnet18_b5_s123.pt      # Piso Supervisionado B=5 (seed 123)
│   ├── part0b_resnet18_b5_s999.pt      # Piso Supervisionado B=5 (seed 999)
│   ├── part0b_resnet18_b20_s42.pt      # Piso Supervisionado B=20 (420 rótulos)
│   ├── part2_student_trilha_c_b1_s42.pt   # Aluno Trilha C B=1 (21 rótulos)
│   ├── part2_student_trilha_c_b5_s42.pt   # Aluno Trilha C B=5 (seed 42)
│   ├── part2_student_trilha_c_b5_s123.pt  # Aluno Trilha C B=5 (seed 123)
│   ├── part2_student_trilha_c_b5_s999.pt  # Aluno Trilha C B=5 (seed 999)
│   ├── part2_student_trilha_c_b20_s42.pt  # Aluno Trilha C B=20 (420 rótulos)
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
│       ├── part0_confusion_matrix_resnet18.png # Matriz de confusão da Parte 0 (b) no test_public
│       ├── part1_teacher_zeroshot_confusion_matrix.png # Matriz de ruído do Professor BioCLIP Zero-Shot
│       ├── part1_teacher_noise_diagnostics.png         # Diagnóstico sistemático de ruído (dia vs. noite/IR)
│       ├── part1_student_c3_curves.png                 # Curvas de treino/val do Aluno
│       ├── part1_student_c3_confusion_matrix.png       # Matriz de confusão do Aluno no test_public
│       ├── part1_student_vs_teacher_per_class.png      # Comparativo por classe: Aluno vs. Professor vs. Piso
│       ├── part2_curva_orcamento.png                   # Curva de orçamento semi-log (Piso vs. Aluno vs. Teto)
│       └── part2_ganho_por_classe_b5.png               # Quebra do ganho por classe no orçamento de 5 rótulos
├── results/                            # Saídas consolidadas e artefatos de avaliação
│   ├── part0_summary.json              # Resumo de métricas e diagnóstico de memorização (Parte 0)
│   ├── part0_per_class_f1.csv          # F1 por classe no test_public (Parte 0a vs 0b)
│   ├── part1_summary.json              # Resumo das ablações da Parte 1
│   ├── part1_summary_table.csv         # Tabela consolidada da Parte 1 (Piso vs. Professor vs. Alunos)
│   ├── part1_per_class_comparison.csv  # F1 por classe: Piso vs. Professor vs. Aluno
│   ├── curva_orcamento.csv             # Tabela oficial da Parte 2 (método, orçamento, seed, F1, acurácia)
│   ├── part2_summary_table.csv         # Resumo agregado com médias e desvios padrão da Parte 2
│   └── part2_per_class_gain_b5.csv     # Quebra detalhada de ganho por classe no orçamento 5 (3 seeds)
├── src/                                # Código-fonte modular separado por responsabilidade
│   ├── __init__.py                     # Interface pública consolidada do pacote
│   ├── data.py                         # Datasets (CameraTrapDataset, TeacherStudentDataset) e DataLoaders
│   ├── models.py                       # SmallCNN e ResNet-18 com verificação estrita de <= 12M params
│   ├── training.py                     # Loops de treino supervisionado e destilação de conhecimento
│   ├── metrics.py                      # Avaliação (F1 Macro, acurácia), diagnóstico de memorização e tabelas
│   ├── teacher.py                      # Foundation Model BioCLIP: zero-shot, pseudo-rótulos e tratamento de ruído
│   ├── experiments.py                  # Execução modular de experimentos (Piso, Ablações e Curva de Orçamento)
│   ├── visualization.py                # Gráficos dinâmicos (curvas, matrizes, ruído, orçamento semi-log)
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

### Parte 1 — Usando o que Não Tem Rótulo: Trilha C (Professor e Aluno — Concluída)
* **Professor BioCLIP Zero-Shot (`imageomics/bioclip` no TreeOfLife-10M):** F1 Macro = **`0,3017`** (Acurácia = `35,39%`).
* **Diagnóstico da Matriz de Ruído:** O professor colapsa em animais pequenos (`eastern_chipmunk`: `0,050`; `bird`: `0,000`) e sofre no infravermelho noturno (48,2% do teste), gerando confusão severa entre raposa vermelha e raposa cinzenta.
* **Ablações de Treinamento do Aluno (`ResNet-18` <= 12M params):**
  - **Aluno C1 (Sem Filtro):** Val F1 = `0,4297` | Test F1 = `0,4666`.
  - **Aluno C2 (Zero-Shot Filtrado $\tau=0,35$ + Alinhamento + Temp $T=2,0$):** Val F1 = **`0,4804`** | Test F1 = **`0,5371`** (Acurácia = `58,45%`).
  - **Aluno C3 (Prof. Adaptado + Filtro Balanceado + Temp $T=2,0$):** Val F1 = `0,4735` | Test F1 = **`0,5138`** (Acurácia = `57,34%`).
* **Aluno vs. Professor:** O Aluno supera o Professor Zero-Shot em **19 das 21 classes**, beneficiando-se da adaptação convolucional aos sensores de câmera trap e da ancoragem nos 105 rótulos limpos.

### Parte 2 — A Curva de Orçamento (Concluída)
Treinamento do Piso Supervisionado e do Aluno da Trilha C nos orçamentos de **1, 5 (3 seeds) e 20 rótulos por classe**, salvando a tabela entregável `curva_orcamento.csv`:

| Orçamento (por classe) | Total de Rótulos | Seeds | Piso F1 Macro (`test_public`) | Aluno Trilha C F1 Macro (`test_public`) | Ganho Líquido (ΔF1 Macro) | Piso Acurácia | Aluno Trilha C Acurácia |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 rótulo** | 21 | `42` | `0,1687` | **`0,3797`** | **`+0,2110` (+125% relativo)** | `21,82%` | `42,05%` |
| **5 rótulos** | 105 | `42, 123, 999` | `0,4493 ± 0,0195` | **`0,5200 ± 0,0189`** | **`+0,0707` (+8,1% acurácia)** | `48,91% ± 2,55%` | `57,01% ± 1,83%` |
| **20 rótulos** | 420 | `42` | `0,5984` | **`0,6399`** | **`+0,0415`** | `65,10%` | `68,56%` |
| **Teto de Referência** | 6.505 | - | - | **`0,6860`** | - | - | - |

* **Gráfico Semi-Log Oficial:** Gerado e salvo em `docs/figures/part2_curva_orcamento.png`.
* **Quebra do Ganho por Classe ($B=5$ rótulos, 3 seeds):**
  - **Maiores Ganhadores:** `wild_turkey` (+0,259), `striped_skunk` (+0,224), `white_tailed_deer` (+0,215), `woodchuck` (+0,207) e `coyote` (+0,166).
  - **Classes com Baixo Ganho / Variância:** `horse` (-0,164, apenas 4 imagens no teste público), `american_crow` (-0,136, ruído de folhas), `eastern_cottontail` (-0,080) e `eastern_chipmunk` (-0,016, efeito de teto já que o piso já atingia 96,7%).
* **Arquivo Entregável:** `curva_orcamento.csv` gerado na raiz e em `results/curva_orcamento.csv`.

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
