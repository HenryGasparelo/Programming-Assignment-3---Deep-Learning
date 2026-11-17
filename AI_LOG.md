# Registro de Uso de Inteligência Artificial (AI_LOG.md)

**Disciplina:** Aprendizado Profundo  
**Projeto:** Programming Assignment 3 — Poucos Rótulos: Fauna em Câmeras Trap com Dados Não Rotulados  
**Professor:** Dario Oliveira  
**Monitor:** Erick Brito  

---

## 1. Ferramentas Utilizadas
* **Antigravity (Google DeepMind / Gemini):** Pair programming, planejamento da arquitetura de pastas, leitura e interpretação do enunciado oficial (`PA3.pdf`), implementação modular em `src/`, estruturação do notebook principal e documentação técnica.

---

## 2. Episódios de Uso e Tomada de Decisão

### Episódio 1: Leitura do Enunciado e Planejamento da Arquitetura
* **Contexto:** Inicialização do PAA3, analisando as lições aprendidas e o padrão de organização modular do PAA2.
* **Uso da IA:** A IA analisou na íntegra o `PA3.pdf`, mapeando regras de engenharia (limite estrito de 12M parâmetros, inferência rápida em CPU, uso de F1 macro para lidar com o desbalanceamento severo de classes, orçamentos 1/5/20 e entregável `predict.py`).
* **Resultado:** Criação da estrutura de pastas padronizada (`data/`, `checkpoints/`, `src/`), criação do `README.md` detalhado com a árvore do repositório e estruturação do notebook `main.ipynb` como condutor experimental completo.

### Episódio 2: Implementação e Diagnóstico da Parte 0 (O Piso Supervisionado)
* **Contexto:** Necessidade de estabelecer o piso supervisionado treinando apenas com os 105 exemplos de `rotulos_5.csv` (5 por classe) em duas versões: (a) CNN pequena do zero e (b) backbone `ResNet-18` partindo da ImageNet com fine-tuning, selecionando a época estritamente pelo conjunto `val`.
* **Uso da IA:** Estruturou os módulos `src/utils.py`, `src/data.py`, `src/models.py`, `src/metrics.py`, `src/training.py` e `src/experiments.py`. Implementou taxas de aprendizado diferenciadas para o fine-tuning da `ResNet-18` (`lr=5e-4` na camada linear `fc` e `lr=1e-4` nas camadas convolucionais pré-treinadas) e detecção automática do início de memorização (*overfitting*).
* **Resultado:** A `SmallCNN` treinada do zero (424,5k parâmetros) atingiu `0,1536` de F1 Macro no `test_public`, enquanto a `ResNet-18` com fine-tuning da ImageNet (11,19M parâmetros) atingiu **`0,4651` de F1 Macro** (`50,68%` de acurácia), mostrando que o modelo começa a decorar as 105 imagens já na **Época 5** (F1/acurácia de treino `> 90%`) e atinge o mínimo da loss de validação na **Época 12**.

### Episódio 3: Parte 1 — Trilha C (Professor BioCLIP e Aluno ResNet-18)
* **Contexto:** Alavancar as 6.505 imagens não rotuladas de `train` usando um foundation model aberto como professor sem violar as restrições de arquitetura ($\le$ 12M params no aluno final) e sem permitir vazamento do conjunto `val`.
* **Uso da IA:** Implementou o pipeline do professor em `src/teacher.py` e as ablações do aluno em `src/experiments.py`. Mapeou os embeddings do BioCLIP (`imageomics/bioclip`), avaliou o F1 zero-shot no `test_public` (`0,3017`), diagnosticou quantitativamente as falhas sistemáticas do professor no infravermelho e em animais de pequena escala, e desenhou três ablações de destilação para o Aluno `ResNet-18` (com filtro de confiança balanceado por classe e rótulos suaves com temperatura $T=2,0$).
* **Resultado:** O Aluno atingiu F1 Macro de **`0,5371` a `0,5406`** (Acurácia: `58,45%` a `58,57%`), superando o Professor Zero-Shot em 19 das 21 classes e o Piso Supervisionado em mais de +7 pontos percentuais.

### Episódio 4: Parte 2 — A Curva de Orçamento e Quebra de Ganho
* **Contexto:** Medir o desempenho do Piso e do Aluno Trilha C nos orçamentos de 1, 5 e 20 rótulos por classe, executando 3 seeds no orçamento padrão (5 rótulos) e gerando o entregável oficial `curva_orcamento.csv`.
* **Uso da IA:** Implementou `run_budget_curve` em `src/experiments.py` com gerenciamento automático de checkpoints para reuso eficiente, cálculo das médias $\pm$ desvios padrão para as seeds `42`, `123` e `999`, geração do gráfico semi-log com barras de erro (`plot_budget_curve`) e quebra horizontal do ganho líquido por classe (`plot_per_class_gain_budget5`).
* **Resultado:** No orçamento de 1 rótulo, o Aluno saltou de `0,1687` para **`0,3797`** (+125% relativo); no orçamento de 5 rótulos, o ganho médio foi de **`+0,0707` no F1 Macro** (`0,4493` para `0,5200`) e +8,10% em acurácia; no orçamento de 20 rótulos, o Aluno atingiu **`0,6399`**, aproximando-se do teto de 6.505 rótulos (`0,6860`). O arquivo oficial `curva_orcamento.csv` foi exportado na raiz do repositório.

### Episódio 5: Reestruturação da Arquitetura por Responsabilidade e Visualização Dinâmica
* **Contexto:** Necessidade de eliminar acoplamentos artificiais organizados por "partes", modularizar o código por domínios de responsabilidade estrita (`models`, `data`, `training`, `teacher`, `metrics`, `experiments`, `visualization`, `utils`) e garantir que o notebook `main.ipynb` gere todas as figuras e gráficos dinamicamente (sem apenas carregar imagens estáticas do disco).
* **Uso da IA:** Reestruturou o pacote `src/` em 8 módulos limpos sem código duplicado. Centralizou os pipelines de visualização (incluindo a galeria de Grad-CAM da Parte 3) em `src/visualization.py`, desacoplou métricas de renderização, e transformou o notebook `main.ipynb` em um condutor didático de alto nível onde cada célula chama funções modulares e gera gráficos ao vivo.
* **Resultado:** Código modular limpo, 100% testado de ponta a ponta com reaproveitamento de checkpoints e geração ao vivo de todos os gráficos obrigatórios da apresentação.

### Episódio 6: Implementação do `predict.py`, CLI de Treino/Avaliação e Verificação Final
* **Contexto:** Revisão de conformidade de todos os entregáveis exigidos na Seção 4 do enunciado (`PA3.pdf`).
* **Uso da IA:** Implementou o script `predict.py` operando estritamente em CPU e sem internet (`pretrained=False`), imprimindo o número de parâmetros do modelo (`11.187.285` <= `12.000.000`) e gerando o CSV com colunas `image_id,label` a partir de `data/classes.csv`. Adicionou a interface de linha de comando em `src/experiments.py` (`--treinar` e `--avaliar`), configurou `checkpoints/best_model.pt` e removeu a linha duplicada do Teto de Referência na função `build_part1_summary_table`.
* **Resultado:** Repositório completo e 100% aderente a todas as exigências técnicas e de entrega do `PA3.pdf`.
