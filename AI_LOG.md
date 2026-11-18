# Registro de Uso de Inteligência Artificial (AI_LOG.md)

**Disciplina:** Aprendizado Profundo  
**Projeto:** Programming Assignment 3 — Poucos Rótulos: Fauna em Câmeras Trap com Dados Não Rotulados  
**Professor:** Dario Oliveira  
**Monitor:** Erick Brito  

---

## 1. Ferramenta Utilizada
* **Antigravity (Google DeepMind / Gemini):** Utilizado como assistente de *pair programming* para diagnóstico de problemas experimentais específicos, desenho de funções em PyTorch e depuração de métricas e visualizações.

---

## 2. Cinco Casos Específicos de Uso no Projeto

### Caso 1: Controle de Memorização Precoce no *Fine-Tuning* com 105 Imagens (`src/training.py`)
* **Problema Específico:** Ao treinar a `ResNet-18` (pré-treinada na ImageNet) apenas nos 105 exemplos de `rotulos_5.csv` com uma taxa de aprendizado única (`1e-3`), os gradientes grandes da camada linear `fc` (inicializada aleatoriamente para as 21 classes) degradavam os filtros convolucionais nas primeiras épocas e faziam a rede decorar o treino imediatamente com baixo F1 no conjunto `val`.
* **Como recorremos à IA:** Consultamos a IA para configurar taxas de aprendizado diferenciadas (*differential learning rates*) por grupo de parâmetros no `AdamW` e instrumentar um detector automático da época exata de início de memorização.
* **Solução Implementada:** Separamos os parâmetros em `src/training.py`, aplicando `lr=5e-4` apenas na cabeça `model.fc` e um multiplicador reduzido (`backbone_lr_mult=0.2`, isto é, `1e-4`) nas camadas convolucionais, combinado com `label_smoothing=0.05` e `CosineAnnealingLR`. Isso elevou o Piso Supervisionado para **`0,4651` de F1 Macro** no `test_public` (contra `0,1536` da `SmallCNN` do zero) e permitiu medir em `src/metrics.py` que a memorização do treino (>90% F1) inicia na **Época 5**, enquanto a *val loss* atinge o mínimo na **Época 12**.

### Caso 2: Correção do Colapso de *Prior* do Professor BioCLIP (*Zero-Shot* Enviesado) (`src/teacher.py`)
* **Problema Específico:** Os pseudo-rótulos brutos do BioCLIP (`imageomics/bioclip`) apresentavam um viés severo de distribuição: no `test_public`, o modelo previa `red_fox` 154 vezes (para apenas 43 reais) e quase zerava classes como `eastern_chipmunk` (1 predição), `grey_fox` (2) e `horse` (0). Como consequência, o Aluno C1 treinado nos pseudo-rótulos brutos atingiu apenas `0,4666` de F1 Macro (praticamente empatado com o Piso).
* **Como recorremos à IA:** Pedimos à IA alternativas matemáticas para desbiasar a distribuição marginal de pseudo-rótulos de modelos CLIP sem usar rótulos do conjunto `val`.
* **Solução Implementada:** Implementamos em `src/teacher.py` o alinhamento marginal de distribuição (*prior debiasing*, dividindo a probabilidade predita de cada classe pela sua frequência média elevada a $\tau_{debias}=0,65$), seguido de um filtro de confiança ($\tau=0,35$) com teto balanceado de no máximo 35 pseudo-rótulos por espécie e destilação por Divergência KL com temperatura $T=2,0$. Com esse tratamento, o Aluno C2 saltou para **`0,5371` / `0,5406` de F1 Macro** e superou o Professor *Zero-Shot* em 19 das 21 classes.

### Caso 3: Inferência Automática de Fotos Noturnas (Infravermelho) sem Metadados (`src/teacher.py`)
* **Problema Específico:** O enunciado exigia diagnosticar se o professor errava de forma sistemática à noite (em infravermelho), porém a faixa inferior com data/hora das câmeras havia sido cortada de todas as imagens do pacote `pa3_dados.zip`.
* **Como recorremos à IA:** Recorremos à IA para formular uma heurística determinística baseada nos canais de cor da imagem que separasse capturas diurnas de capturas noturnas em câmeras trap.
* **Solução Implementada:** Como o flash infravermelho noturno produz imagens em escala de cinza (canais $R \approx G \approx B$), implementamos em `evaluate_teacher_zero_shot` (`src/teacher.py`) o cálculo da diferença absoluta média entre canais ($\frac{1}{3}(|R-G| + |G-B| + |R-B|) < 3,0$). Isso permitiu identificar automaticamente que **48,2% das imagens de teste são noturnas** e comprovar que o F1 Macro do BioCLIP cai de **`0,3503` (diurno colorido)** para **`0,2249` (noturno IR)**.

### Caso 4: Regra Adaptativa de Destilação na Curva de Orçamento ($B=1, 5$ vs. $B=20$) (`src/experiments.py`)
* **Problema Específico:** Ao rodar a curva de orçamento da Parte 2, observamos que ajustar um classificador linear + protótipos sobre os embeddings do BioCLIP (Professor Melhorado) gerava ótimos pseudo-rótulos com 20 rótulos por classe (420 imagens, F1 do professor = `0,6186`), mas sobreajustava quando havia apenas 1 rótulo por classe (21 imagens).
* **Como recorremos à IA:** Discutimos com a IA como estruturar o pipeline `run_budget_curve` para evitar que a alta variância de um classificador linear com 1 exemplo por classe degradasse os pseudo-rótulos nos orçamentos pequenos.
* **Solução Implementada:** Em `src/experiments.py`, configuramos a destilação do Aluno para usar as probabilidades *zero-shot* com alinhamento de distribuição (`zs_debiased_probs_train`, `kd_alpha=0.5`, $T=2,0$) nos regimes de escassez extrema ($B \le 5$), e chavear para as probabilidades do Professor Melhorado adaptado no domínio (`probs_train`, `kd_alpha=0.35`, $T=1,8$) em $B=20$, alcançando **`0,3797`** em $B=1$ (+125% sobre o Piso) e **`0,6399`** em $B=20$.

### Caso 5: Auditoria de Atalhos de Fundo via Grad-CAM Puro em PyTorch (`src/visualization.py`)
* **Problema Específico:** Na Parte 3, precisávamos extrair mapas de ativação espacial da `ResNet-18` para verificar se predições corretas de alta confiança focavam no animal ou no cenário estático da câmera, sem adicionar dependências externas incompatíveis.
* **Como recorremos à IA:** Solicitamos à IA a implementação enxuta do algoritmo Grad-CAM diretamente sobre os tensores e *hooks* nativos do PyTorch.
* **Solução Implementada:** Implementamos a classe `_GradCAMExtractor` em `src/visualization.py` acoplando `register_forward_hook` e `register_full_backward_hook` ao bloco `model.layer4` (mapa $7\times 7$ anterior ao *Global Average Pooling*), ponderando os canais pelos gradientes médios do logit predito. A visualização revelou casos concretos de atalho de fundo tanto nos erros — como a galinha `c6660a8c2a1ace3b` ignorada enquanto a rede olhava para a estrutura de madeira e lona azul ao fundo — quanto nos acertos — como o corvo `f1ac43bb420fc5b1` classificado com `92,1%` de confiança com o mapa de calor focado na clareira e linha de árvores da câmera.
