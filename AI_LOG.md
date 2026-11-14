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
* **Resultado:** Criação da estrutura de pastas padronizada (`data/`, `checkpoints/`, `src/`, `docs/`, `results/`), criação do `README.md` detalhado com a árvore planejada e estruturação do notebook `main.ipynb` como guia de desenvolvimento.

### Episódio 2: Implementação e Diagnóstico da Parte 0 (O Piso Supervisionado)
* **Contexto:** Necessidade de estabelecer o piso supervisionado treinando apenas com os 105 exemplos de `rotulos_5.csv` (5 por classe) em duas versões: (a) CNN pequena do zero e (b) backbone `ResNet-18` partindo da ImageNet com fine-tuning, selecionando a época estritamente pelo conjunto `val`.
* **Uso da IA:** Estruturou os módulos `src/utils.py`, `src/data.py`, `src/models.py`, `src/metrics.py`, `src/training.py` e `src/part0.py`. Implementou taxas de aprendizado diferenciadas para o fine-tuning da `ResNet-18` (`lr=5e-4` na camada linear `fc` e `lr=1e-4` nas camadas convolucionais pré-treinadas) e detecção automática do início de memorização (*overfitting*).
* **Resultado:** A `SmallCNN` treinada do zero (424,5k parâmetros) atingiu `0,1536` de F1 Macro no `test_public`, enquanto a `ResNet-18` com fine-tuning da ImageNet (11,19M parâmetros) atingiu **`0,4651` de F1 Macro** (`50,68%` de acurácia), mostrando que o modelo começa a decorar as 105 imagens já na **Época 5** (F1/acurácia de treino `> 90%`) e atinge o mínimo da loss de validação na **Época 12**.
