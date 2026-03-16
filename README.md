# Fine-Tuning BERT para Classificação de Atendimento ao Cliente

Projeto desenvolvido como parte do desafio de Fine-Tuning da **Rocketseat**, onde o objetivo é treinar um modelo de linguagem para classificar automaticamente mensagens de clientes de uma rede varejista de produtos domésticos.

## Objetivo

Realizar o fine-tuning do modelo **BERTimbau** (`neuralmind/bert-base-portuguese-cased`) para classificação binária de mensagens em:

| Categoria | Descrição | Exemplo |
|-----------|-----------|---------|
| **venda** | Intenção de compra | *"Quero comprar um micro-ondas, vocês têm sugestões?"* |
| **suporte** | Dúvidas ou problemas | *"Meu ar-condicionado está com um ruído estranho, o que pode ser?"* |

## Modelo Utilizado

O **BERTimbau** foi escolhido por ser um BERT pré-treinado especificamente em textos em português brasileiro, oferecendo desempenho superior ao BERT multilingual para tarefas de NLP em PT-BR.

- **Modelo base:** `neuralmind/bert-base-portuguese-cased`
- **Parâmetros:** ~110M
- **Tipo:** Classificação de sequência (`SequenceClassification`)

## Estrutura do Projeto

```
├── finetuning_bert_classificacao.ipynb   # Notebook com o pipeline completo
├── treino.jsonl                          # Dataset de treino
├── teste.jsonl                           # Dataset de teste/validação
├── .gitignore
└── README.md
```

## Pipeline do Notebook

1. **Instalação de dependências** — Transformers, Datasets, PyTorch, scikit-learn
2. **Carregamento dos dados** — Leitura dos arquivos JSONL
3. **Análise exploratória** — Distribuição das classes
4. **Tokenização** — Tokenizer do BERTimbau com `max_length=64`
5. **Split de validação** — 80/20 estratificado
6. **Fine-tuning** — 5 épocas, lr=2e-5, weight decay, early stopping
7. **Avaliação** — Accuracy, F1-Score, Precision, Recall, Matriz de Confusão
8. **Inferência** — Teste com mensagens novas via `pipeline`

## Como Executar

### Google Colab (recomendado)

1. Abra o [Google Colab](https://colab.research.google.com/)
2. Faça upload do notebook `finetuning_bert_classificacao.ipynb`
3. Faça upload dos datasets `treino.jsonl` e `teste.jsonl`
4. Em **Runtime > Change runtime type**, selecione **GPU**
5. Execute todas as células na ordem

### Local

```bash
pip install transformers datasets torch scikit-learn accelerate
jupyter notebook finetuning_bert_classificacao.ipynb
```

## Dados

Os datasets estão no formato JSONL, com a seguinte estrutura:

```json
{"prompt": "Olá, gostaria de fazer a aquisição do novo produto", "completion": "venda"}
{"prompt": "Tudo bom, queria verificar como funciona a TV Smart x0912", "completion": "suporte"}
```

> **Nota:** Os dados foram gerados de forma sintética, exclusivamente para fins didáticos.

## Tecnologias

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- scikit-learn
- BERTimbau (neuralmind)

## Autor

**Levi Cury** — [GitHub](https://github.com/LeviCury)

---

Desenvolvido como desafio da trilha de AI da [Rocketseat](https://www.rocketseat.com.br/).
