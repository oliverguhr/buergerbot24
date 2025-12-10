# 🤖 BürgerBot24 - RAG Evaluation Tutorial

A hands-on tutorial project for learning how to evaluate Retrieval-Augmented Generation (RAG) systems. This project demonstrates the complete workflow from creating evaluation datasets to measuring RAG performance using the [RAGAS](https://docs.ragas.io/) framework.

## 📚 What You'll Learn

- How to create synthetic Q&A datasets for RAG evaluation
- How to evaluate RAG system responses against ground truth
- Understanding key RAG evaluation metrics
- Working with local LLMs for cost-effective evaluation

## 🏗️ Project Overview

This project evaluates a German citizen services chatbot ("BürgerBot") that answers questions about administrative procedures, legal requirements, and public services in Germany.

### Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Documents   │───▶│  LLM generates│───▶│  Q&A Pairs   │      │
│  │  (Markdown)  │    │  Q&A pairs    │    │  (JSON)      │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│        │                                        │               │
│        │            create_eval_dataset.ipynb   │               │
│        ▼                                        ▼               │
│  ┌──────────────────────────────────────────────────────┐      │
│  │                  RAG System                           │      │
│  │  ┌─────────┐    ┌─────────────┐    ┌─────────────┐   │      │
│  │  │ Question│───▶│  Retriever  │───▶│  Generator  │   │      │
│  │  └─────────┘    │  (finds     │    │  (answers   │   │      │
│  │                 │   relevant  │    │   based on  │   │      │
│  │                 │   docs)     │    │   context)  │   │      │
│  │                 └─────────────┘    └─────────────┘   │      │
│  └──────────────────────────────────────────────────────┘      │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              RAGAS Evaluation                         │      │
│  │                                                       │      │
│  │   Response ◄──────────────────────────► Reference     │      │
│  │   (RAG output)        compare          (Ground Truth) │      │
│  │                          │                            │      │
│  │                          ▼                            │      │
│  │                    Score (0-1)                        │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                 │
│                    evaluate_rag.ipynb                           │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```text
buergerbot24/
├── docs/                          # Source documents (German admin topics)
│   ├── Adoption.md
│   ├── Arbeitslosigkeit.md
│   └── ... (400+ markdown files)
│
├── create_eval_dataset.ipynb      # Step 1: Generate Q&A pairs
├── evaluate_rag.ipynb             # Step 2: Evaluate RAG system
│
├── question_answer_pairs.json     # Generated evaluation dataset
├── filtered_results.html          # Low-scoring samples for analysis
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A local LLM server (e.g., [LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.ai/))
- A RAG system to evaluate (e.g., [Open WebUI](https://github.com/open-webui/open-webui))

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/buergerbot24.git
cd buergerbot24

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. **LLM Server**: Start your local LLM (default: `http://localhost:1234/v1`)
2. **RAG System**: Start your RAG system (default: `http://localhost:3000`)
3. **Update tokens/URLs** in the notebooks as needed

## 📓 Notebooks Walkthrough

### 1️⃣ `create_eval_dataset.ipynb` - Create Evaluation Data

This notebook generates question-answer pairs from your document corpus.

**What it does:**

1. Loads all markdown documents from `/docs`
2. For each document, asks an LLM to generate a relevant Q&A pair
3. Saves results to `question_answer_pairs.json`

**Key concept:** We use an LLM to create "ground truth" answers that serve as the reference for evaluation.

```python
# Example generated Q&A pair
{
  "question": "What documents are required for a German passport application?",
  "answer": "You need a valid ID, biometric photo, and proof of German citizenship.",
  "source_document": "Reisepass_beantragen.md"
}
```

### 2️⃣ `evaluate_rag.ipynb` - Evaluate RAG Performance

This notebook measures how well your RAG system answers questions.

**What it does:**

1. Loads Q&A pairs from JSON
2. Sends each question to the RAG system
3. Compares RAG responses against reference answers using RAGAS metrics
4. Identifies poorly performing samples for improvement

**Key concept:** We compare the RAG system's response against our ground truth to calculate accuracy scores.

## 📊 Understanding the Metrics

### Answer Accuracy (NVIDIA Metric)

The primary metric used in this project. It measures agreement between the RAG response and the reference answer.

**How it works:**

```text
┌────────────────────────────────────────────────────────────┐
│  LLM Judge 1: Compare Response → Reference             │
│  LLM Judge 2: Compare Reference → Response (swapped!)  │
│                                                        │
│  Final Score = Average of both judgments               │
│                                                        │
│  Scoring Scale:                                        │
│  • 0.0 = Completely wrong or irrelevant                │
│  • 0.5 = Partially correct                             │
│  • 1.0 = Fully correct and complete                    │
└────────────────────────────────────────────────────────┘
```

**Why dual judges?** Using two prompts with swapped roles makes the evaluation more robust and reduces LLM bias.

### Other RAGAS Metrics (for future exploration)

| Metric | What it measures | When to use |
|--------|-----------------|-------------|
| **Faithfulness** | Is the response grounded in retrieved context? | When you have access to retrieved documents |
| **Context Precision** | Are retrieved documents relevant? | To evaluate retrieval quality |
| **Context Recall** | Were all relevant documents retrieved? | To check retrieval completeness |
| **Factual Correctness** | Are individual claims in the response correct? | For detailed error analysis |

## 🔧 Customization

### Using Different Models

```python
# In evaluate_rag.ipynb
evaluator_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model="your-model-name",
        base_url="http://your-server:port/v1",
        api_key="your-api-key"
    )
)
```

### Adjusting Evaluation Thresholds

```python
# Filter samples with score below threshold
threshold = 0.5
filtered_df = df[df['nv_accuracy'] <= threshold]
```

### Adding More Metrics

```python
from ragas.metrics import AnswerAccuracy, FactualCorrectness

scorers = [
    AnswerAccuracy(llm=evaluator_llm),
    FactualCorrectness(llm=evaluator_llm)
]

results = evaluate(dataset, metrics=scorers)
```

## 📈 Interpreting Results

After running the evaluation, you'll get a DataFrame with scores for each sample:

| user_input | response | reference | nv_accuracy |
|------------|----------|-----------|-------------|
| "When was...?" | "Einstein was born..." | "Albert Einstein..." | 1.0 |
| "What is...?" | "The capital is..." | "Berlin is..." | 0.5 |

**Next steps based on results:**

- **Score < 0.5**: Investigate why the RAG system failed
- **Check retrieved documents**: Was the right information retrieved?
- **Improve prompts**: Adjust system prompts for better responses
- **Add documents**: Ensure knowledge base covers the topic

## 🔗 Resources

### RAG Evaluation

- [RAGAS Documentation](https://docs.ragas.io/)
- [RAGAS Metrics Overview](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)
- [DeepEval](https://docs.confident-ai.com/) - Alternative evaluation framework

### RAG Systems

- [Open WebUI](https://github.com/open-webui/open-webui) - Used in this project
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)

### Local LLMs

- [LM Studio](https://lmstudio.ai/) - Easy local LLM hosting
- [Ollama](https://ollama.ai/) - Run LLMs locally

## 🤝 Contributing

Contributions are welcome! Some ideas:

- Add more evaluation metrics
- Create visualizations for results
- Add retrieval context evaluation
- Translate documentation

## 📄 License

MIT License - feel free to use this for learning and projects.

---

Happy evaluating! 🚀

If you have questions, open an issue or check out the [RAGAS Discord](https://discord.gg/ragas).
