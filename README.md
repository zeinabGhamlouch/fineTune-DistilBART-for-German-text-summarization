# Fine-Tuning DistilBART for German Text Summarization 🇩🇪

This project focuses on **fine-tuning a Small Language Model (SLM)** — specifically **DistilBART** — for the task of **German text summarization**.  
It explores efficient model adaptation using **LoRA (Low-Rank Adaptation)** and **BitsAndBytes 4-bit quantization**, balancing performance with limited computational resources.

---

## 📘 Project Overview

While Large Language Models (LLMs) achieve impressive summarization results, they are often computationally expensive.  
This project demonstrates how a smaller model like **DistilBART** can be fine-tuned effectively to handle **non-English (German)** text summarization tasks, using parameter-efficient methods.

We evaluate the model with **ROUGE**, **BERTScore**, and **LLM-as-a-Judge** to measure lexical, semantic, and qualitative performance.

---

## 🚀 Getting Started

### Requirements

- Python 3.8+
- PyTorch  
- Transformers (Hugging Face)  
- Datasets  
- PEFT (LoRA fine-tuning)  
- BitsAndBytes (quantization)  
- Scikit-learn  
- ROUGE-score  
- BERTScore  
- LangChain (for LLM-based evaluation)  
- tqdm  
- Other dependencies listed in `requirements.txt`

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/fine-tuning-DistilBART-for-German-text-summarization.git
   cd fine-tuning-DistilBART-for-German-text-summarization ```

2. **install dependencies**
   ```pip install -r requirements.txt```
   
3. **Download the dataset**
   Follow the instructions in data/README.md to download the German Political Speeches Corpus from Kaggle.

4. **Run the notebook**
   fine_tune_german_summarization.ipynb
   
---

## 🧩 Methodology

### 1. Dataset
- **Dataset:** [German Political Speeches Corpus](https://www.kaggle.com/datasets/mexwell/german-political-speeches-corpus)  
- **Size used:** 5,000 speeches (subset)  
- **Preprocessing:** Focused on the `rohtext` field (raw text only)  
- **Splits:**  
  - 70% Training  
  - 15% Validation  
  - 15% Testing  

### 2. Model & Fine-Tuning
- **Base Model:** [`sshleifer/distilbart-cnn-12-6`](https://huggingface.co/sshleifer/distilbart-cnn-12-6)
- **Libraries Used:**
  - 🤗 Hugging Face Transformers  
  - PEFT (for LoRA fine-tuning)  
  - BitsAndBytes (4-bit quantization)
- **Hyperparameters:**
  - Learning rate: `2e-5`  
  - Batch size: `8`  
  - Epochs: `6`  
  - Weight decay: `0.01`  
  - LoRA config: `r=8, alpha=16, dropout=0.05`

### 3. Evaluation
Metrics used:
- **ROUGE-1 / ROUGE-2 / ROUGE-L**
- **BERTScore (multilingual)**  
- **LLM-as-a-Judge** — Using *Llama 3.2* to rate:
  - Relevance  
  - Coherence  
  - Conciseness  
  - Fluency  

---

## 📊 Results Summary

| Metric | Score |
|--------|-------:|
| **ROUGE-1 F1** | 0.3335 |
| **ROUGE-2 F1** | 0.0828 |
| **ROUGE-L F1** | 0.1721 |
| **BERT F1** | 0.6752 |
| **LLM Coherence** | 0.5392 |
| **LLM Fluency** | 0.4343 |

**Interpretation:**  
The fine-tuned DistilBART model demonstrates strong *semantic understanding* (high BERTScore) but moderate *lexical overlap* (lower ROUGE-2).  
It achieves good coherence but could improve in relevance, conciseness, and fluency.

---

## 🧠 Key Takeaways
- **Small models** can perform well in multilingual summarization with proper fine-tuning.
- **LoRA + Quantization** enable training on free-tier hardware (e.g., Google Colab GPU).
- **Evaluation beyond ROUGE** (e.g., BERTScore, LLM-as-a-Judge) gives deeper insight into summary quality.


## 📚 References

1. **Mexwell.** “*German Political Speeches Corpus.*”  
   [https://www.kaggle.com/datasets/mexwell/german-political-speeches-corpus](https://www.kaggle.com/datasets/mexwell/german-political-speeches-corpus).  
   Retrieved March 3, 2025.

2. **A. Meta.** “*Llama 3.2: Revolutionizing Edge AI and Vision with Open, Customizable Models.*”  
   *Meta AI Blog,* September 2024. Retrieved March 10, 2025.

3. **F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer,  
   R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, E. Duchesnay, and G. Louppe.**  
   “*Scikit-learn: Machine Learning in Python.*”  
   *Journal of Machine Learning Research,* vol. 12, 2012.

4. **S. Shleifer.** “*sshleifer/distilbart-cnn-12-6.*”  
   [https://huggingface.co/sshleifer/distilbart-cnn-12-6](https://huggingface.co/sshleifer/distilbart-cnn-12-6).  
   Retrieved March 10, 2025.

5. **E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, and W. Chen.**  
   “*LoRA: Low-Rank Adaptation of Large Language Models.*”  
   *CoRR,* abs/2106.09685, 2021.

6. **C.-Y. Lin.** “*ROUGE: A Package for Automatic Evaluation of Summaries.*”  
   In *Text Summarization Branches Out,* Association for Computational Linguistics, Barcelona, Spain, 2004.

7. **T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, and Y. Artzi.**  
   “*BERTScore: Evaluating Text Generation with BERT.*” 2020.

