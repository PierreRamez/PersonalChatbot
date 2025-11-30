# 🐍 Ouro: The Self-Healing Chatbot

*A personalized conversational AI that learns from its mistakes in real-time.*

---

## 🚀 Overview

**Ouro** is not just another static chatbot. It is a **Closed-Loop Active Learning System** designed to solve model stagnation.  
Most chatbots are "frozen" after training. **Ouro** is different: it features a **Human-in-the-Loop (HITL)** pipeline that allows users to correct the bot during conversation.  
All corrections are automatically captured, validated, and used to fine-tune the model's adapter weights, creating a system that continuously evolves to the user's specific needs.

---

### ✨ Key Features

* **🧠 Dynamic Intelligence:** Fine-tuned **Llama 3.2 3B** model optimized for edge deployment.  
* **🔄 Self-Healing Pipeline:** Automated workflow to ingest feedback, retrain LoRA adapters, and hot-swap weights without downtime.  
* **⚡ 4-Bit Quantization:** Efficient on consumer hardware (CPU/T4 GPU) using **Unsloth**.  
* **🛠️ MLOps Maturity:** Full experiment tracking via **MLflow**, artifacts on **Hugging Face**, and CI/CD with **GitHub Actions**.  

---

## 🏗️ System Architecture

**Ouro** operates on a decoupled microservices architecture to ensure scalability and resilience.

```mermaid
graph TD
    User[User] -->|Chat| UI[Streamlit Frontend]
    UI -->|Request| API[FastAPI Backend]
    API -->|Inference| Model[Llama 3.2 3B]

    User -->|Correction| UI
    UI -->|Feedback| DB[(feedback.jsonl)]

    subgraph "Continuous Learning Loop"
        DB -->|Data Prep| Pipeline[Training Pipeline]
        Pipeline -->|Fine-Tune| LoRA[LoRA Adapters]
        LoRA -->|Hot Reload| Model
    end
````

---

## 🛠️ Tech Stack

| Component     | Technology              | Description                                |
| ------------- | ----------------------- | ------------------------------------------ |
| **Model**     | Llama 3.2 3B            | Optimized instruction-tuned base model.    |
| **Training**  | Unsloth                 | 2× faster training, 60% less memory usage. |
| **Inference** | FastAPI                 | High-performance async backend.            |
| **Interface** | Streamlit               | Interactive chat UI with feedback widgets. |
| **Tracking**  | MLflow / DagsHub        | Experiment logging and metrics tracking.   |
| **DevOps**    | Docker & GitHub Actions | Containerization and CI/CD workflows.      |

---

## 🚀 Quick Start

### Prerequisites

* Python 3.10+
* Hugging Face Token (with write access)
* (Optional) GPU for faster inference

### 1. Clone the Repository

```bash
git clone https://github.com/PierreRamez/PersonalChatbot.git
cd PersonalChatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run with Docker (Recommended)

```bash
# Build Docker image
docker build -t ouro-chatbot .

# Run the container
docker run -p 7860:7860 ouro-chatbot
```

### 4. Local Development

```bash
# Start backend server
uvicorn app:app --reload --port 7860
```

Visit [http://localhost:7860/docs](http://localhost:7860/docs) to explore the Swagger API documentation.

---

## 📊 Performance & Evaluation

**Ouro** was evaluated using a **Sliding Window** strategy to measure conversational coherence across multi-turn dialogues.

| Metric             | Score  | Description                                      |
| ------------------ | ------ | ------------------------------------------------ |
| **Perplexity**     | 4.08   | High predictive confidence.                      |
| **BERTScore F1**   | 0.88   | Strong semantic alignment with human references. |
| **Inference Time** | ~150ms | Latency per token on T4 GPU.                     |

> **Note:** The 3B model was chosen over the 8B variant to allow CPU-only deployment on Hugging Face Spaces Free *Tier*.

---

## 📂 Project Structure

```
.
├── data/                     # Raw and processed datasets
├── finetune_prep/            # Scripts for preparing training batches
├── app.py                    # FastAPI backend entrypoint
├── test_app.py               # Unit tests (Pytest)
├── Dockerfile                # Container configuration
├── learning_pipeline.ipynb   # HITL training orchestrator
├── 3b_model.ipynb            # Base model fine-tuning notebook
└── main.yml                  # CI/CD workflow
```

---

## 🤝 Contributing

This project is part of the **Digital Egypt Pioneers Initiative (DEPI)**. While primarily for educational assessment, we welcome issues and feature requests.

1. Fork the repository
2. Create your feature branch:

   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes:

   ```bash
   git commit -m "Add some AmazingFeature"
   ```
4. Push to your branch:

   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a pull request

---

## 📜 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

<div align="center">
  <p>Built with ❤️</b></p>
  <p><i>Part of the DEPI Generative AI Track - Round 3</i></p>
</div>

