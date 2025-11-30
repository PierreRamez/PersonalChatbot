# **🐍 Ouro: The Self-Healing Chatbot**

A personalized conversational AI that learns from its mistakes in real-time.

## **🚀 Overview**

**Ouro** isn't just another static chatbot. It is a **Closed-Loop Active Learning System** designed to solve the problem of model stagnation.  
Most chatbots are "frozen" after training. Ouro is different. It features a built-in **Human-in-the-Loop (HITL)** pipeline that allows users to correct the bot during conversation. These corrections are automatically captured, validated, and used to fine-tune the model's adapter weights, creating a system that continuously evolves and adapts to its user's specific needs.

### **✨ Key Features**

* **🧠 Dynamic Intelligence:** Fine-tuned **Llama 3.2 3B** model optimized for edge deployment.  
* **🔄 Self-Healing Pipeline:** Automated workflow to ingest feedback, retrain LoRA adapters, and hot-swap weights without downtime.  
* **⚡ 4-Bit Quantization:** Runs efficiently on consumer hardware (CPU/T4 GPU) using **Unsloth**.  
* **🛠️ MLOps Maturity:** Full experiment tracking with **MLflow**, artifacts on **Hugging Face**, and CI/CD via **GitHub Actions**.

## **🏗️ System Architecture**

Ouro operates on a decoupled microservices architecture to ensure scalability and resilience.  
graph TD  
    User\[User\] \--\>|Chat| UI\[Streamlit Frontend\]  
    UI \--\>|Request| API\[FastAPI Backend\]  
    API \--\>|Inference| Model\[Llama 3.2 3B\]  
      
    User \--\>|Correction| UI  
    UI \--\>|Feedback| DB\[(feedback.jsonl)\]  
      
    subgraph "Continuous Learning Loop"  
        DB \--\>|Data Prep| Pipeline\[Training Pipeline\]  
        Pipeline \--\>|Fine-Tune| LoRA\[LoRA Adapters\]  
        LoRA \--\>|Hot Reload| Model  
    end

## **🛠️ Tech Stack**

| Component | Technology | Description |
| :---- | :---- | :---- |
| **Model** | **Llama 3.2 3B** | Optimized instruction-tuned base model. |
| **Training** | **Unsloth** | 2x faster training, 60% less memory. |
| **Inference** | **FastAPI** | High-performance async backend. |
| **Interface** | **Streamlit** | Interactive chat UI with feedback widgets. |
| **Tracking** | **MLflow** / **DagsHub** | Experiment logging and metrics. |
| **DevOps** | **Docker & GitHub Actions** | Containerization and CI/CD. |

## **🚀 Quick Start**

### **Prerequisites**

* Python 3.10+  
* Hugging Face Token (with write access)  
* (Optional) GPU for faster inference

### **1\. Clone the Repository**

git clone \[https://github.com/PierreRamez/PersonalChatbot.git\](https://github.com/PierreRamez/PersonalChatbot.git)  
cd PersonalChatbot

### **2\. Install Dependencies**

pip install \-r requirements.txt

### **3\. Run with Docker (Recommended)**

\# Build the image  
docker build \-t ouro-chatbot .

\# Run the container  
docker run \-p 7860:7860 ouro-chatbot

### **4\. Local Development**

\# Start the backend  
uvicorn app:app \--reload \--port 7860

Visit http://localhost:7860/docs to see the Swagger API documentation.

## **📊 Performance & Evaluation**

We evaluated Ouro using a **Sliding Window** strategy to assess conversational coherence across multi-turn dialogues.

| Metric | Score | Description |
| :---- | :---- | :---- |
| **Perplexity** | **4.08** | Indicates high predictive confidence. |
| **BERTScore F1** | **0.88** | Strong semantic alignment with human references. |
| **Inference Time** | **\~150ms** | Latency per token on T4 GPU. |

*Note:* The 3B model was selected over the 8B variant to enable CPU-only deployment on Hugging Face Spaces Free *Tier.*

## **📂 Project Structure**

.  
├── 📂 data/                 \# Raw and processed datasets  
├── 📂 finetune\_prep/        \# Scripts for preparing training batches  
├── 📜 app.py                \# FastAPI Backend Entrypoint  
├── 📜 test\_app.py           \# Unit Tests (Pytest)  
├── 📜 Dockerfile            \# Container Configuration  
├── 📜 learning\_pipeline.ipynb \# The HITL Training Orchestrator  
├── 📜 3b\_model.ipynb        \# Base Model Fine-Tuning Notebook  
└── 📜 main.yml              \# CI/CD Workflow

## **🤝 Contributing**

This project is part of the **Digital Egypt Pioneers Initiative (DEPI)**. While it is primarily for educational assessment, we welcome issues and feature requests.

1. Fork the Project  
2. Create your Feature Branch (git checkout \-b feature/AmazingFeature)  
3. Commit your Changes (git commit \-m 'Add some AmazingFeature')  
4. Push to the Branch (git push origin feature/AmazingFeature)  
5. Open a Pull Request

## **📜 License**

Distributed under the MIT License. See LICENSE for more information.  
\<div align="center"\>  
\<p\>Built with ❤️ by \<b\>Pierre Ramez\</b\>\</p\>  
\<p\>\<i\>Part of the DEPI Generative AI Track \- Round 3\</i\>\</p\>  
\</div\>