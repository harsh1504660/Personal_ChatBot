# Personal ChatBot 🤖

A simple personal chatbot built in Python to interact via the command line or GUI (depending on implementation). This project demonstrates how to set up a chatbot, define intents and responses, and deploy a local conversational agent.


🔗 **Live Demo / GitHub Repo**: [Portfolio](https://harsh-joshi-portfolio-zeta.vercel.app/bot)
---

## 🧩 Table of Contents

- [Features](#features)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
  - [Usage](#usage)  
- [🗂️ Project Structure](#project-structure)  
- [🚀 Architecture](#architecture)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Features

- Load intents and responses from a structured JSON or YAML file  
- User input processing with basic NLP (tokenization, bag‑of‑words)  
- Intent classification via machine learning (e.g. scikit-learn or TensorFlow)  
- Generate contextual responses  
- Easy to customize and extend  

---

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.8+  
- pip  

### Installation

1. Clone this repo  
   ```bash
   git clone https://github.com/harsh1504660/Personal_ChatBot.git
   cd Personal_ChatBot
2. Create and activate a virtual environment (recommended)
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # Mac/Linux
    venv\Scripts\activate      # Windows
    ```
3. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```
4. Usage
    ```bash
    python main.py
    ```

## 🚀 Architecture
Here's a high-level overview of the chatbot architecture:
```mermaid
flowchart LR
    A[User Input] --> B[Text Preprocessing]
    B --> C{Intent Classifier}
    C -->|Intent A| D[Select Response from A]
    C -->|Intent B| E[Select Response from B]
    C -->|Fallback| F[Default Response]
    D --> G[Return Response]
    E --> G
    F --> G
    G --> H[User]
