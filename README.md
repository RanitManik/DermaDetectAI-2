# DermaDetectAI-v2

**DermaDetectAI-v2** is the upgraded, simplified version of [DermaDetectAI](https://github.com/RanitManik/DermaDetectAI), rebuilt for efficiency and ease of use. This project was created as part of a college initiative by **Ranit Kumar Manik**, **Mohammad**, **Sayak Bal**, and **Partha Sarathi Manna**, this version is fully web-based and deployed using **Streamlit**. It leverages a PyTorch-based deep learning model to predict skin diseases directly from images, all from a user-friendly browser interface.

## 🚀 Features

- ⚡ **Fast, Lightweight UI** – Built with Streamlit for quick interaction and easy deployment.
- 🧠 **Deep Learning Powered** – Uses a PyTorch model trained on a diverse skin disease dataset.
- 🖼️ **Image Upload** – Upload your skin image and get predictions instantly.
- 🧪 **Live Demo** – Easily deploy or run locally to test.
- 🎯 **Accurate Classification** – Fine-tuned model with optimized inference pipeline.

## 📦 Tech Stack

- **Frontend/UI**: Streamlit
- **Model & Inference**: PyTorch, TorchVision
- **Image Processing**: Pillow
- **Data Handling**: pandas


## 🔧 Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/RanitManik/DermaDetectAI-v2.git
   cd DermaDetectAI-v2
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the App**
   ```bash
   streamlit run app.py
   ```

5. Visit `http://localhost:8501` in your browser.

## 🧠 Model

The deep learning model is trained using PyTorch and TorchVision. It classifies images into multiple skin disease categories. The model weights are loaded at runtime, ensuring quick predictions without the need for retraining.

