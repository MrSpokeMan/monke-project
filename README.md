# Environment Setup

1. **Install Ollama**
   - Download from [Ollama Website](https://ollama.com/download/windows)
   - After installation, open **CMD** and run:
     ```cmd
     ollama run llama3.2
     ```

2. **Set Up Milvus Database**
   - Follow the instructions at [Milvus Installation (Windows)](https://milvus.io/docs/install_standalone-windows.md)

3. **Create and Activate a Virtual Environment**
   - In your project directory, run:
     ```cmd
     python -m venv <venv_name>
     ```
   - Activate the virtual environment (Windows command prompt):
     ```cmd
     <venv_name>\Scripts\activate
     ```

4. **Install Dependencies**
   - While the environment is active, install the required packages:
     ```cmd
     pip install -r requirements.txt
     ```

5. **OPTIONAL** Install PyTorch
   - You might install PyTorch to use GPU while calculating embeddings:
     ```cmd
     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
     ```

---

# Load Dataset into the Database

- Populate the vector database by running:
  ```cmd
  py src\vector_db.py
  ```

---

# Run Frontend

- Start the Streamlit application by executing the following command in the terminal:
  ```cmd
  streamlit run src\frontend.py
  ```

---

# Run Evaluation

1. Create a `.env` file in the root directory of your project.
2. Generate an API key by visiting: [Google AI Studio – API Key](https://aistudio.google.com/app/apikey?hl=pl)
3. Inside the `.env` file, add the following line (replace `<API_KEY>` with your actual key):
   ```env
   GOOGLE_API_KEY=<API_KEY>
   ```
4. Execute the evaluation script:
   ```cmd
   py src\evaluation.py
   ```
