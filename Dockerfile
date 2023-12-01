
FROM python:3.11


WORKDIR /app

# Install the dependencies
COPY requirements.txt .
RUN pip install torch
RUN pip install -r requirements.txt

# Copiez le reste des fichiers de l'application dans le conteneur
COPY collect_url.py .
COPY gradio_interface.py .
COPY models.py .
COPY logical_state.py .
COPY utils.py .
COPY open_ai_key.txt .

EXPOSE 7860

# Commande pour exécuter le fichier python_test.py
CMD ["python", "gradio_interface.py"]