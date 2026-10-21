FROM python:3.12-slim

# Run as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

COPY --chown=user requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY --chown=user recommender.py app.py api.py cover-not-found.jpg ./
COPY --chown=user data/processed/books_with_emotions.csv data/processed/
COPY --chown=user data/chroma_db data/chroma_db

EXPOSE 7860
# OPENAI_API_KEY is passed at run time, never baked into the image:
#   docker run -p 7860:7860 --env-file .env book-recommender
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
