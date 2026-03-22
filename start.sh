#!/bin/bash

ollama serve &

sleep 5

echo "pulling ollama"
ollama pull llama3.2

echo "running streamlit"
streamlit run app.py --server.port=7860 --server.address=0.0.0.0
