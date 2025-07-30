""" Evaluate Medical Tests Classification in LLMS """
## Setup
#### Load the API key and libraries.
import os

import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModel
import torch

## Hugging face Models
class HuggingFaceEmbeddings:
    """
    A class to handle text embedding generation using a Hugging Face pre-trained transformer model.
    This class loads the model, tokenizes the input text, generates embeddings, and provides an option 
    to save the embeddings to a CSV file.

    Args:
        model_name (str, optional): The name of the Hugging Face pre-trained model to use for generating embeddings. 
                                    Default is 'sentence-transformers/all-MiniLM-L6-v2'.
        path (str, optional): The path to the CSV file containing the text data. Default is 'data/file.csv'.
        save_path (str, optional): The directory path where the embeddings will be saved. Default is 'Models'.
        device (str, optional): The device to run the model on ('cpu' or 'cuda'). If None, it will automatically detect 
                                a GPU if available; otherwise, it defaults to CPU.

    Attributes:
        model_name (str): The name of the Hugging Face model used for embedding generation.
        tokenizer (transformers.AutoTokenizer): The tokenizer corresponding to the chosen model.
        model (transformers.AutoModel): The pre-trained model loaded for embedding generation.
        path (str): Path to the input CSV file.
        save_path (str): Directory where the embeddings CSV will be saved.
        device (torch.device): The device on which the model and data are processed (CPU or GPU).

    Methods:
        get_embedding(text):
            Generates embeddings for a given text input using the pre-trained model.

        get_embedding_df(column, directory, file):
            Reads a CSV file, computes embeddings for a specified text column, and saves the resulting DataFrame 
            with embeddings to a new CSV file in the specified directory.

    Example:
        embedding_instance = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', 
                                                   path='data/products.csv', save_path='output')
        text_embedding = embedding_instance.get_embedding("Sample product description.")
        embedding_instance.get_embedding_df(column='description', directory='output', file='product_embeddings.csv')

    Notes:
        - The Hugging Face model and tokenizer are downloaded from the Hugging Face hub.
        - The function supports large models and can run on either GPU or CPU, depending on device availability.
        - The input text will be truncated and padded to a maximum length of 512 tokens to fit into the model.
    """
    
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', path='data/file.csv', save_path=None, device=None):
        """
        Initializes the HuggingFaceEmbeddings class with the specified model and paths.

        Args:
            model_name (str, optional): The name of the Hugging Face pre-trained model. Default is 'sentence-transformers/all-MiniLM-L6-v2'.
            path (str, optional): The path to the CSV file containing text data. Default is 'data/file.csv'.
            save_path (str, optional): Directory path where the embeddings will be saved. Default is 'Models'.
            device (str, optional): Device to use for model processing. Defaults to 'cuda' if available, otherwise 'cpu'.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.path = path
        self.save_path = save_path or 'Models'
        
        # Define device
        if device is None:
            # Note: If you have a mac, you may want to change 'cuda' to 'mps' to use GPU
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        # Move model to the specified device
        self.model.to(self.device)
        print(f"Model moved to device: {self.device}")
        print(f"Model: {model_name}")
        
    def get_embedding(self, text):
        """
        Generates embeddings for a given text using the Hugging Face model.

        Args:
            text (str): The input text for which embeddings will be generated.

        Returns:
            np.ndarray: A numpy array containing the embedding vector for the input text.
        """
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        
        # Move the inputs to the device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
        
        return embeddings

    def get_embedding_df(self, column, directory, file):
        # Load the CSV file
        df = pd.read_csv(self.path)
        df["embeddings"] = df[column].apply(lambda x: self.get_embedding(x).tolist())
        
        os.makedirs(directory, exist_ok=True)
        df.to_csv(os.path.join(directory, file), index=False)