from abc import ABC, abstractmethod
from transformers import BertModel, BertTokenizer, LongformerModel, LongformerTokenizer
import torchvision.models as models
import torch
from PIL import Image
from torchvision import transforms
import logging

class TextEmbedder(ABC):
    @abstractmethod
    def token_count(self, input_text):
        pass

    @abstractmethod
    def get_text_embedding(self, input_text):
        pass

class ImageEmbedder(ABC):
    @abstractmethod
    def get_image_embedding(self, image_path):
        pass

class BERTTextEmbedder(TextEmbedder):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(self.device)

    def token_count(self, input_text):
        tokens = self.tokenizer(input_text)
        return len(tokens.input_ids)

    def get_text_embedding(self, input_text):
        try:
            encoded_input = self.tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            with torch.no_grad():
                output = self.model(**encoded_input)
                last_hidden_states = output.last_hidden_state
                mean_embedding = last_hidden_states.mean(dim=1).squeeze().tolist()
            return mean_embedding
        except Exception as e:
            logging.error("Error in generating text embedding: {}".format(e))
            return None

class LongformerTextEmbedder(TextEmbedder):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        self.model = LongformerModel.from_pretrained('allenai/longformer-base-4096').to(self.device)

    def token_count(self, input_text):
        tokens = self.tokenizer(input_text)
        return len(tokens.input_ids)

    def get_text_embedding(self, input_text):
        try:
            encoded_input = self.tokenizer(input_text, return_tensors='pt', max_length=4096, truncation=True, padding='max_length')
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            with torch.no_grad():
                output = self.model(**encoded_input)
                last_hidden_states = output.last_hidden_state
                mean_embedding = last_hidden_states.mean(dim=1).squeeze().tolist()
            return mean_embedding
        except Exception as e:
            logging.error("Error in generating text embedding: {}".format(e))
            return None

class ImageEmbedder:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_model = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1").to(self.device)
        self.image_model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_image_embedding(self, image_path):
        try:
            image = Image.open(image_path)
            image = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.image_model(image)
            return output.squeeze().tolist()
        except Exception as e:
            logging.error("Error in generating image embedding: {}".format(e))
            return None

class EmbeddingModel:
    def __init__(self, text_model_name, image_model_name):
        if text_model_name == 'bert':
            self.text_embedder = BERTTextEmbedder()
        elif text_model_name == 'longformer':
            self.text_embedder = LongformerTextEmbedder()
        else:
            raise ValueError("Unsupported text model name")

        if image_model_name == 'resnet':
            self.image_embedder = ImageEmbedder()
        else:
            raise ValueError("Unsupported image model name")

    def get_text_embedding(self, input_text):
        return self.text_embedder.get_text_embedding(input_text)

    def token_count(self, input_text):
        return self.text_embedder.token_count(input_text)

    def get_image_embedding(self, image_path):
        return self.image_embedder.get_image_embedding(image_path)
