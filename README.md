# tweet_classification
Classification of Metaphorical Tweets by Fine-tunning DistilBert.

###########################################
# loading and saving
# Import the libraries
import torch
from transformers import DistilBertModel

# Specify the path or URL to the model file
PATH = "distilbert_model.pt"

# Load the model's state_dict
state_dict = torch.load(PATH, map_location="cpu")

# Create an instance of the model class
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Load the state_dict into the model
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

