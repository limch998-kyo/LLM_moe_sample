import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch.optim as optim
from sklearn.model_selection import train_test_split

import json
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import requests


class MixtureOfExpertsModel(nn.Module):
    def __init__(self, input_dim, conversation_dim):
        super(MixtureOfExpertsModel, self).__init__()
        # Adjust the input dimension to include conversation embedding
        self.classifier = nn.Linear(input_dim + conversation_dim, 1)
    
    def forward(self, embedding_conversation, embedding_facebook, embedding_microsoft):
        # Concatenate all embeddings
        combined_embedding = torch.cat((embedding_conversation, embedding_facebook, embedding_microsoft), dim=1)
        # Get binary decision
        decision = torch.sigmoid(self.classifier(combined_embedding))
        return decision



# Load models and tokenizers
tokenizer_facebook = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model_facebook = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")

# Initialize Microsoft tokenizer with left padding
tokenizer_microsoft = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side='left')
model_microsoft = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# tokenizer_microsoft = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono", padding_size="left")
# model_microsoft = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")

sentence_encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def generate_responses(input_question):
    # Generate response using Facebook model
    inputs_facebook = tokenizer_facebook(input_question, return_tensors="pt")
    outputs_facebook = model_facebook.generate(**inputs_facebook)
    response_facebook = tokenizer_facebook.decode(outputs_facebook[0], skip_special_tokens=True)

    # Generate response using Microsoft model
    inputs_microsoft = tokenizer_microsoft.encode(input_question + tokenizer_microsoft.eos_token, return_tensors="pt")
    outputs_microsoft = model_microsoft.generate(inputs_microsoft, max_length=50, pad_token_id=tokenizer_microsoft.eos_token_id)
    response_microsoft = tokenizer_microsoft.decode(outputs_microsoft[0], skip_special_tokens=True)

    return response_facebook, response_microsoft

def extract_qna_pairs(jsonl_file):
    qna_pairs = []

    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            dialogue = data['dialogue']
            turns = dialogue.split('\n')
            if len(turns) >= 2:
                question = turns[0].split(': ', 1)[1]
                answer = turns[1].split(': ', 1)[1]
                qna_pairs.append((question, answer))
    
    return qna_pairs


input_dim = sentence_encoder.get_sentence_embedding_dimension() * 2  # Since we concatenate two embeddings

# Initialize MoE model with updated dimensions
conversation_dim = sentence_encoder.get_sentence_embedding_dimension()  # Dimension of the conversation embedding
moe_model = MixtureOfExpertsModel(input_dim, conversation_dim)

criterion = nn.BCELoss()
optimizer = optim.Adam(moe_model.parameters(), lr=0.001)

def train(model, data, optimizer, criterion):
    model.train()
    total_loss = 0
    for embedding_input, embedding_facebook, embedding_microsoft, label in data:
        # Forward pass with conversation embedding
        decision = model(embedding_input, embedding_facebook, embedding_microsoft)

        # Squeeze the decision tensor to remove extra dimensions if any
        decision = decision.squeeze()

        # Convert label to a scalar
        label_scalar = torch.tensor(label, dtype=torch.float)

        # Calculate the loss
        loss = criterion(decision, label_scalar)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data)

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/msmarco-distilbert-base-tas-b"
api_token = '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'
headers = {"Authorization": f"Bearer {api_token}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def get_similarity_scores(actual_answer, response_facebook, response_microsoft):
    payload = {
        "inputs": {
            "source_sentence": actual_answer,
            "sentences": [response_facebook, response_microsoft]
        }
    }
    data = query(payload)

    # Assuming the API returns similarity scores in the same order as the sentences
    similarity_facebook = data[0]
    similarity_microsoft = data[1]

    return similarity_facebook, similarity_microsoft

# Rest of your code for extracting question-answer pairs...

# Prepare dataset
dataset = []
jsonl_file = '/Users/limchaehyun/Projects/KSL_seminar/DialogSum_Data/dialogsum.train.jsonl'
qna_pairs = extract_qna_pairs(jsonl_file)

print(len(qna_pairs))
qna_pairs = qna_pairs[:30]

for question, actual_answer in qna_pairs:
    response_facebook, response_microsoft = generate_responses(question)


    # Encode responses
    embedding_input = sentence_encoder.encode([question], convert_to_tensor=True)
    embedding_facebook = sentence_encoder.encode([response_facebook], convert_to_tensor=True)
    embedding_microsoft = sentence_encoder.encode([response_microsoft], convert_to_tensor=True)

    # Get similarity scores
    similarity_facebook, similarity_microsoft = get_similarity_scores(actual_answer, response_facebook, response_microsoft)

    # Label based on similarity scores
    label = 1 if similarity_facebook > similarity_microsoft else 0
    dataset.append((embedding_input, embedding_facebook, embedding_microsoft, label))

# Training loop
num_epochs = 100  # You can adjust this
for epoch in range(num_epochs):
    train_loss = train(moe_model, dataset, optimizer, criterion)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}')

# Save the model weights
model_save_path = 'moe_model_weights.pth'
torch.save(moe_model.state_dict(), model_save_path)


# Load test data
test_jsonl_file = '/Users/limchaehyun/Projects/KSL_seminar/DialogSum_Data/dialogsum.test.jsonl'
test_qna_pairs = extract_qna_pairs(test_jsonl_file)

# Prepare for sampling outputs
moe_model.eval()  # Set the model to evaluation mode

for question, _ in test_qna_pairs[:10]:  # Limiting to first 5 for sampling
    response_facebook, response_microsoft = generate_responses(question)

    # Encode responses
    embedding_input = sentence_encoder.encode([question], convert_to_tensor=True)
    embedding_facebook = sentence_encoder.encode([response_facebook], convert_to_tensor=True)
    embedding_microsoft = sentence_encoder.encode([response_microsoft], convert_to_tensor=True)

    # Forward pass through MoE model
    with torch.no_grad():  # No need to track gradients
        decision = moe_model(embedding_input, embedding_facebook, embedding_microsoft).squeeze()

    # Choose response based on MoE decision
    chosen_response = response_facebook if decision.item() > 0.5 else response_microsoft

    # Print the question, responses, and chosen response
    print(f"Question: {question}")
    print(f"Response Facebook: {response_facebook}")
    print(f"Response Microsoft: {response_microsoft}")
    print(f"Chosen Response (MoE): {chosen_response}\n")


# Initialize models


# import requests
# headers = {"Authorization": f"Bearer {api_token}"}
# API_URL = "https://datasets-server.huggingface.co/is-valid?dataset=mozilla-foundation/common_voice_10_0"
# def query():
#     response = requests.get(API_URL, headers=headers)
#     return response.json()
# data = query()
# print(data)