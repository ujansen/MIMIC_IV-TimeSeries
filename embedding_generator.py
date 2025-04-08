from transformers import AutoTokenizer, AutoModel
import os
import ast
import torch

custom_directory = '/datasets/MIMIC-IV/bio-clinical-bert'
if not os.path.exists(custom_directory):
    os.makedirs(custom_directory)
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    tokenizer.save_pretrained(custom_directory)
    model.save_pretrained(custom_directory)
else:
    tokenizer = AutoTokenizer.from_pretrained(custom_directory)
    model = AutoModel.from_pretrained(custom_directory)

model.eval()  # disable dropout etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_embeddings(text):

    concatenated_text = " ".join(ast.literal_eval(text))
    inputs = tokenizer(concatenated_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state[0, 0, :]
