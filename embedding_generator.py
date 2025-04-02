from transformers import AutoTokenizer, AutoModel
import os
import ast

def get_embeddings(text):
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
    concatenated_text = " ".join(ast.literal_eval(text))
    inputs = tokenizer(concatenated_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)

    return outputs.last_hidden_state[0, 0, :]
