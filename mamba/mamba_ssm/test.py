import warnings
warnings.filterwarnings("ignore")
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging
logging.set_verbosity_error()
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses INFO (1), WARNING (2), and ERROR (3)
tf.get_logger().setLevel('ERROR')         # Suppress TensorFlow logger

model = AutoModelForCausalLM.from_pretrained("/scratch/vetgpt/vetgpt-rlp/mamba/mamba_ssm/mamba_finetuned_results")
tokenizer = AutoTokenizer.from_pretrained("/scratch/vetgpt/vetgpt-rlp/mamba/mamba_ssm/mamba_finetuned_results")

prompt = "When is vomiting in cats considered an emergency?"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(inputs["input_ids"], max_length=64, temperature=1)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)