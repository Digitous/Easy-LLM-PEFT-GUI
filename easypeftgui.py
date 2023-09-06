from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os
import shutil
import tkinter as tk
from tkinter import filedialog

def select_path(title):
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    path = filedialog.askdirectory(title=title)  # Open the directory selection dialog
    return path

def get_args():
    base_model_name_or_path = select_path("Select the base model")
    peft_model_path = select_path("Select PEFT adapter")
    output_dir = select_path("Select output dir")
    
    return base_model_name_or_path, peft_model_path, output_dir

def main():
    base_model_name_or_path, peft_model_path, output_dir = get_args()
    
    # Check GPU availability and print
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    
    # Set device
    device = "cuda" if cuda_available else "cpu"
    device_arg = { 'device_map': device }

    print(f"Loading base model: {base_model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float16,
        **device_arg
    )

    print(f"Loading PEFT: {peft_model_path}")
    model = PeftModel.from_pretrained(base_model, peft_model_path, **device_arg)
    print(f"Running merge_and_unload")
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

    model.save_pretrained(f"{output_dir}")
    tokenizer.save_pretrained(f"{output_dir}")
    print(f"Model saved to {output_dir}")

if __name__ == "__main__" :
    main()
