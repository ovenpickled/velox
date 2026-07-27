import torch
import json
import struct
import sys
import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "The capital of France is"
    output_file = sys.argv[3] if len(sys.argv) > 3 else "reference_logits.bin"
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float32)
    model.eval()
    
    print(f"Prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"Token IDs: {input_ids[0].tolist()}")
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]  # last position logits
    
    logits_np = logits.numpy()
    print(f"Logits shape: {logits_np.shape}")
    print(f"Top 10 tokens:")
    top_indices = np.argsort(logits_np)[-10:][::-1]
    for idx in top_indices:
        token = tokenizer.decode([idx])
        print(f"  {idx}: {token!r} = {logits_np[idx]:.4f}")
    
    # Save logits as binary
    with open(output_file, 'wb') as f:
        f.write(struct.pack('i', len(logits_np)))
        f.write(logits_np.tobytes())
    
    print(f"Saved {len(logits_np)} logits to {output_file}")
    
    # Also save token IDs for the C++ engine to use
    token_ids_file = output_file.replace('.bin', '_tokens.json')
    with open(token_ids_file, 'w') as f:
        json.dump({
            'prompt': prompt,
            'token_ids': input_ids[0].tolist(),
            'vocab_size': len(logits_np)
        }, f, indent=2)
    print(f"Saved token IDs to {token_ids_file}")

if __name__ == '__main__':
    main()
