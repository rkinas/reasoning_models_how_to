import argparse
import random
import numpy as np
from functools import partial

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

import wandb
from tqdm import tqdm

def seed_everything(seed=42):
    """Ustawia losowy seed dla replikowalności eksperymentu."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def compute_dpo_loss(model_pref_logprob, model_dispref_logprob,
                      ref_pref_logprob, ref_dispref_logprob, beta=0.5):
    """Oblicza stratę DPO na podstawie log-prob modelu i modelu referencyjnego."""
    delta_pref = model_pref_logprob - ref_pref_logprob
    delta_dispref = model_dispref_logprob - ref_dispref_logprob
    loss = -F.logsigmoid(beta * (delta_pref - delta_dispref)).mean()
    
    return loss, delta_pref.mean(), delta_dispref.mean(), (delta_pref > delta_dispref).float().mean(), (delta_pref - delta_dispref).mean()

def extract_log_prob(logits, labels):
    """Pobiera log-probability dla etykiet na podstawie predykcji modelu."""
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1).mean(-1)

def collate_fn(batch, tokenizer, max_length, device):
    """Przygotowuje batch do treningu, używając wbudowanego chat_template tokenizera."""
    prompts = [tokenizer.apply_chat_template([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": item['prompt']},
        {"role": "assistant", "content": item['chosen']}
    ], tokenize=False) for item in batch]
    
    rejects = [tokenizer.apply_chat_template([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": item['prompt']},
        {"role": "assistant", "content": item['rejected']}
    ], tokenize=False) for item in batch]
    
    encode = lambda texts: tokenizer(texts, padding=True, return_tensors="pt", max_length=max_length, truncation=True)['input_ids'].to(device)
    
    prompt_pref, prompt_dispref = encode(prompts), encode(rejects)
    mask_pref, mask_dispref = torch.ones_like(prompt_pref), torch.ones_like(prompt_dispref)
    
    return {'prompt_pref': prompt_pref, 'prompt_dispref': prompt_dispref, 'mask_pref': mask_pref, 'mask_dispref': mask_dispref}

def train(model, ref_model, tokenizer, optimizer, dataloader, epochs=1, beta=0.1):
    """Trenuje model używając algorytmu DPO."""
    model.train()
    ref_model.eval()
    
    for epoch in range(epochs):
        for batch in tqdm(dataloader, desc=f'Training Epoch {epoch+1}'):
            optimizer.zero_grad()

            # Forward pass
            logits_pref = model(batch['prompt_pref'], attention_mask=batch['mask_pref']).logits
            logits_dispref = model(batch['prompt_dispref'], attention_mask=batch['mask_dispref']).logits
            ref_logits_pref = ref_model(batch['prompt_pref'], attention_mask=batch['mask_pref']).logits
            ref_logits_dispref = ref_model(batch['prompt_dispref'], attention_mask=batch['mask_dispref']).logits
            
            # Compute log probabilities
            model_pref_logprob = extract_log_prob(logits_pref, batch['prompt_pref'])
            model_dispref_logprob = extract_log_prob(logits_dispref, batch['prompt_dispref'])
            ref_pref_logprob = extract_log_prob(ref_logits_pref, batch['prompt_pref'])
            ref_dispref_logprob = extract_log_prob(ref_logits_dispref, batch['prompt_dispref'])
            
            # Compute DPO loss
            loss, delta_pref, delta_dispref, accuracy, margin = compute_dpo_loss(
                model_pref_logprob, model_dispref_logprob, ref_pref_logprob, ref_dispref_logprob, beta
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Log metrics
            wandb.log({'loss': loss.item(), 'pref_logprob_delta': delta_pref, 'dispref_logprob_delta': delta_dispref, 'accuracy': accuracy, 'margin': margin})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="microsoft/phi-2")
    parser.add_argument("--dataset_name", type=str, default="jondurbin/truthy-dpo-v0.1")
    parser.add_argument("--wandb_project", type=str, default="truthy-dpo")
    args = parser.parse_args()

    # Setup
    seed_everything(args.seed)
    wandb.init(project=args.wandb_project, config=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Load dataset & create dataloader
    dataset = load_dataset(args.dataset_name, split="train")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer, max_length=args.max_length, device=device))

    # Train
    train(model, ref_model, tokenizer, optimizer, dataloader, epochs=args.epochs, beta=args.beta)

    # Save model
    model.save_pretrained("model-DPO")

if __name__ == "__main__":
    main()
