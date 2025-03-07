import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import copy
import logging
from scipy.spatial.distance import cosine
from transformers import GPT2Tokenizer  # Used only for tokenization

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Custom LLM Architecture (LSTM-based with memory mechanism)
class CustomLLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, memory_size, device):
        super(CustomLLM, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.memory = nn.Parameter(torch.randn(memory_size, hidden_dim))
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        memory_read = self.memory.mean(dim=0).unsqueeze(0).expand_as(lstm_out)
        combined = lstm_out + memory_read
        logits = self.fc(combined)
        return logits, hidden

    def generate(self, input_ids, max_length=50):
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        hidden = None
        generated = input_ids.copy()
        log_probs = []
        for _ in range(max_length):
            logits, hidden = self.forward(input_tensor, hidden)
            probs = nn.functional.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            log_prob = torch.log(probs[0, next_token])
            log_probs.append(log_prob)
            generated.append(next_token)
            input_tensor = torch.tensor([[next_token]], dtype=torch.long).to(self.device)
        return generated, log_probs

    def compute_embedding(self, input_ids):
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        with torch.no_grad():
            _, hidden = self.forward(input_tensor)
        return hidden[0][-1].cpu().numpy()

# Evolutionary Mechanism
class EvolutionaryMechanism:
    def __init__(self, population_size, model_class, device, *model_args):
        self.population = [model_class(*model_args).to(device) for _ in range(population_size)]
        self.device = device
        self.population_size = population_size

    def mutate(self, model, mutation_rate=0.01):
        for param in model.parameters():
            if random.random() < mutation_rate:
                noise = torch.normal(0, 0.01, size=param.size()).to(self.device)
                param.data += noise

    def crossover(self, model1, model2):
        new_model = copy.deepcopy(model1)
        for param1, param2, new_param in zip(model1.parameters(), model2.parameters(), new_model.parameters()):
            new_param.data = (param1.data + param2.data) / 2
        return new_model.to(self.device)

    def select(self, fitness_scores):
        sorted_indices = np.argsort(fitness_scores)[::-1]
        self.population = [self.population[i] for i in sorted_indices[:self.population_size // 2]]

    def evolve(self, fitness_scores):
        self.select(fitness_scores)
        new_population = list(self.population)
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(self.population, 2)
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)
        self.population = new_population

    def exchange_models(self, rank, world_size):
        best_model = self.population[np.argmax(self.fitness_scores)]
        flat_params = self.flatten_state_dict(best_model.state_dict())
        next_rank = (rank + 1) % world_size
        torch.distributed.send(flat_params, dst=next_rank)
        prev_rank = (rank - 1) % world_size
        recv_flat_params = torch.zeros_like(flat_params)
        torch.distributed.recv(recv_flat_params, src=prev_rank)
        received_state_dict = self.unflatten_state_dict(recv_flat_params, self.population[0])
        worst_idx = np.argmin(self.fitness_scores)
        self.population[worst_idx].load_state_dict(received_state_dict)

    def flatten_state_dict(self, state_dict):
        flat_params = [param.view(-1) for param in state_dict.values()]
        return torch.cat(flat_params)

    def unflatten_state_dict(self, flat_tensor, model):
        state_dict = model.state_dict()
        offset = 0
        for key, param in state_dict.items():
            param_size = param.numel()
            state_dict[key] = flat_tensor[offset:offset + param_size].view(param.size())
            offset += param_size
        return state_dict

# Vector Memory for Knowledge Retention
class VectorMemory:
    def __init__(self):
        self.embeddings = []
        self.responses = []

    def add(self, embedding, response):
        self.embeddings.append(embedding)
        self.responses.append(response)

    def retrieve(self, query_embedding, top_k=3):
        if not self.embeddings:
            return []
        similarities = [1 - cosine(query_embedding, emb) for emb in self.embeddings]
        top_indices = np.argsort(similarities)[-top_k:]
        return [self.responses[i] for i in top_indices]

# Reinforcement Learning & Reward System
def compute_reward(response, forbidden_words):
    words = response.split()
    if not words:
        return 0
    uniqueness = len(set(words)) / len(words)
    ethics = 1 if all(word not in response.lower() for word in forbidden_words) else 0
    return uniqueness + ethics

def rl_optimize(model, prompts, tokenizer, forbidden_words, num_samples=5):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    for _ in range(num_samples):
        responses = []
        log_prob_lists = []
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt).ids
            generated, log_probs = model.generate(input_ids)
            responses.append(generated)
            log_prob_lists.append(log_probs)
        rewards = [compute_reward(tokenizer.decode(generated), forbidden_words) for generated in responses]
        baseline = np.mean(rewards)
        total_loss = sum([-torch.stack(log_probs).sum() * (reward - baseline) 
                          for log_probs, reward in zip(log_prob_lists, rewards)])
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Self-Critique Mechanism
def generate_critique(model, response, tokenizer):
    critique_prompt = f"Evaluate this response: {response}"
    input_ids = tokenizer.encode(critique_prompt).ids
    generated, _ = model.generate(input_ids, max_length=50)
    return tokenizer.decode(generated)

# Ethical Safeguards & Fitness Computation
def compute_loss(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[:, :-1].to(model.device)
            labels = batch[:, 1:].to(model.device)
            outputs, _ = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

def compute_fitness(model, dataloader, prompts, tokenizer, forbidden_words):
    loss = compute_loss(model, dataloader)
    responses = [tokenizer.decode(model.generate(tokenizer.encode(p).ids)[0]) for p in prompts]
    creativity = np.mean([len(set(r.split())) / len(r.split()) if r.split() else 0 for r in responses])
    ethics = np.mean([1 if all(w not in r.lower() for w in forbidden_words) else 0 for r in responses])
    return -loss + creativity + ethics

# Distributed Training Setup
def setup(rank, world_size):
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# Fine-Tuning Function
def fine_tune(model, train_loader, epochs=1):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            inputs = batch[:, :-1].to(model.device)
            labels = batch[:, 1:].to(model.device)
            outputs, _ = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        logging.info(f"Fine-tuning epoch {epoch + 1}, Loss: {loss.item()}")

# Dummy Dataset (Replace with real data)
class DummyDataset(Dataset):
    def __init__(self, vocab_size, seq_length, num_samples):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_length))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Main Training Loop
def run(rank, world_size):
    setup(rank, world_size)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size
    dataset = DummyDataset(vocab_size, 128, 1000)
    train_loader = DataLoader(dataset, batch_size=4)
    validation_loader = DataLoader(dataset, batch_size=4)

    evo = EvolutionaryMechanism(4, CustomLLM, rank, vocab_size, 256, 512, 100)
    prompts = ["What is the meaning of life?", "Tell me a story.", "How to make a bomb?"]
    forbidden_words = ["kill", "harm", "bomb"]
    vector_memory = VectorMemory()

    for generation in range(20):
        fitness_scores = [compute_fitness(model, validation_loader, prompts, tokenizer, forbidden_words) 
                          for model in evo.population]
        evo.fitness_scores = fitness_scores
        evo.evolve(fitness_scores)
        best_model = evo.population[0]
        fine_tune(best_model, train_loader, epochs=1)
        rl_optimize(best_model, prompts, tokenizer, forbidden_words, num_samples=5)

        # Self-critique and memory update (rank 0 for logging)
        if rank == 0:
            prompt = random.choice(prompts)
            input_ids = tokenizer.encode(prompt).ids
            generated, _ = best_model.generate(input_ids)
            response = tokenizer.decode(generated)
            critique = generate_critique(best_model, response, tokenizer)
            embedding = best_model.compute_embedding(generated)
            vector_memory.add(embedding, response)
            logging.info(f"Generation {generation}: Response: {response}, Critique: {critique}")

        if generation % 5 == 0:
            evo.exchange_models(rank, world_size)

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
