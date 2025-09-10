Mini-GPT-2 Fine-Tune on Reddit Jokes

This project fine-tunes GPT-2 (or DistilGPT-2 for faster runs) on the Reddit Jokes dataset. The goal is to teach a small language model to generate one-liners and dad jokes.



📊 Project Details

Model: gpt2 (124M params) or distilgpt2 (82M params)

Dataset: Reddit Jokes (title + body → text) with a 90/10 train–eval split

Training Setup:

Epochs: 3

Context length (block_size): 256

Batch size: 16 (adjust if OOM)

Learning rate: 5e-5

Scheduler: cosine decay with warmup

Mixed precision (fp16): enabled on CUDA GPUs

Metrics: Evaluation loss + Perplexity (PPL)

Samples: Generated outputs are saved in samples.txt