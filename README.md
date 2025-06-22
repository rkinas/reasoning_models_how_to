# **Reasoning Model and RLHF Research Notes**  

This repository serves as a collection of research notes and resources on **training large language models (LLMs)** and **Reinforcement Learning from Human Feedback (RLHF)**. It focuses on the latest research, methodologies, and techniques for fine-tuning language models.  

## **Repository Contents**  

### **Reinforcement Learning and RLHF Overview**  
A curated list of materials providing an introduction to RL and RLHF:  
- Research papers and books covering key concepts in reinforcement learning.  
- Video lectures explaining the fundamentals of RLHF.  

### **Methods for LLM Training**  
An extensive collection of state-of-the-art approaches for optimizing preferences and model alignment:  
- Key techniques such as PPO, DPO, KTO, ORPO, and more.  
- The latest ArXiv publications and publicly available implementations.  
- Analysis of effectiveness across different optimization strategies.  

## **Purpose of this Repository**  
This repository is designed as a reference for researchers and engineers working on **reinforcement learning and large language models**. If you're interested in **model alignment**, **experiments with DPO and its variants**, or **alternative RL-based methods**, you will find valuable resources here.  

## RL overview
- [Reinforcement Learning: An Overview](https://arxiv.org/pdf/2412.05265)
- [A COMPREHENSIVE SURVEY OF LLM ALIGNMENT TECHNIQUES: RLHF, RLAIF, PPO, DPO AND MORE](https://arxiv.org/pdf/2407.16216)
- [Book-Mathematical-Foundation-of-Reinforcement-Learning](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)
- [The FASTEST introduction to Reinforcement Learning on the internet](https://www.youtube.com/watch?v=VnpRp7ZglfA)
- [rlhf-book](https://github.com/natolambert/rlhf-book)
- [Notes on reinforcement learning](https://newfacade.github.io/notes-on-reinforcement-learning/01-intro.html)

## Methods for LLM training
- [PPO - Proximal Policy Optimization Algorithm - OpenAI](https://arxiv.org/pdf/1707.06347)
- [DPO - Direct Preference Optimization: Your Language Model is Secretly a Reward Model - Standford](https://arxiv.org/pdf/2305.18290)
- [online DPO]()
- [KTO - KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/pdf/2402.01306)
- [SimPO imple Preference Optimization with a Reference-Free Reward - Princeton](https://arxiv.org/pdf/2405.14734v1)
- [ORPO - Monolithic Preference Optimization without Reference Model - Kaist AI](https://arxiv.org/pdf/2403.07691v2)
- [Sample Efficient Reinforcement Learning with REINFORCE](https://arxiv.org/pdf/2010.11364)
- [REINFORCE++](https://arxiv.org/pdf/2501.03262v1)
- [RPO Reward-aware Preference Optimization: A Unified Mathematical Framework for Model Alignment](https://arxiv.org/pdf/2501.03262v1)
- [RLOO - Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/pdf/2402.14740) 
- [GRPO](https://arxiv.org/pdf/2402.03300)
- [ReMax -  Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models](https://arxiv.org/pdf/2310.10505)
- [DPOP - Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive](https://arxiv.org/abs/2402.13228)
- [BCO - Binary Classifier Optimization for Large Language Model Alignment](https://arxiv.org/pdf/2404.04656v1)

## Minimal implementation
|    Method                                                                                              |
|--------------------------------------------------------------------------------------------------------|
| [DPO](https://github.com/rkinas/rlhf_thinking_model/blob/main/minimal_implementation/dpo_trainer.py)   |   

## Tutorials
Notes for learning RL: Value Iteration -> Q Learning -> DQN -> REINFORCE -> Policy Gradient Theorem -> TRPO -> PPO
- [CS234: Reinforcement Learning Winter 2025 ](https://web.stanford.edu/class/cs234/)
- [CS285 Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/)
- [Welcome to Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/index.html)
- [deep-rl-course from Huggingface](https://huggingface.co/learn/deep-rl-course/unit0/introduction)
- [RL Course by David Silver](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-)


## RLHF training techniques explained
- [Reinforcement Learning from Human Feedback explained with math derivations and the PyTorch code.](https://www.youtube.com/watch?v=qGyFrqc34yc)
- [Direct Preference Optimization (DPO) explained: Bradley-Terry model, log probabilities, math](https://www.youtube.com/watch?v=hvGa5Mba4c8)
- [GRPO vs PPO](https://yugeten.github.io/posts/2025/01/ppogrpo/)
- [Unraveling RLHF and Its Variants: Progress and Practical Engineering Insights](https://hijkzzz.notion.site/Unraveling-RLHF-and-Its-Variants-Progress-and-Practical-Engineering-Insights-147d9a33ecc980199dc5cb967c5e9374)

## Training frameworks
- [VERL](https://github.com/volcengine/verl) - Volcano Engine Reinforcement Learning for LLMs
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) - OpenRLHF is the first easy-to-use, high-performance open-source RLHF framework built on Ray, vLLM, ZeRO-3 and HuggingFace Transformers, designed to make RLHF training simple and accessible
- [TRL](https://huggingface.co/docs/trl/) - TRL is a full stack library where we provide a set of tools to train transformer language models with methods like Supervised Fine-Tuning (SFT), Group Relative Policy Optimization (GRPO), Direct Preference Optimization (DPO), Reward Modeling, and more. 
- [Nemo-RL](https://github.com/NVIDIA-NeMo/RL) - Nemo RL: A Scalable and Efficient Post-Training Library
- [ROLL](https://github.com/alibaba/ROLL/) - Large scale training with megatron support, a feature-rich codebase from Alibaba
- [RL2](https://github.com/ChenmienTan/RL2) - Ray Less Reinforcement Learning. The NanoGPT of RL with it's small and hackable size (<1k lines)
- [AReal](https://github.com/inclusionAI/AReaL) - AReaL (Ant Reasoning RL): LLM generation runs in a streaming manner, with each rollout worker continuously producing outputs without waiting
- [OAT](https://github.com/sail-sg/oat) - Oat 🌾 is a simple yet efficient framework for running online LLM alignment algorithms.
- [](https://arxiv.org/html/2505.24034v2) - Meta GenAI - LlamaRL: A Distributed Asynchronous Reinforcement Learning Framework for Efficient Large-scale LLM Training

## RLHF methods implementation (only with detailed explanations)
- GRPO
  - [GRPO A.Burkov](https://github.com/aburkov/theLMbook/blob/main/GRPO.py)
  - [Minimal implementation by willccbb](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb)
  - [TinyZero](https://github.com/Jiayi-Pan/TinyZero)
  - [microGRPO](https://github.com/superlinear-ai/microGRPO)

## Articles
- [Reasoning LLMs](https://docs.google.com/document/d/1TW7wEUgo61FZnPckZMploGTdB0eNcemiDPDqdmzsCvA/edit?tab=t.0)
- [Process Reinforcement through Implicit Rewards](https://curvy-check-498.notion.site/Process-Reinforcement-through-Implicit-Rewards-15f4fcb9c42180f1b498cc9b2eaf896f)
- [DeepScaleR: Surpassing O1-Preview with a 1.5B Model by Scaling RL](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2)
- [On the Emergence of Thinking in LLMs I: Searching for the Right Intuition](https://arxiv.org/pdf/2502.06773)
- [LIMR: Less is More for RL Scaling](https://arxiv.org/pdf/2502.11886)
- [LIMO: Less Is More for Reasoning](https://github.com/GAIR-NLP/LIMO)
- [s1: Simple test-time scaling](https://github.com/simplescaling/s1) and s1.1 
- [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- [Online-DPO-R1: Unlocking Effective Reasoning Without the PPO Overhead](https://efficient-unicorn-451.notion.site/Online-DPO-R1-Unlocking-Effective-Reasoning-Without-the-PPO-Overhead-1908b9a70e7b80c3bc83f4cf04b2f175) and [github](https://github.com/RLHFlow/Online-DPO-R1)
- [a reinforcement learning guide](https://naklecha.notion.site/a-reinforcement-learning-guide)
- [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)
- [How to align open LLMs in 2025 with DPO & and synthetic data](https://www.philschmid.de/rl-with-llms-in-2025-dpo)
- DeepSeek-R1 -> [The Illustrated DeepSeek-R1](https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1), [DeepSeek R1's recipe to replicate o1 and the future of reasoning LMs](https://www.interconnects.ai/p/deepseek-r1-recipe-for-o1), [DeepSeek R1 and R1-Zero Explained](https://thelmbook.com/articles/#!./DeepSeek-R1.md)

- 2025.03.23
  - [Reinforcement Learning for Reasoning in Small LLMs: What Works and WhatDoesn’t](https://arxiv.org/pdf/2503.16219)
  - [Understanding R1-zero](https://github.com/sail-sg/understand-r1-zero/blob/main/understand-r1-zero.pdf)

- 2025.02.22
  - [Small Models Struggle to Learn from Strong Reasoners](https://arxiv.org/pdf/2502.12143v1)
  - [Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning](https://arxiv.org/pdf/2502.14768)
  - [LongPO: Long Context Self-Evolution of Large Language Models through Short-to-Long Preference Optimization](https://www.arxiv.org/abs/2502.13922)
  - [Open Reasoner Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero) An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model


# Thinking process

## Repos
- [Awesome-System2-Reasoning-LLM](https://github.com/zzli2022/Awesome-System2-Reasoning-LLM)

## Articles
- ✨ [LLM Reasoning: Curated Insights](https://shangshangwang.notion.site/llm-reasoning)
- [LLMs Can Easily Learn to Reason from Demonstrations Structure, not content, is what matters!](https://arxiv.org/pdf/2502.07374)
- [LLM Post-Training: A Deep Dive into Reasoning Large Language Models](https://arxiv.org/pdf/2502.21321)

## Papers
- [SelfCite: Self-Supervised Alignment for Context Attribution in Large Language Models](https://arxiv.org/abs/2502.09604)
- [ReasonFlux: Hierarchical LLM Reasoning via Scaling Thought Templates](https://arxiv.org/abs/2502.06772)
- [A Minimalist Approach to Offline Reinforcement Learning](https://arxiv.org/abs/2106.06860)
- [Training Language Models to Reason Efficiently](https://arxiv.org/abs/2502.04463)
- [Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search](https://arxiv.org/abs/2502.02508)


## Open-source project to reproduce DeepSeek R1
- [DeepScaleR - Democratizing Reinforcement Learning for LLMs](https://github.com/agentica-project/deepscaler)

## Datasets - thinking models
- [R1 - distill] [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)
- [R1 - distill] [s1K-1.1](https://huggingface.co/datasets/simplescaling/s1K-1.1)
- [R1 - distill] [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k)
- [R1 - distill] [LIMO](https://huggingface.co/datasets/GAIR/LIMO)
- [R1 - distill] [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)
- [Llama-70B - distill] [natural_reasoning](https://huggingface.co/datasets/facebook/natural_reasoning) - licence for non commercial use
- [Open Reasoning Data ](https://gr.inc/)
- [Big-Math: A Large-Scale, High-Quality Math Dataset for Reinforcement Learning in Language Models](https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified)

# Evaluation and benchmarks
- [Open R1 - A fully open reproduction of DeepSeek-R1](https://github.com/huggingface/open-r1)
- [GMIL CM Benchmark - Math Reasoning as an 11-Year-Old](https://github.com/przadka/gmil-cm-benchmark?tab=readme-ov-file)
  
