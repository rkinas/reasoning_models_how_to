# Thinking model research notes

# RL overview
- [Reinforcement Learning: An Overview](https://arxiv.org/pdf/2412.05265)
- [A COMPREHENSIVE SURVEY OF LLM ALIGNMENT TECHNIQUES: RLHF, RLAIF, PPO, DPO AND MORE](https://arxiv.org/pdf/2407.16216)
- [Book-Mathematical-Foundation-of-Reinforcement-Learning](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)
- [The FASTEST introduction to Reinforcement Learning on the internet](https://www.youtube.com/watch?v=VnpRp7ZglfA)
- [rlhf-book](https://github.com/natolambert/rlhf-book)

# Methods for LLM training
- [PPO - Proximal Policy Optimization Algorithm - OpenAI](https://arxiv.org/pdf/1707.06347)
- [DPO - Direct Preference Optimization: Your Language Model is Secretly a Reward Model - Standford](https://arxiv.org/pdf/2305.18290)
- [online DPO]()
- [KTO - KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/pdf/2402.01306)
- [SimPO imple Preference Optimization with a Reference-Free Reward - Princeton](https://arxiv.org/pdf/2405.14734v1)
- [ORPO - Monolithic Preference Optimization without Reference Model - Kaist AI](https://arxiv.org/pdf/2403.07691v2)
- [REINFORCE]
- [REINFORCE++](https://arxiv.org/pdf/2501.03262v1)
- [RPO Reward-aware Preference Optimization: A Unified Mathematical Framework for Model Alignment](https://arxiv.org/pdf/2501.03262v1)
- RLOO 
- [GRPO]
- [ReMax -  Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models](https://arxiv.org/pdf/2310.10505)
- [DPOP]
- [BCO]
- [GKD]

# Tutorials
Notes for learning RL: Value Iteration -> Q Learning -> DQN -> REINFORCE -> Policy Gradient Theorem -> TRPO -> PPO
- [CS234: Reinforcement Learning Winter 2025 ](https://web.stanford.edu/class/cs234/)
- [CS285 Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/)
- [Welcome to Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/index.html)
- [deep-rl-course from Huggingface](https://huggingface.co/learn/deep-rl-course/unit0/introduction)


# RL explained
- [GRPO vs PPO](https://yugeten.github.io/posts/2025/01/ppogrpo/)

# Training frameworks
- [VERL](https://github.com/volcengine/verl)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [TRL](https://huggingface.co/docs/trl/)

# Articles
- [LIMR: Less is More for RL Scaling](https://arxiv.org/pdf/2502.11886)
- [LIMO: Less Is More for Reasoning](https://github.com/GAIR-NLP/LIMO)
- [s1: Simple test-time scaling](https://github.com/simplescaling/s1) and s1.1 
- [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- [Online-DPO-R1: Unlocking Effective Reasoning Without the PPO Overhead](https://efficient-unicorn-451.notion.site/Online-DPO-R1-Unlocking-Effective-Reasoning-Without-the-PPO-Overhead-1908b9a70e7b80c3bc83f4cf04b2f175) and [github](https://github.com/RLHFlow/Online-DPO-R1)
- [a reinforcement learning guide](https://naklecha.notion.site/a-reinforcement-learning-guide)

# Datasets - thinking models
- [Y] https://huggingface.co/datasets/open-r1/OpenR1-Math-220k

- [Y] https://huggingface.co/datasets/simplescaling/s1K-1.1
- [Y] https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k
- [Y] https://huggingface.co/datasets/GAIR/LIMO
- [Y] https://huggingface.co/datasets/AI-MO/NuminaMath-CoT

LIMR: Less is More for RL Scaling - only math question for RL training - This repository presents LIMR, an approach that challenges the assumption about data scaling in reinforcement learning for LLMs. We demonstrate that the quality and relevance of training samples matter far more than their quantity. Our Learning Impact Measurement (LIM) methodology enables automated evaluation of training sample effectiveness, eliminating the need for manual curation while achieving comparable or superior results with 6x less data. Notably, all our investigations are conducted directly from base models without distillation, providing clear insights into the core dynamics of RL training.
- https://huggingface.co/datasets/GAIR/LIMR

# Evaluation 
- https://github.com/huggingface/open-r1
  
