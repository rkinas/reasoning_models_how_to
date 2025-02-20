# Thinking model and RLHF research notes

## RL overview
- [Reinforcement Learning: An Overview](https://arxiv.org/pdf/2412.05265)
- [A COMPREHENSIVE SURVEY OF LLM ALIGNMENT TECHNIQUES: RLHF, RLAIF, PPO, DPO AND MORE](https://arxiv.org/pdf/2407.16216)
- [Book-Mathematical-Foundation-of-Reinforcement-Learning](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)
- [The FASTEST introduction to Reinforcement Learning on the internet](https://www.youtube.com/watch?v=VnpRp7ZglfA)
- [rlhf-book](https://github.com/natolambert/rlhf-book)

## Methods for LLM training
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

## Tutorials
Notes for learning RL: Value Iteration -> Q Learning -> DQN -> REINFORCE -> Policy Gradient Theorem -> TRPO -> PPO
- [CS234: Reinforcement Learning Winter 2025 ](https://web.stanford.edu/class/cs234/)
- [CS285 Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/)
- [Welcome to Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/index.html)
- [deep-rl-course from Huggingface](https://huggingface.co/learn/deep-rl-course/unit0/introduction)
- [RL Course by David Silver](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-)


## RL explained
- [GRPO vs PPO](https://yugeten.github.io/posts/2025/01/ppogrpo/)

## Training frameworks
- [VERL](https://github.com/volcengine/verl)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [TRL](https://huggingface.co/docs/trl/)

## RLHF methods implementation (only with detailed explanations)
- [GRPO A.Burkov](https://github.com/aburkov/theLMbook/blob/main/GRPO.py)

## Articles
- [On the Emergence of Thinking in LLMs I: Searching for the Right Intuition](https://arxiv.org/pdf/2502.06773)
- [LIMR: Less is More for RL Scaling](https://arxiv.org/pdf/2502.11886)
- [LIMO: Less Is More for Reasoning](https://github.com/GAIR-NLP/LIMO)
- [s1: Simple test-time scaling](https://github.com/simplescaling/s1) and s1.1 
- [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- [Online-DPO-R1: Unlocking Effective Reasoning Without the PPO Overhead](https://efficient-unicorn-451.notion.site/Online-DPO-R1-Unlocking-Effective-Reasoning-Without-the-PPO-Overhead-1908b9a70e7b80c3bc83f4cf04b2f175) and [github](https://github.com/RLHFlow/Online-DPO-R1)
- [a reinforcement learning guide](https://naklecha.notion.site/a-reinforcement-learning-guide)
- [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)

# Thinking process

## Papers
- [SelfCite: Self-Supervised Alignment for Context Attribution in Large Language Models](https://arxiv.org/abs/2502.09604)
- [ReasonFlux: Hierarchical LLM Reasoning via Scaling Thought Templates](https://arxiv.org/abs/2502.06772)
- [A Minimalist Approach to Offline Reinforcement Learning](https://arxiv.org/abs/2106.06860)
- [Training Language Models to Reason Efficiently](https://arxiv.org/abs/2502.04463)
- [Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search](https://arxiv.org/abs/2502.02508)

## Datasets - thinking models
- [Y] https://huggingface.co/datasets/open-r1/OpenR1-Math-220k
- [Y] https://huggingface.co/datasets/simplescaling/s1K-1.1
- [Y] https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k
- [Y] https://huggingface.co/datasets/GAIR/LIMO
- [Y] https://huggingface.co/datasets/AI-MO/NuminaMath-CoT

# Evaluation 
- https://github.com/huggingface/open-r1
  
