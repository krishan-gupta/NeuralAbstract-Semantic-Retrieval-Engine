# Qualitative retrieval checks

Generated: 2026-11-01T11:24:02.284331+00:00
Corpus: `papers.db`; dense index: `faiss_local.index`; sparse index: `bm25_index`.
Cutoff: top 10 per method.

Overlap is agreement between ranked lists, not retrieval accuracy. Review the titles below to judge whether Hybrid adds useful results.

## medical

- Overlap: **40.0%** (4 shared; 10 Hybrid results; 10 BM25 results).
- Titles found only by Hybrid:
  - Large Language Models Need Holistically Thought in Medical Conversational QA
  - Analyzing Medical Data with Process Mining: a COVID-19 Case Study
  - Artificial intelligence in medicine and healthcare: a review and classification of current and near-future applications and their ethical and social Impact
  - Applying Language Models in Clinical Medicine: Recent Trends and Perspectives
  - Large language models in medicine: the potentials and pitfalls
  - Advances in Large Language Models for Medicine
- Titles found only by BM25:
  - MedGen: Unlocking Medical Video Generation by Scaling Granularly-annotated Medical Videos
  - Apollo: A Lightweight Multilingual Medical LLM towards Democratizing Medical AI to 6B People
  - COGNET-MD, an evaluation framework and dataset for Large Language Model benchmarks in the medical domain
  - When Medical Safety Alignment Fails: A Benchmark for Evaluating LLMs on High-Risk Medical Queries
  - MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making
  - Grounded Knowledge-Enhanced Medical Vision-Language Pre-training for Chest X-Ray

## LoRA

- Overlap: **50.0%** (5 shared; 10 Hybrid results; 10 BM25 results).
- Titles found only by Hybrid:
  - ALLoRA: Adaptive Learning Rate Mitigates LoRA Fatal Flaws
  - LoRA-XS: Low-Rank Adaptation with Extremely Small Number of Parameters
  - Each Rank Could be an Expert: Single-Ranked Mixture of Experts LoRA for Multi-Task Learning
  - SEQR: Secure and Efficient QR-based LoRA Routing
  - \k{appa}-LoRA: Condition Numbers Reveal Which LoRA Matrices Worth Updating
- Titles found only by BM25:
  - KD-LoRA: A Hybrid Approach to Efficient Fine-Tuning with LoRA and Knowledge Distillation
  - Uni-LoRA: One Vector is All You Need
  - Quantum-Inspired Fine-Tuning for Few-Shot AIGC Detection via Phase-Structured Reparameterization
  - DR-LoRA: Dynamic Rank LoRA for Fine-Tuning Mixture-of-Experts Models
  - Tied-Lora: Enhancing parameter efficiency of LoRA with weight tying

## making large models run on phones

- Overlap: **60.0%** (6 shared; 10 Hybrid results; 10 BM25 results).
- Titles found only by Hybrid:
  - iOS as Acceleration
  - OpenPhone: Mobile Agentic Foundation Models
  - PhoneLM:an Efficient and Capable Small Language Model Family through Principled Pre-training
  - Tiny-QMoE
- Titles found only by BM25:
  - Edge Phoneme Recognition for Children's Speech through Age-Aware Training
  - Self-Supervised Speech Models Encode Phonetic Context via Position-dependent Orthogonal Subspaces
  - Knowledge Distillation of Transformer-based Language Models Revisited
  - Adopting Two Supervisors for Efficient Use of Large-Scale Remote Deep Neural Networks

## teaching AI to say it doesn't know

- Overlap: **50.0%** (5 shared; 10 Hybrid results; 10 BM25 results).
- Titles found only by Hybrid:
  - Abstraction, Validation, and Generalization for Explainable Artificial Intelligence
  - Metrics for Explainable AI: Challenges and Prospects
  - TED: Teaching AI to Explain its Decisions
  - Scope and Sense of Explainability for AI-Systems
  - From Neural Activations to Concepts: A Survey on Explaining Concepts in Neural Networks
- Titles found only by BM25:
  - HistoLens: An Interactive XAI Toolkit for Verifying and Mitigating Flaws in Vision-Language Models for Histopathology
  - PretrainRL: Alleviating Factuality Hallucination of Large Language Models at the Beginning
  - One Size doesn't Fit All: A Personalized Conversational Tutoring Agent for Mathematics Instruction
  - KnowRL: Teaching Language Models to Know What They Know
  - Human-AI Interaction Design in Machine Teaching

## reducing hallucinations in chatbots

- Overlap: **20.0%** (2 shared; 10 Hybrid results; 10 BM25 results).
- Titles found only by Hybrid:
  - From "Hallucination" to "Suture": Insights from Language Philosophy to Enhance Large Language Models
  - Mental Health Assessment for the Chatbots
  - Leveraging Large Language Models to Power Chatbots for Collecting User Self-Reported Data
  - Artificial Empathy: AI based Mental Health
  - Some hypotheses on how chatbots work in problem-solution-driven conversations: Large Language Models as confirmation of the Innovation Illusion
  - Chatbots to ChatGPT in a Cybersecurity Space: Evolution, Vulnerabilities, Attacks, Challenges, and Future Recommendations
  - Towards Healthy AI: Large Language Models Need Therapists Too
  - Towards Standard Criteria for human evaluation of Chatbots: A Survey
- Titles found only by BM25:
  - HonkaiChat: Companions from Anime that feel alive!
  - Assistive Chatbots for healthcare: a succinct review
  - A Proposal for Evaluating the Operational Risk for ChatBots based on Large Language Models
  - Supporting Student Decisions on Learning Recommendations: An LLM-Based Chatbot with Knowledge Graph Contextualization for Conversational Explainability and Mentoring
  - From Textbook to Talkbot: A Case Study of a Greek-Language RAG-Based Chatbot in Higher Education
  - Hallucination Diversity-Aware Active Learning for Text Summarization
  - Reducing Hallucinations in Neural Machine Translation with Feature Attribution
  - ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability
