# **Awesome NeurIPS 2024 Molecular ML Paper Collection**

This repo contains a comprehensive compilation of **molecular ML** papers that were accepted at the [Thirty-Eighth Annual Conference on Neural Information Processing Systems 2024](https://neurips.cc/). Molecular machine learning (Molecular ML) leverages AI to predict chemical properties and accelerate drug discovery, enabling faster, cost-effective advancements in healthcare and materials science. 

**Short Overview**: We've got ~110 papers focusing on molecular ML in NeurIPS'24. This year's NeurIPS emphasize the convergence of machine learning, molecular modeling, and biological sciences, showcasing innovations across generative models, optimization, and representation learning. Key focus areas include **protein language modeling**, **3D molecular generation**, **molecular property prediction**, and **graph-based approaches** for molecular dynamics and design. These works advance techniques such as diffusion models, geometric learning, and multi-modal transfer learning to address challenges in **drug discovery**, **protein engineering**, and **single-cell genomics**, paving the way for faster, more accurate predictions and designs in molecular biology and chemistry.


**Have a look and throw me a review (and, a star ⭐, maybe!)** Thanks!


---



## **All Topics:** 

<details open>
  <summary><b>View Topic list!</b></summary>

- [Molecule Generation](#Generative)
  - [Diffusion Models](#Diffusion)
      - [Graph Diffusion](#GraphDiffusion)
  - [Graph, Geometry and GNN Models](#GNN-Gen)
  - [Flow-Matching](#Flow-Matching)
  - [GFlowNets](#GFlowNets)
  - [Others](#Others-gen)
- [Multi-modal Models](#Multi-modal)
- [Interactions](#Interactions)
- [Single-cell Application Works](#Single-cell)
- [Graphs and GNNs](#ggnns)
- [Protein Language Models](#PLMs)
- [Property Prediction and Optimization](#|Property)
- [RNA](#RNA)
- [3D Modeling and Representation Learning](#3D)
- [Others](#Others)
</details>



<a name="Generative" />

## Generative Modeling

<a name="Diffusion" />

### Diffusion Models
- [Bridging Model-Based Optimization and Generative Modeling via Conservative Fine-Tuning of Diffusion Models](https://openreview.net/pdf?id=zIr2QjU4hl)
- [Immiscible Diffusion: Accelerating Diffusion Training with Noise Assignment](https://openreview.net/pdf?id=kK23oMGe9g)
- [DEFT: Efficient Fine-tuning of Diffusion Models by Learning the Generalised $h$-transform](https://openreview.net/pdf?id=AKBTFQhCjm)
- [Equivariant Blurring Diffusion for Hierarchical Molecular Conformer Generation](https://openreview.net/pdf?id=Aj0Zf28l6o)
- [Aligning Target-Aware Molecule Diffusion Models with Exact Energy Optimization](https://openreview.net/pdf?id=EWcvxXtzNu)
- [Full-Atom Peptide Design with Geometric Latent Diffusion](https://openreview.net/pdf?id=IAQNJUJe8q)
- [Geometric Trajectory Diffusion Models](https://openreview.net/pdf?id=OYmms5Mv9H)
- [Capturing the denoising effect of PCA via compression ratio](https://openreview.net/pdf?id=a4J7nDLXEM)
- [Reprogramming Pretrained Target-Specific Diffusion Models for Dual-Target Drug Design](https://openreview.net/pdf?id=Y79L45D5ts)

<a name="GraphDiffusion" />

#### Graph Diffusion Models
- [Equivariant Neural Diffusion for Molecule Generation](https://openreview.net/pdf?id=40pE5pFhWl)
- [Graph Diffusion Policy Optimization](https://openreview.net/pdf?id=8ohsbxw7q8)
- [Graph Diffusion Transformers for Multi-Conditional Molecular Generation](https://openreview.net/pdf?id=cfrDLD1wfO)
- [Discrete-state Continuous-time Diffusion for Graph Generation](https://openreview.net/pdf?id=YkSKZEhIYt)
- [Pard: Permutation-Invariant Autoregressive Diffusion for Graph Generation](https://openreview.net/pdf?id=x4Kk4FxLs3)
- [Diffusion Twigs with Loop Guidance for Conditional Graph Generation](https://openreview.net/pdf?id=fvOCJAAYLx)
- [SubgDiff: A Subgraph Diffusion Model to Improve Molecular Representation Learning](https://openreview.net/pdf?id=iSMTo0toDO)


<a name="GNN-Gen" />

### Graph, Geometry and GNN Models
- [Variational Flow Matching for Graph Generation](https://openreview.net/pdf?id=UahrHR5HQh)
- [Unified Guidance for Geometry-Conditioned Molecular Generation](https://openreview.net/pdf?id=HeoRsnaD44)

### Flow-Matching
- [Score-based 3D molecule generation with neural fields](https://openreview.net/pdf?id=9lGJrkqJUw)
- [Molecule Generation with Fragment Retrieval Augmentation](https://openreview.net/pdf?id=56Q0qggDlp)
- [Generalized Protein Pocket Generation with Prior-Informed Flow Matching](https://openreview.net/pdf?id=WyVTj77KEV)
- [ET-Flow: Equivariant Flow-Matching for Molecular Conformer Generation](https://openreview.net/pdf?id=avsZ9OlR60)
- [Sequence-Augmented SE(3)-Flow Matching For Conditional Protein Generation](https://openreview.net/pdf?id=paYwtPBpyZ)
- [Fisher Flow Matching for Generative Modeling over Discrete Data](https://openreview.net/pdf?id=6jOScqwdHU)
- [Generating Highly Designable Proteins with Geometric Algebra Flow Matching](https://openreview.net/pdf?id=nAnEStxyfy)

### GFlowNets
- [RGFN: Synthesizable Molecular Generation Using GFlowNets](https://openreview.net/pdf?id=hpvJwmzEHX)
- [Genetic-guided GFlowNets for Sample Efficient Molecular Optimization](https://openreview.net/pdf?id=B4q98aAZwt)
- [Pessimistic Backward Policy for GFlowNets](https://openreview.net/pdf?id=L8Q21Qrjmd)
- [On Divergence Measures for Training GFlowNets](https://openreview.net/pdf?id=N5H4z0Pzvn)

<a name="Others-gen" />

### Others
- [MSA Generation with Seqs2Seqs Pretraining: Advancing Protein Structure Predictions](https://openreview.net/pdf?id=D0DLlMOufv)
- [QVAE-Mole: The Quantum VAE with Spherical Latent Variable Learning for 3-D Molecule Generation](https://openreview.net/pdf?id=RqvesBxqDo)

<a name="Multi-modal" />

## Multi-modal Models
- [Multi-modal Transfer Learning between Biological Foundation Models](https://openreview.net/pdf?id=xImeJtdUiw)
- [MMSite: A Multi-modal Framework for the Identification of Active Sites in Proteins](https://openreview.net/pdf?id=XHdwlbNSVb)

## Interactions
- [Iteratively Refined Early Interaction Alignment for Subgraph Matching based Graph Retrieval](https://openreview.net/pdf?id=udTwwF7tks)
- [Neural P$^3$M: A Long-Range Interaction Modeling Enhancer for Geometric GNNs](https://openreview.net/pdf?id=ncqauwSyl5)
- [Customized Subgraph Selection and Encoding for Drug-drug Interaction Prediction](https://openreview.net/pdf?id=crlvDzDPgM)

## Single-cell
- [Semi-supervised Knowledge Transfer Across Multi-omic Single-cell Data](https://openreview.net/pdf?id=sKEhebkEdz)
- [GENOT: Entropic (Gromov) Wasserstein Flow Matching with Applications to Single-Cell Genomics](https://openreview.net/pdf?id=hjspWd7jvg)
- [Gene-Gene Relationship Modeling Based on Genetic Evidence for Single-Cell RNA-Seq Data Imputation](https://openreview.net/pdf?id=gW0znG5JCG)
- [Absorb & Escape: Overcoming Single Model Limitations in Generating Heterogeneous Genomic Sequences](https://openreview.net/pdf?id=XHTl2k1LYk)

<a name="ggnns" />

## Graphs and GNNs
- [Equivariant Neural Diffusion for Molecule Generation](https://openreview.net/pdf?id=40pE5pFhWl)
- [Graph Diffusion Policy Optimization](https://openreview.net/pdf?id=8ohsbxw7q8)
- [Graph Diffusion Transformers for Multi-Conditional Molecular Generation](https://openreview.net/pdf?id=cfrDLD1wfO)
- [Discrete-state Continuous-time Diffusion for Graph Generation](https://openreview.net/pdf?id=YkSKZEhIYt)
- [Pard: Permutation-Invariant Autoregressive Diffusion for Graph Generation](https://openreview.net/pdf?id=x4Kk4FxLs3)
- [Diffusion Twigs with Loop Guidance for Conditional Graph Generation](https://openreview.net/pdf?id=fvOCJAAYLx)
- [SubgDiff: A Subgraph Diffusion Model to Improve Molecular Representation Learning](https://openreview.net/pdf?id=iSMTo0toDO)
- [Any2Graph: Deep End-To-End Supervised Graph Prediction With An Optimal Transport Loss](https://openreview.net/pdf?id=tPgagXpvcV)
- [On the Scalability of GNNs for Molecular Graphs](https://openreview.net/pdf?id=klqhrq7fvB)
- [GFT: Graph Foundation Model with Transferable Tree Vocabulary](https://openreview.net/pdf?id=0MXzbAv8xy)
- [Temporal Graph Neural Tangent Kernel with Graphon-Guaranteed](https://openreview.net/pdf?id=266nH7kLSV)
- [Empowering Active Learning for 3D Molecular Graphs with Geometric Graph Isomorphism](https://openreview.net/pdf?id=He2GCHeRML)
- [Variational Flow Matching for Graph Generation](https://openreview.net/pdf?id=UahrHR5HQh)
- [LLaMo: Large Language Model-based Molecular Graph Assistant](https://openreview.net/pdf?id=WKTNdU155n)
- [Enhancing Graph Transformers with Hierarchical Distance Structural Encoding](https://openreview.net/pdf?id=U4KldRgoph)


<a name="PLMs" />

## Protein Language Models
- [Training Compute-Optimal Protein Language Models](https://openreview.net/pdf?id=uCZI8gSfD4)
- [DePLM: Denoising Protein Language Models for Property Optimization](https://openreview.net/pdf?id=MU27zjHBcW)
- [Ultrafast classical phylogenetic method beats large protein language models on variant effect prediction](https://openreview.net/pdf?id=H7mENkYB2J)
- [MutaPLM: Protein Language Modeling for Mutation Explanation and Engineering](https://openreview.net/pdf?id=yppcLFeZgy)
- [ProSST: Protein Language Modeling with Quantized Structure and Disentangled Attention](https://openreview.net/pdf?id=4Z7RZixpJQ)
- [Data-Driven Discovery of Dynamical Systems in Pharmacology using Large Language Models](https://openreview.net/pdf?id=KIrZmlTA92)

<a name="Property" />

## Property Prediction and Optimization
- [DePLM: Denoising Protein Language Models for Property Optimization](https://openreview.net/pdf?id=MU27zjHBcW)
- [Instructor-inspired Machine Learning for Robust Molecular Property Prediction](https://openreview.net/pdf?id=j7sw0nXLjZ)
- [Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction](https://openreview.net/pdf?id=859DtlwnAD)

## RNA
- [Gene-Gene Relationship Modeling Based on Genetic Evidence for Single-Cell RNA-Seq Data Imputation](https://openreview.net/pdf?id=gW0znG5JCG)

<a name="3D" />

## 3D Modeling and Representation Learning
- [S-MolSearch: 3D Semi-supervised Contrastive Learning for Bioactive Molecule Search](https://openreview.net/pdf?id=wJAF8TGVUG)
- [Empowering Active Learning for 3D Molecular Graphs with Geometric Graph Isomorphism](https://openreview.net/pdf?id=He2GCHeRML)
- [Score-based 3D molecule generation with neural fields](https://openreview.net/pdf?id=9lGJrkqJUw)
- [Conditional Synthesis of 3D Molecules with Time Correction Sampler](https://openreview.net/pdf?id=gipFTlvfF1)
- [S-MolSearch: 3D Semi-supervised Contrastive Learning for Bioactive Molecule Search](https://openreview.net/pdf?id=wJAF8TGVUG)
- [A probability contrastive learning framework for 3D molecular representation learning](https://openreview.net/pdf?id=HYiR6tGQPv)



# Others
- [Generative Modeling of Molecular Dynamics Trajectories](https://openreview.net/pdf?id=yRRCH1OsGW)
- [Approximation-Aware Bayesian Optimization](https://openreview.net/pdf?id=t7euV5dl5M)
- [MSAGPT: Neural Prompting Protein Structure Prediction via MSA Generative Pre-Training](https://openreview.net/pdf?id=pPeXYByHNd)
- [Direct Preference-Based Evolutionary Multi-Objective Optimization with Dueling Bandits](https://openreview.net/pdf?id=owHj0G15cd)
- [Quadratic Quantum Variational Monte Carlo](https://openreview.net/pdf?id=lDtABI541U)
- [TurboHopp: Accelerated Molecule Scaffold Hopping with Consistency Models](https://openreview.net/pdf?id=lBh5kuuY1L)
- [Multi-Scale Representation Learning for Protein Fitness Prediction](https://openreview.net/pdf?id=kWMVzIdCEn)
- [Kermut: Composite kernel regression for protein variant effects](https://openreview.net/pdf?id=jM9atrvUii)
- [Contrastive losses as generalized models of global epistasis](https://openreview.net/pdf?id=hLoiXOzoly)
- [Foundation Inference Models for Markov Jump Processes](https://openreview.net/pdf?id=f4v7cmm5sC)
- [CryoGEM: Physics-Informed Generative Cryo-Electron Microscopy](https://openreview.net/pdf?id=edOZifvwMi)
- [Implicitly Guided Design with PropEn: Match your Data to Follow the Gradient](https://openreview.net/pdf?id=dhFHO90INk)
- [Molecule Design by Latent Prompt Transformer](https://openreview.net/pdf?id=dg3tI3c2B1)
- [DeltaDock: A Unified Framework for Accurate, Efficient, and Physically Reliable Molecular Docking](https://openreview.net/pdf?id=dao67XTSPd)
- [UniIF: Unified Molecule Inverse Folding](https://openreview.net/pdf?id=clqX9cVDKV)
- [Learning Macroscopic Dynamics from Partial Microscopic Observations](https://openreview.net/pdf?id=cjH0Qsgd0D)
- [Extracting Training Data from Molecular Pre-trained Models](https://openreview.net/pdf?id=cV4fcjcwmz)
- [MatrixNet: Learning over symmetry groups using learned group representations](https://openreview.net/pdf?id=b8jwgZrAXG)
- [Cell ontology guided transcriptome foundation model](https://openreview.net/pdf?id=aeYNVtTo7o)
- [Navigating Chemical Space with Latent Flows](https://openreview.net/pdf?id=aAaV4ZbQ9j)
- [The Importance of Being Scalable: Improving the Speed and Accuracy of Neural Network Interatomic Potentials Across Chemical Domains](https://openreview.net/pdf?id=Y4mBaZu4vy)
- [Protein-Nucleic Acid Complex Modeling with Frame Averaging Transformer](https://openreview.net/pdf?id=Xngi3Z3wkN)
- [When is an Embedding Model  More Promising than Another?](https://openreview.net/pdf?id=VqFz7iTGcl)
- [Mixture of neural fields for heterogeneous reconstruction in cryo-EM](https://openreview.net/pdf?id=TuspoNzIdB)
- [Inversion-based Latent Bayesian Optimization](https://openreview.net/pdf?id=TrN5TcWY87)
- [From Biased to Unbiased Dynamics: An Infinitesimal Generator Approach](https://openreview.net/pdf?id=TGmwp9jJXl)
- [b"Doob's Lagrangian: A Sample-Efficient Variational Approach to Transition Path Sampling"](https://openreview.net/pdf?id=ShJWT0n7kX)
- [Bridge-IF: Learning Inverse Protein Folding with Markov Bridges](https://openreview.net/pdf?id=Q8yfhrBBD8)
- [Towards Stable Representations for Protein Interface Prediction](https://openreview.net/pdf?id=OEWBkLrRZu)
- [Enhancing Protein Mutation Effect Prediction through a Retrieval-Augmented Framework](https://openreview.net/pdf?id=LgeHswiWef)
- [How Molecules Impact Cells: Unlocking Contrastive PhenoMolecular Retrieval](https://openreview.net/pdf?id=LQBlSGeOGm)
- [Double-Ended Synthesis Planning with Goal-Constrained Bidirectional Search](https://openreview.net/pdf?id=LJNqVIKSCr)
- [Association Pattern-aware Fusion for Biological Entity Relationship Prediction](https://openreview.net/pdf?id=LI5KmimXbM)
- [Deep Homomorphism Networks](https://openreview.net/pdf?id=KXUijdMFdG)
- [Contrastive dimension reduction: when and how?](https://openreview.net/pdf?id=IgU8gMKy4D)
- [Neural Pfaffians: Solving Many Many-Electron Schrodinger Equations](https://openreview.net/pdf?id=HRkniCWM3E)
- [Approximating mutual information of high-dimensional variables using learned representations](https://openreview.net/pdf?id=HN05DQxyLl)
- [Physical Consistency Bridges Heterogeneous Data in Molecular Multi-Task Learning](https://openreview.net/pdf?id=GnF9tavqgc)
- [Beyond Efficiency: Molecular Data Pruning for Enhanced Generalization](https://openreview.net/pdf?id=GJ0qIevGjD)
- [Hybrid Generative AI for De Novo Design of Co-Crystals with Enhanced Tabletability](https://openreview.net/pdf?id=G4vFNmraxj)
- [Neural Network Reparametrization for Accelerated Optimization in Molecular Simulations](https://openreview.net/pdf?id=FwxOHl0BEl)
- [Revisiting K-mer Profile for Effective and Scalable Genome Representation Learning](https://openreview.net/pdf?id=Ehsd856Ltb)
- [HORSE: Hierarchical Representation for Large-Scale Neural Subset Selection](https://openreview.net/pdf?id=DONsOc7rY1)
- [Learning Identifiable Factorized Causal Representations of Cellular Responses](https://openreview.net/pdf?id=AhlaBDHMQh)
- [Pruning neural network models for gene regulatory dynamics using data and domain knowledge](https://openreview.net/forum?id=FNtsZLwkGr)
- [Transferable Boltzmann Generators](https://openreview.net/pdf?id=AYq6GxxrrY)
- [Persistent Homology for High-dimensional Data Based on Spectral Methods](https://openreview.net/pdf?id=ARV1gJSOzV)
- [Exploring Molecular Pretraining Model at Scale](https://openreview.net/pdf?id=64V40K2fDv)
- [On the Adversarial Robustness of Benjamini Hochberg](https://openreview.net/pdf?id=5jYFoldunM)
- [FlexSBDD: Structure-Based Drug Design with Flexible Protein Modeling](https://openreview.net/pdf?id=4AB54h21qG)
- [Generative Adversarial Model-Based Optimization via Source Critic Regularization](https://openreview.net/pdf?id=3RxcarQFRn)
- [CryoSPIN: Improving Ab-Initio Cryo-EM Reconstruction with Semi-Amortized Pose Inference](https://openreview.net/pdf?id=1MCseWaFZb)
- [AdaNovo: Towards Robust De Novo Peptide Sequencing in Proteomics against Data Biases](https://openreview.net/pdf?id=0zfUiSX5si)
- [ProtGO: Function-Guided Protein Modeling for Unified Representation Learning](https://openreview.net/pdf?id=0oUutV92YF)
- [Learning Complete Protein Representation by Dynamically Coupling of Sequence and Structure](https://openreview.net/pdf?id=0e5uOaJxo1)


---


**Missing any paper?**
If any paper is absent from the list, please feel free to [mail](mailto:azminetoushik.wasi@gmail.com) or [open an issue](https://github.com/azminewasi/Awesome-MoML-NeurIPS24/issues/new/choose) or submit a pull request. I'll gladly add that! Also, If I mis-categorized, please knock!

---

## More Collectons:
- [**Awesome NeurIPS 2024 Graph Paper Collection**](https://github.com/azminewasi/Awesome-Graph-Research-NeurIPS2024)
- [**Awesome ICML 2024 Graph Paper Collection**](https://github.com/azminewasi/Awesome-Graph-Research-ICML2024)
- [**Awesome ICLR 2024 Graph Paper Collection**](https://github.com/azminewasi/Awesome-Graph-Research-ICLR2024)
- [**Awesome-LLMs-ICLR-24**](https://github.com/azminewasi/Awesome-LLMs-ICLR-24/)

---

## ✨ **Credits**
**Azmine Toushik Wasi**

 [![website](https://img.shields.io/badge/-Website-blue?style=flat-square&logo=rss&color=1f1f15)](https://azminewasi.github.io) 
 [![linkedin](https://img.shields.io/badge/LinkedIn-%320beff?style=flat-square&logo=linkedin&color=1f1f18)](https://www.linkedin.com/in/azmine-toushik-wasi/) 
 [![kaggle](https://img.shields.io/badge/Kaggle-%2320beff?style=flat-square&logo=kaggle&color=1f1f1f)](https://www.kaggle.com/azminetoushikwasi) 
 [![google-scholar](https://img.shields.io/badge/Google%20Scholar-%2320beff?style=flat-square&logo=google-scholar&color=1f1f18)](https://scholar.google.com/citations?user=X3gRvogAAAAJ&hl=en) 
 [![facebook](https://img.shields.io/badge/Facebook-%2320beff?style=flat-square&logo=facebook&color=1f1f15)](https://www.facebook.com/cholche.gari.zatrabari/)
