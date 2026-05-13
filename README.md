# KAN-TRaCE: Kolmogorov-Arnold Network-based Temporal Reasoning System with Chain-of-Explanation

## 🌟 Vision
Traditional AI in medical imaging often operates as a "black box," providing results without logical justification. **KAN-TRaCE** redefines this by building a "glass-box" system where every clinical decision is traceable, quantifiable, and explainable through a **Chain-of-Explanation (CoE)**.

## ⚠️ The Problem: "Dynamic Hallucination"
In temporal Chest X-ray analysis (comparing scans over time), AI often misinterprets changes in patient posture, breathing phases, or imaging angles as actual disease progression. KAN-TRaCE addresses this "dynamic hallucination" by integrating non-rigid spatial registration and physics-informed modeling.

## 🏗️ System Architecture & Pipeline

### 1. Data Engineering (Phase 0)
* **Unified Metadata:** Synchronizing disparate sources (MIMIC-CXR, Chest ImaGenome, RadGraph) into a flattened, high-fidelity `[dataset]_metadata.jsonl` ecosystem.
* **Standardization:** Automated image resizing to 512x512 with synchronized bounding box (BBox) scaling.

### 2. Multimodal Alignment & Grounding (Phases 1-3)
* **Contrastive Pre-training:** Aligning Image (BioViL-T) and Text (CXR-BERT) embeddings using InfoNCE and BCE loss to capture semantic medical nuances.
* **Open-vocabulary Grounding:** Localizing abnormalities by extracting anatomic attributes and relationships from clinical reports using **Llama-3.1** and **ChEX**.
* **Hybrid KAN Refinement:** Utilizing **Kolmogorov-Arnold Networks** with B-spline activation functions to refine BBox coordinates and provide uncertainty scores ($\sigma$), outperforming traditional MLP-based detection.

### 3. Temporal Reasoning & Registration (Phases 3.5-4)
* **Non-rigid Registration:** Employing **VoxelMorph** to compute Deformable Vector Fields (DVF), aligning the "Prior" scan to the "Current" scan's anatomic axis.
* **T-KAN Modeling:** A novel **Temporal KAN** module that processes patches via RoIAlign to quantify physical variations:
    * **$\Delta$ Morphology:** Change in lesion size/shape.
    * **$\Delta$ Radiomics:** Change in opacity/density.

### 4. Grounded Chain-of-Thought (Phase 5)
* **CoT Generation:** A fine-tuned LVLM translates T-KAN embeddings into natural language.
* **Consistency Loss:** A safety-net mechanism that forces the LLM to strictly adhere to physical measurements, ensuring the AI "speaks the truth" rather than hallucinating medical text.

## 🔬 Academic Contributions
* **First-of-its-kind** application of KANs for refining medical coordinates and modeling temporal disease trajectories.
* **Elimination of artifacts** through integrated DVF warping before comparative analysis.
* **Interpretable Logic:** Moves beyond simple classification to provide a step-by-step diagnostic rationale.

## 🛠️ Technical Stack
| Category | Tools & Frameworks |
| :--- | :--- |
| **Core AI** | PyTorch, Hugging Face, BioViL-T, CXR-BERT |
| **Architectures** | KAN (Kolmogorov-Arnold), T-KAN, VoxelMorph, Llama-3.1 |
| **Data Ops** | Python 3.10+, Docker, Rclone, Registry-based Loaders |
| **Deployment** | Gradio (Interactive Demo), FastAPI |

## 📊 Evaluation Metrics
* **Spatial:** IoU (Intersection over Union).
* **Temporal:** F1-Score for progression classification.
* **Linguistic:** RadGraph-F1, BLEU for report quality.
