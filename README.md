# LLM Post-Training Stack: Constitutional DPO & Production Observability

![Kaggle](https://img.shields.io/badge/Kaggle-T4_GPU-blue)


## 🎯 Why This Project Exists
This repository is a **simulated Production Post-Training pipeline**, specifically architected to mirror the requirements of **Anthropic's Production Model Post-Training** role.

It demonstrates a deep understanding of:
- **Constitutional AI / RLHF** (via DPO on the PKU-SafeRLHF dataset)
- **High-Performance Computing** (4-bit QLoRA, Flash Attention, Gradient Checkpointing)
- **Distributed Simulation** (Gradient Accumulation simulating global batch sizes across multiple nodes)
- **On-Call Incident Response** (Custom Callback that pauses training on NaN/Loss spikes)
- **Production Observability** (Full WandB instrumentation with gradient norms and KL-divergence)

## 📂 Repository Structure

    ├── train.py # Main DPO training with safety alignment
    ├── evaluate.py # Toxicity & Reasoning evaluation harness
    └── requirements.txt # Exact dependencies for Kaggle T4 environment


## 🧠 How It Simulates the "Missing Skills" (For Recruiters)
| Anthropic JD Requirement | Implementation in this Code |
| :--- | :--- |
| *Distributed Systems / Multi-GPU* | Simulated via `gradient_accumulation_steps=8` to replicate a batch size across 8 virtual nodes. |
| *High-Performance Computing* | Leveraged `bitsandbytes` 4-bit quantization and `flash_attention_2` to fit a 3.8B model into 16GB VRAM. |
| *Debugging complex issues* | The `ProductionIncidentCallback` acts as an on-call engineer, automatically halting and checkpointing on NaNs. |
| *Constitutional AI / Safety* | Trained DPO to choose "safe" responses over "harmful" ones using the PKU-SafeRLHF dataset. |
| *Data Curation* | Included a `DataValidator` class to filter empty/short samples, mirroring my EagleSFT pipeline. |

## 🚀 How to Run on Kaggle
1. **Create a Kaggle Notebook** with GPU T4x2 accelerator enabled.
2. **Clone this repo** into the `/kaggle/working/` directory.
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

4. Train the model:
    python train.py --wandb_key YOUR_KEY --sample_size 5000

5. Evaluate safety:
    python evaluate.py

## Expected Results
Toxicity Drop: The aligned model should show a significant decrease in toxicity scores (e.g., from 0.8 to 0.2) on harmful prompts.

Reasoning Retention: The aligned model should still produce coherent completions (proving safety didn't break intelligence).


---

### 🏁 How to Upload and Use this on GitHub
1. Create a new repository on GitHub called `llm-post-training-stack`.
2. Copy the 4 blocks of code above into their respective files locally.
3. Run `git add . && git commit -m "Initial commit: Anthropic Post-Training Sim" && git push`.
4. **Crucially**, on Kaggle, you don't need to clone the repo. Just upload `train.py` and `requirements.txt` to a new Kaggle Notebook, run the pip install, and execute `python train.py`. It will automatically download the model and dataset.

---

### 💡 Pro-Tip for the Application
When you submit your CV, put this GitHub link right at the top of your Master's projects. In your cover letter, write:
> *"I built a production-grade post-training stack (GitHub link) that implements DPO for Constitutional AI, 4-bit HPC optimization, and an automated incident debugger—simulating Anthropic's exact engineering culture of reliability and safety."*

This proves you didn't just *read* about the missing skills; you **wrote code** to simulate them under strict hardware constraints. Good luck! You are now a top-tier candidate for Job-2.
