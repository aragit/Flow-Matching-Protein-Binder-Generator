# 🧬 Flow-Matching-Protein-Binder-Generator

<p align="center"><b>A minimal, deterministic implementation of Flow Matching for structural molecular generation.</b></p>

<p align="center"><sub>PyTorch · Biopython · Flow Matching Primitives · Optimal Transport</sub></p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-📐%20Active%20Blueprint-blue" alt="Active Blueprint">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Tests-12%20passing-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT">
</p>

---

## 📐 Active Blueprint

> **Architecture documented, core components stubbed, runnable skeleton.**
>
> SE(3) equivariant layers and FoldSeek API integration are interface-ready but require GPU cluster + API keys for full activation. This repo runs entirely on CPU with zero external API dependencies.

---

## 🎯 Problem Statement

Traditional diffusion models (DDPMs) require ~1000 denoising steps for protein structure generation, making them slow and stochastic. **Flow Matching** learns straight-line optimal transport paths, enabling generation in **20-50 ODE steps** with deterministic guarantees.

This engine generates high-affinity novel protein-ligand binders targeted against specific mock receptor sites using continuous normalizing flows.

---

## 🏗️ System Architecture

**Data Flow:** `Noise x_0 ~ N(0,I)` → `Flow Matching Network v_theta` → `ODE Solver (Euler/RK4/Heun)` → `Generated Binder x_1` → `Biophysical Validation`

**Components:**
- **Flow Matching Network**: Predicts the vector field that transports noise to data
- **ODE Solver**: Integrates the vector field over time t ∈ [0,1]
- **Biophysical Validator**: Checks clash scores, radius of gyration, contact maps, and fold classification
- **DDPM Baseline**: For ablation comparison (requires ~1000 steps vs 20-50 for Flow Matching)

---

## 🔬 Core Components

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| **Time Embedding** | `core/time_embedding.py` | Functional | Sinusoidal Fourier features for temporal conditioning |
| **Vector Field Network** | `core/vector_field.py` | Stub | SE(3)-aware network with Transformer placeholder; EGNN-ready interface |
| **OT Coupling** | `core/ot_coupling.py` | Interface | Optimal Transport plan computation; exact Hungarian or fast identity fallback |
| **Equivariant Layers** | `core/equivariant_layers.py` | Interface | EGNN/TFN layer stubs with documented API |
| **Probability Paths** | `sampling/path_schedules.py` | Functional | Linear and cosine conditional paths p_t(x|x_1) |
| **ODE Solvers** | `sampling/ode_solvers.py` | Functional | Euler, RK4, Heun with modular abstract base class |
| **Biophysics** | `biophysics/validator.py` | Functional | Clash score, Rg, contact maps, fold classification |
| **DDPM Baseline** | `baselines/ddpm_baseline.py` | Stub | For ablation comparison |
| **Training** | `training/losses.py`, `training/train.py` | Scaffold | Full Flow Matching loss + standalone trainer |
| **Config** | `configs/base.yaml` | Schema | Hydra-compatible YAML configuration |

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/aragit/Flow-Matching-Protein-Binder-Generator.git
cd Flow-Matching-Protein-Binder-Generator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
