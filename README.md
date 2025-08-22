# Policy Gradient Active Learning for Animal Image Classification

This project applies reinforcement learning (RL) to the sample selection process in image classification. A policy gradient agent (REINFORCE) learns to actively choose the most informative images to be labeled, aiming to maximize model improvement with fewer labeled samples.

## Problem Statement
Rather than labeling all data points, this project trains a REINFORCE-based agent to identify which samples should be labeled next. The goal is to maximize classification performance while minimizing labeled data. Each step involves evaluating model uncertainty and selecting samples that contribute most to the classifier's learning progress.

## Project Structure

```
├── src/
│   ├── datasets/
│   │   ├── preprocess.py    # Dataset preprocessing utilities
│   │   └── loaders.py       # PyTorch data loaders
│   ├── models/
│   │   └── cnn.py          # ResNet18 binary classifier
│   ├── al/
│   │   └── strategies.py   # Active learning strategies (Random, Uncertainty)
│   ├── rl/
│   │   └── policy.py       # REINFORCE policy network
│   └── utils/
│       ├── logger.py       # Logging utilities
│       ├── metrics.py      # Evaluation metrics
│       └── plots.py        # Plotting and JSON utilities
├── main_week1.py           # Dataset preprocessing
├── main_week2.py           # Baseline CNN training
├── main_week3.py           # Active Learning baselines
├── main_week4.py           # REINFORCE agent
├── data/                   # Dataset storage
├── logs/                   # Training logs
├── checkpoints/            # Model checkpoints
└── outputs/                # Results and plots
```

## Setup

### Requirements
- Python ≥ 3.9
- PyTorch ≥ 2.1
- torchvision ≥ 0.16
- See `requirements.txt` for complete list

### Installation
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Edit Dataset Path
Edit the `RAW_DATASET_DIR` variable in `main_week1.py` to point to your local raw dataset:

```python
RAW_DATASET_DIR = r"C:\\Users\\sidaa\\IIT Project\\data\\dogs-vs-cats\\train"  # <- set your raw path
```

### Step 2: Run Scripts in Order

From the project root directory:

```bash
python main_week1.py    # Preprocess dataset
python main_week2.py    # Baseline CNN training + checkpoints
python main_week3.py    # Active Learning baselines (Random, Uncertainty)
python main_week4.py    # RL (REINFORCE) agent + comparison
```

## Output Structure

All outputs are saved with clear paths:

- **Logs**: `logs/weekX_*.log`
- **Checkpoints**: `checkpoints/weekX/*.pth`
- **Results**: `outputs/weekX/*.json` / `*.csv`
- **Plots**: `outputs/weekX/plots/*.png`

## Week-by-Week Plan & Implementation

### Week 1: Data Prep & Baseline Setup
- Preprocess Dogs vs Cats dataset (resize to 128×128, normalize)
- Split into labeled/unlabeled pools
- Train baseline CNN classifier
- Implement uncertainty sampling baseline

### Week 2: RL Agent (REINFORCE) Development
- Extract image embeddings using CNN
- Design state (entropy, margin), action, reward scheme
- Build REINFORCE policy network
- Begin training with classifier interaction loop

### Week 3: Evaluation & Comparisons
- Track accuracy vs # of labeled samples
- Visualize learning curves, entropy, reward trends
- Compare RL policy with random & uncertainty sampling

### Week 4: Final Report & Visualization
- Prepare notebook with evaluation plots
- Generate demo of sample selection process
- Create model card with policy insights and classifier performance

## Key Features

- **Robust preprocessing**: Handles various image formats and folder structures
- **Modular design**: Clean separation of concerns across modules
- **Comprehensive logging**: Detailed logs for debugging and monitoring
- **Visualization**: Automatic generation of training curves and comparison plots
- **Checkpointing**: Model states saved for reproducibility and analysis
- **Cross-platform**: Works on Windows, Linux, and macOS

## Configuration

Key hyperparameters can be modified in each main script:

- `INIT_LABELED`: Initial number of labeled samples (default: 1000)
- `ACQ_SIZE`: Number of samples to acquire per round (default: 500)
- `ROUNDS`: Number of acquisition rounds (default: 8)
- `EPOCHS_PER_ROUND`: Training epochs per round (default: 3)
- `BATCH`: Batch size (default: 64)
- Learning rates and other parameters

## 🎉 Complete Results - All Weeks Finished!

### **Week 1 - Data Preprocessing**: ✅ COMPLETED
- Successfully processed 25,000 images (20,000 train, 5,000 val)
- Images resized to 128×128 pixels and organized for training

### **Week 2 - Baseline CNN Training**: ✅ COMPLETED
- **Peak Performance**: **95.22%** validation accuracy 🏆
- **Training Loss**: Reduced from 0.194 to 0.057
- **Convergence**: Excellent baseline established in 5 epochs

### **Week 3 - Active Learning Baselines**: ✅ COMPLETED
- **Random Sampling**: Peak 92.8% accuracy
- **Uncertainty Sampling**: Peak **93.54%** accuracy
- **Sample Efficiency**: Both methods achieved >90% with limited data

### **Week 4 - REINFORCE Policy**: ✅ COMPLETED  
- **Peak Performance**: **93.78%** validation accuracy
- **Policy Learning**: Successfully trained REINFORCE agent
- **Sample Selection**: Learned to identify most informative samples

### **🚀 BONUS: Enhanced with LLM & Interactive UI**
- **Interactive Dashboard**: Streamlit web application
- **AI Explanations**: LLM-powered insights and recommendations
- **Real-time Classification**: Upload and classify images instantly
- **Multi-Provider LLM**: HuggingFace, Ollama, OpenAI support

## 📊 **Final Performance Comparison**

| Method | Peak Accuracy | Sample Efficiency | Best Use Case |
|--------|---------------|-------------------|---------------|
| **Baseline CNN** | **95.22%** 🏆 | Full Dataset | Maximum accuracy |
| **Uncertainty Sampling** | **93.54%** | Excellent | Consistent AL |
| **REINFORCE Policy** | **93.78%** | Excellent | Adaptive AL |
| **Random Sampling** | 92.8% | Good | Simple baseline |

**🎯 Key Achievement**: REINFORCE policy achieved competitive performance (93.78%) while learning intelligent sample selection strategies!

## 🚀 **How to Use the Enhanced Project**

### **Launch Interactive UI**:
```bash
python start_ui.py
# OR
streamlit run app.py
```

### **Test LLM Integration**:
```bash
python demo_llm.py
```

**Project Status**: **100% COMPLETE + SIGNIFICANTLY ENHANCED** 🏆