# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies
```bash
# Option 1: Automatic installation
python install_requirements.py

# Option 2: Manual installation
pip install -r requirements.txt
```

### Step 2: Validate Setup
```bash
# Run comprehensive validation
python validate_project.py

# Or run individual checks
python check_setup.py
python test_imports.py
```

### Step 3: Configure Dataset Path
Edit `main_week1.py` and change the dataset path:
```python
RAW_DATASET_DIR = r"C:\path\to\your\dogs-vs-cats\train"  # <- Change this
```

### Step 4: Run the Project
```bash
python main_week1.py    # Data preprocessing
python main_week2.py    # Baseline CNN training  
python main_week3.py    # Active learning baselines
python main_week4.py    # REINFORCE policy training
```

### Step 5: Explore Results
- Check `outputs/` for results and plots
- Open `notebooks/demo_visualization.ipynb` for interactive analysis
- Review `model_card.md` for performance metrics

## 🔧 Troubleshooting

### Common Issues:

**Import Errors:**
```bash
python test_imports.py  # Diagnose import issues
pip install -r requirements.txt  # Reinstall dependencies
```

**Dataset Not Found:**
- Ensure dataset path in `main_week1.py` is correct
- Check that image files contain 'cat' or 'dog' in filenames

**CUDA Issues:**
- Project works with CPU-only PyTorch
- For GPU: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

**Permission Errors:**
- Run with administrator privileges on Windows
- Check write permissions for `data/`, `logs/`, `checkpoints/`, `outputs/`

### Getting Help:
1. Run `python validate_project.py` for detailed diagnostics
2. Check error logs in `logs/` directory
3. Verify all files exist with `python check_setup.py`

## 📊 Expected Output Structure

After running all scripts:
```
outputs/
├── week1/
│   └── preprocess_stats.json
├── week2/
│   ├── metrics.json
│   └── plots/
│       ├── val_acc.png
│       └── train_loss.png
├── week3/
│   ├── curves.json
│   └── plots/
│       └── baselines.png
└── week4/
    ├── rl_curve.json
    └── plots/
        └── rl_curve.png
```

## 🎯 Success Indicators

✅ **Week 1**: `preprocess_stats.json` shows train/val counts  
✅ **Week 2**: Validation accuracy > 80% after 5 epochs  
✅ **Week 3**: Learning curves show improvement over rounds  
✅ **Week 4**: REINFORCE policy shows competitive performance  

## 📈 Next Steps

1. **Analysis**: Open Jupyter notebook for detailed visualization
2. **Tuning**: Modify hyperparameters in main scripts
3. **Extension**: Add new active learning strategies
4. **Comparison**: Implement additional baseline methods