# Hate Speech Detection Project

A RoBERTa-based hate speech detection model with multi-output regression for 10 target labels.

## Project Structure

```
hatespeechdetection/
├── notebooks/          # Jupyter notebooks
├── src/               # Source code modules
├── scripts/           # Executable scripts
├── test/              # Unit tests
├── requirements.txt   # Dependencies
└── LICENSE           # MIT License
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train model:
```bash
python scripts/train_model.py --output_dir ./results
```

3. Make predictions:
```bash
python scripts/predict.py --model_path ./results
```

4. Evaluate model:
```bash
python scripts/evaluate.py --model_path ./results
```

5. Run tests:
```bash
python scripts/run_tests.py
```

## Target Labels

The model predicts scores for: sentiment, respect, insult, humiliate, status, dehumanize, violence, genocide, attack_defend, hatespeech.

## License

MIT License - see [LICENSE](LICENSE) file.
