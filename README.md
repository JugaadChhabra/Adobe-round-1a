# Multimodal Heading Classification

This project provides a multimodal deep learning pipeline for heading classification in documents, combining visual (image) and textual (semantic and numeric) features.

## Project Structure
- `multimodal_dataset.py`: Dataset class for extracting and encoding image and text features from PDFs and JSON annotations.
- `multimodal_model.py`: Neural network model and training function.
- `main.py`: Example script to train the model.
- `requirements.txt`: List of dependencies.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare your PDF and JSON annotation files.
3. Update the paths in `main.py` to point to your data.

## Usage
Run the training script:
```bash
python main.py
```

## Notes
- The model uses both visual and semantic features for robust heading classification.
- Make sure your JSON annotations match the expected format (see code for details).

 
## HOW TO RUN 
# Build the image
docker build -t pdf-heading-classifier .

# Run with interactive mode
docker run -it pdf-heading-classifier

# Run specific scripts
docker run -it pdf-heading-classifier python predict_pdf_headings.py