import torch
from torch.utils.data import DataLoader
from multimodal_model import MultiModalHeadingClassifier
import torchvision.transforms as transforms
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import json
from transformers import AutoTokenizer, AutoModel
import pickle
import os
import glob

PDF_DIR = r'C:\Users\Jugaad\Documents\Python\nn_heading\pdfs\pdfs'
# To predict for a single PDF, set PDF_FILENAME below:
PDF_FILENAME = '80.pdf'  # Change this to any PDF in the folder
PDF_PATH = os.path.join(PDF_DIR, PDF_FILENAME)
OUTPUT_JSON = f'{os.path.splitext(PDF_FILENAME)[0]}_predicted_headings.json'

# To predict for all PDFs in the folder, uncomment the following lines:
# pdf_files = glob.glob(os.path.join(PDF_DIR, '*.pdf'))
# for pdf_path in pdf_files:
#     PDF_PATH = pdf_path
#     OUTPUT_JSON = f'{os.path.splitext(os.path.basename(pdf_path))[0]}_predicted_headings.json'
#     ... (rest of the script inside this loop) ...

# Load label encoder from pickle file
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Model and tokenizer
num_classes = len(label_encoder.classes_)
model = MultiModalHeadingClassifier(num_classes)
model.load_state_dict(torch.load('best_multimodal_heading_classifier.pth', map_location='cpu'))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Text encoder for semantic embedding
text_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
text_encoder = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Numeric feature keys (must match training)
numeric_feature_keys = [
    'text_length', 'word_count', 'has_numbers', 'has_special_chars', 'is_uppercase',
    'is_title_case', 'starts_with_number', 'semantic_title_score', 'semantic_h1_score',
    'semantic_h2_score', 'semantic_h3_score', 'relative_x_position', 'relative_y_position',
    'text_width_ratio', 'is_centered', 'document_position', 'headings_before', 'headings_after', 'font_size', 'is_bold'
]

heading_patterns = {
    'title': ['title', 'main', 'document', 'report', 'company', 'organization'],
    'h1': ['chapter', 'section', 'part', 'introduction', 'conclusion', 'overview', 'summary'],
    'h2': ['subsection', 'methodology', 'results', 'discussion', 'analysis', 'background'],
    'h3': ['subheading', 'details', 'specifications', 'features', 'benefits', 'requirements']
}

def extract_text_features(text, page, text_rect, full_document_text, span):
    """
    Extracts a set of numeric features from the text and its context.
    (This function is duplicated from the dataset class for standalone prediction)
    """
    features = {}
    features['text_length'] = len(text)
    features['word_count'] = len(text.split())
    import re
    features['has_numbers'] = bool(re.search(r'\d', text))
    features['has_special_chars'] = bool(re.search(r'[^\w\s]', text))
    features['is_uppercase'] = text.isupper()
    features['is_title_case'] = text.istitle()
    features['starts_with_number'] = text.strip()[0].isdigit() if text.strip() else False
    text_lower = text.lower()
    for level, keywords in heading_patterns.items():
        features[f'semantic_{level}_score'] = sum(
            1 for keyword in keywords if keyword in text_lower
        ) / len(keywords)
    x0, y0, x1, y1 = text_rect
    page_width = page.rect.width
    page_height = page.rect.height
    features['relative_x_position'] = x0 / page_width
    features['relative_y_position'] = y0 / page_height
    features['text_width_ratio'] = (x1 - x0) / page_width
    features['is_centered'] = abs((x0 + x1) / 2 - page_width / 2) < page_width * 0.1
    text_position = full_document_text.find(text)
    if text_position != -1:
        doc_progress = text_position / len(full_document_text)
        features['document_position'] = doc_progress
        context_window = 200
        before_context = full_document_text[max(0, text_position-context_window):text_position]
        after_context = full_document_text[text_position+len(text):text_position+len(text)+context_window]
        features['headings_before'] = len(re.findall(r'^[A-Z][A-Za-z\s]+$', before_context, re.MULTILINE))
        features['headings_after'] = len(re.findall(r'^[A-Z][A-Za-z\s]+$', after_context, re.MULTILINE))
    else:
        features['document_position'] = 0.0
        features['headings_before'] = 0
        features['headings_after'] = 0
    features['font_size'] = span['size']
    features['is_bold'] = 'bold' in span['font'].lower()
    try:
        inputs = text_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = text_encoder(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            features['semantic_embedding'] = embeddings
    except:
        features['semantic_embedding'] = np.zeros(384)
    return features

def extract_text_region(page, text_rect, padding=20):
    x0, y0, x1, y1 = text_rect
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(page.rect.width, x1 + padding)
    y1 = min(page.rect.height, y1 + padding)
    clip_rect = fitz.Rect(x0, y0, x1, y1)
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat, clip=clip_rect)
    import io
    img_data = pix.tobytes("ppm")
    img = Image.open(io.BytesIO(img_data))
    img = img.resize((224, 224))
    return img

results = []
pdf_doc = fitz.open(PDF_PATH)
full_text = "".join([page.get_text() for page in pdf_doc])

for page_num, page in enumerate(pdf_doc, 1):
    blocks = page.get_text("dict")["blocks"]
    for block in blocks:
        if block['type'] == 0: # 0 indicates a text block
            for line in block['lines']:
                for span in line['spans']:
                    x0, y0, x1, y1 = span['bbox']
                    text = span['text'].strip()
                    if not text:
                        continue
                    if len(text) < 3:
                        continue
                    # Extract features for the current text span.
                    text_rect = (x0, y0, x1, y1)
                    region_image = extract_text_region(page, text_rect)
                    image = transform(region_image).unsqueeze(0).to(device)
                    text_features = extract_text_features(text, page, text_rect, full_text, span)
                    numeric_features = [float(text_features.get(key, 0.0)) for key in numeric_feature_keys]
                    numeric_features_tensor = torch.tensor(numeric_features, dtype=torch.float32).unsqueeze(0).to(device)
                    # Make a prediction with the model.
                    with torch.no_grad():
                        outputs = model(image, numeric_features_tensor, [text])
                        probs = torch.softmax(outputs, dim=1)
                        _, predicted = torch.max(probs, 1)
                        predicted_label = label_encoder.inverse_transform(predicted.cpu().numpy())[0]
                    print(f"Predicted: {predicted_label} | Text: {text[:50]}")
                    # Save the predicted heading to the results list.
                    if predicted_label != 'none':
                        results.append({
                            'level': predicted_label,
                            'text': text,
                            'page': page_num
                        })

pdf_doc.close()

# Save the predicted headings to a JSON file.
with open(OUTPUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)

print(f'Predicted headings saved to {OUTPUT_JSON}')