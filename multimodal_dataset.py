import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import json
import fitz  # PyMuPDF
import numpy as np
from transformers import AutoTokenizer, AutoModel
import re
from sklearn.preprocessing import LabelEncoder
import difflib
import io
import random

FIXED_CLASSES = ['title', 'h1', 'h2', 'h3', 'none']

class MultiModalHeadingDataset(Dataset):
    def __init__(self, pdf_paths, json_paths, transform=None, image_size=(224, 224), match_threshold=0.6):
        self.pdf_paths = pdf_paths
        self.json_paths = json_paths
        self.transform = transform
        self.image_size = image_size
        self.data = []
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.text_encoder = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.heading_patterns = {
            'title': ['title', 'main', 'document', 'report', 'company', 'organization'],
            'h1': ['chapter', 'section', 'part', 'introduction', 'conclusion', 'overview', 'summary'],
            'h2': ['subsection', 'methodology', 'results', 'discussion', 'analysis', 'background'],
            'h3': ['subheading', 'details', 'specifications', 'features', 'benefits', 'requirements']
        }
        self.valid_levels = set(FIXED_CLASSES) - {'none'}
        self.numeric_feature_keys = [
            'text_length', 'word_count', 'has_numbers', 'has_special_chars', 'is_uppercase',
            'is_title_case', 'starts_with_number', 'semantic_title_score', 'semantic_h1_score',
            'semantic_h2_score', 'semantic_h3_score', 'relative_x_position', 'relative_y_position',
            'text_width_ratio', 'is_centered', 'document_position', 'headings_before', 'headings_after', 'font_size', 'is_bold'
        ]
        self.match_threshold = match_threshold
        self._load_data()

    def _load_data(self):
        all_labels = []
        all_samples = []
        for pdf_path, json_path in zip(self.pdf_paths, self.json_paths):
            samples, labels = self._process_pdf_json_pair(pdf_path, json_path)
            all_samples.extend(samples)
            all_labels.extend(labels)

        heading_samples, heading_labels = [], []
        none_samples, none_labels = [], []

        for sample, label in zip(all_samples, all_labels):
            if label == 'none':
                none_samples.append(sample)
                none_labels.append(label)
            else:
                heading_samples.append(sample)
                heading_labels.append(label)

        upsample_factor = max(1, len(none_samples) // max(len(heading_samples), 1))
        heading_samples_upsampled = heading_samples * upsample_factor
        heading_labels_upsampled = heading_labels * upsample_factor

        final_samples = heading_samples_upsampled + none_samples
        final_labels = heading_labels_upsampled + none_labels

        combined = list(zip(final_samples, final_labels))
        random.shuffle(combined)
        final_samples, final_labels = zip(*combined) if combined else ([], [])

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(FIXED_CLASSES)

        for sample, label in zip(final_samples, final_labels):
            encoded_label = self.label_encoder.transform([label])[0]
            self.data.append((sample, encoded_label))

    def _process_pdf_json_pair(self, pdf_path, json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error in file: {json_path}")
            print(e)
            raise e

        pdf_doc = fitz.open(pdf_path)
        samples = []
        labels = []

        annos_by_page = {}
        for anno in annotations:
            page = anno['page'] - 1
            level = anno['level'].lower()
            if level not in self.valid_levels:
                continue
            annos_by_page.setdefault(page, []).append({'text': anno['text'], 'level': level})

        full_text = "".join(page.get_text() for page in pdf_doc)

        for page_num, page in enumerate(pdf_doc):
            page_annos = annos_by_page.get(page_num, [])
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block['type'] != 0:
                    continue
                for line in block['lines']:
                    for span in line['spans']:
                        x0, y0, x1, y1 = span['bbox']
                        text = span['text'].strip()
                        if not text or len(text) < 3:
                            continue

                        matched_label = 'none'
                        def normalize(t):
                            return re.sub(r'\s+', ' ', t.strip().lower())

                        for anno in page_annos:
                            ratio = difflib.SequenceMatcher(None, normalize(text), normalize(anno['text'])).ratio()
                            if ratio > self.match_threshold:
                                matched_label = anno['level']
                                break
                        text_rect = (x0, y0, x1, y1)
                        region_image = self._extract_text_region(page, text_rect)
                        text_features = self._extract_text_features(text, page, text_rect, full_text, span)

                        if region_image is not None and text_features is not None:
                            sample = {
                                'image': region_image,
                                'text_features': text_features,
                                'raw_text': text
                            }
                            samples.append(sample)
                            labels.append(matched_label)

        pdf_doc.close()
        return samples, labels

    def _extract_text_features(self, text, page, text_rect, full_document_text, span):
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'has_numbers': bool(re.search(r'\d', text)),
            'has_special_chars': bool(re.search(r'[^\w\s]', text)),
            'is_uppercase': text.isupper(),
            'is_title_case': text.istitle(),
            'starts_with_number': text.strip()[0].isdigit() if text.strip() else False
        }

        text_lower = text.lower()
        for level, keywords in self.heading_patterns.items():
            features[f'semantic_{level}_score'] = sum(1 for keyword in keywords if keyword in text_lower) / len(keywords)

        x0, y0, x1, y1 = text_rect
        page_width, page_height = page.rect.width, page.rect.height
        features.update({
            'relative_x_position': x0 / page_width,
            'relative_y_position': y0 / page_height,
            'text_width_ratio': (x1 - x0) / page_width,
            'is_centered': abs((x0 + x1) / 2 - page_width / 2) < page_width * 0.1
        })

        text_position = full_document_text.find(text)
        if text_position != -1:
            context_window = 200
            features['document_position'] = text_position / len(full_document_text)
            before = full_document_text[max(0, text_position - context_window):text_position]
            after = full_document_text[text_position + len(text):text_position + len(text) + context_window]
            features['headings_before'] = len(re.findall(r'^[A-Z][A-Za-z\s]+$', before, re.MULTILINE))
            features['headings_after'] = len(re.findall(r'^[A-Z][A-Za-z\s]+$', after, re.MULTILINE))
        else:
            features['document_position'] = 0.0
            features['headings_before'] = 0
            features['headings_after'] = 0

        features['font_size'] = span['size']
        features['is_bold'] = 'bold' in span['font'].lower()

        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                outputs = self.text_encoder(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            features['semantic_embedding'] = embeddings
        except Exception as e:
            print(f"Embedding error for text: {text} -> {e}")
            features['semantic_embedding'] = np.zeros(384)

        return features

    def _extract_text_region(self, page, text_rect, padding=20):
        x0, y0, x1, y1 = text_rect
        x0 = max(0, x0 - padding)
        y0 = max(0, y0 - padding)
        x1 = min(page.rect.width, x1 + padding)
        y1 = min(page.rect.height, y1 + padding)
        clip_rect = fitz.Rect(x0, y0, x1, y1)

        # Validate dimensions
        if clip_rect.is_empty or clip_rect.width <= 0 or clip_rect.height <= 0:
            print(f"Skipping invalid region: {clip_rect}")
            return None

        try:
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat, clip=clip_rect)
            img_data = pix.tobytes("ppm")
            img = Image.open(io.BytesIO(img_data)).resize(self.image_size)
            return img
        except Exception as e:
            print(f"Failed to extract image region for rect {clip_rect}: {e}")
            return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, label = self.data[idx]
        image = sample['image']
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        text_features = sample['text_features']
        numeric_features = [
            float(text_features.get(key, 0.0)) for key in self.numeric_feature_keys
        ]
        numeric_features_tensor = torch.tensor(numeric_features, dtype=torch.float32)
        raw_text = sample['raw_text']
        return {
            'image': image,
            'numeric_features': numeric_features_tensor,
            'raw_text': raw_text
        }, label 