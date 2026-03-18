"""
Experiment: OCR Model Comparison
---------------------------------
Compares Tesseract and EasyOCR on handwriting images to evaluate
whether OCR-extracted text is reliable enough for feature extraction
on children's handwriting.

Finding: Both engines struggled with children's handwriting
(accuracy 30-50%), leading to the decision to use image-level
CV features instead.

Usage:
    python ocr_comparison.py --image-dir ../samples/handwriting
"""

import argparse
import os
import csv
import time
from pathlib import Path

def run_tesseract(image_path):
    """Extract text using Tesseract OCR."""
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        conf_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        confidences = [int(c) for c in conf_data['conf'] if int(c) > 0]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return text.strip(), avg_conf
    except ImportError:
        return None, 0.0


def run_easyocr(image_path):
    """Extract text using EasyOCR."""
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False)
        results = reader.readtext(str(image_path))
        text = ' '.join([r[1] for r in results])
        avg_conf = sum(r[2] for r in results) / len(results) if results else 0.0
        return text.strip(), avg_conf * 100
    except ImportError:
        return None, 0.0


def compare_ocr(image_dir, output_csv='ocr_comparison_results.csv'):
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"Directory not found: {image_dir}")
        return

    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    images = [f for f in image_dir.iterdir() if f.suffix.lower() in exts]
    if not images:
        print("No images found.")
        return

    results = []
    for img_path in sorted(images):
        print(f"Processing: {img_path.name}")

        t0 = time.time()
        tess_text, tess_conf = run_tesseract(str(img_path))
        tess_time = time.time() - t0

        t0 = time.time()
        easy_text, easy_conf = run_easyocr(str(img_path))
        easy_time = time.time() - t0

        results.append({
            'image': img_path.name,
            'tesseract_text': tess_text or 'N/A',
            'tesseract_confidence': round(tess_conf, 1),
            'tesseract_time_s': round(tess_time, 2),
            'easyocr_text': easy_text or 'N/A',
            'easyocr_confidence': round(easy_conf, 1),
            'easyocr_time_s': round(easy_time, 2),
        })

    # Save results
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_csv}")
    print(f"Images processed: {len(results)}")

    # Summary
    tess_confs = [r['tesseract_confidence'] for r in results if r['tesseract_text'] != 'N/A']
    easy_confs = [r['easyocr_confidence'] for r in results if r['easyocr_text'] != 'N/A']
    if tess_confs:
        print(f"Tesseract avg confidence: {sum(tess_confs)/len(tess_confs):.1f}%")
    if easy_confs:
        print(f"EasyOCR avg confidence:   {sum(easy_confs)/len(easy_confs):.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare OCR engines on handwriting')
    parser.add_argument('--image-dir', required=True, help='Directory with handwriting images')
    parser.add_argument('--output', default='ocr_comparison_results.csv',
                        help='Output CSV file')
    args = parser.parse_args()
    compare_ocr(args.image_dir, args.output)
