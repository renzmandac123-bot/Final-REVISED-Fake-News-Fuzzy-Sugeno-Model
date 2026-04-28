# Filipino Fake News Detection System

A Type 1 Takagi-Sugeno Zero-Order Fuzzy Inference System for detecting fake news in Filipino online news articles.

## Features

- Emotional Tone (ET) using VADER and Filipino sentiment lexicon
- Grammar Quality (GQ) using heuristic signals
- Citation Strength (CS) checking for trusted sources and attribution
- Image Emotional Provocativeness (IEP) via HSV color analysis
- Picture-Text Consistency (PTC) via image metadata matching
- Full Sugeno inference with plain-language explanation

## How to use

### Manual Input tab
1. Paste the article text into the text box
2. Optionally upload the headline image
3. Click Analyse

### URL Input tab
1. Paste a news article URL (e.g. from Rappler, Inquirer, GMA News)
2. Click Fetch and Analyse
3. The system automatically extracts the article text and headline image

## Classification labels

| Label | Sugeno score range |
|---|---|
| Real | 0.80 – 1.00 |
| Mostly Real | 0.65 – 0.79 |
| Half Real | 0.50 – 0.64 |
| Mostly Fake | 0.35 – 0.49 |
| Fake | 0.20 – 0.34 |
| Pants on Fire | 0.00 – 0.19 |

## Notes

- LanguageTool grammar checking is not available on Streamlit Cloud (no Java runtime). GQ uses heuristic signals instead.
- Image features are only computed when an image is provided or successfully extracted from the URL.
- For best results use Philippine news article URLs (Rappler, Inquirer, GMA News, PhilStar).
