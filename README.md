# Bahraini Currency Detection
Build an AI system that understands Bahraini currency (BHD)

## Project structure
```
notebook/       money.ipynb
streamlit/      money.py
train-7/        weights/best.pt
```
| File | Purpose |
|---|---|
| `notebook/money.ipynb` | Building the model with Neural Networks and YOLO |
| `streamlit/money.py` | Streamlit app: upload a Bahraini Currency image to recieve its predicting lable |
| `train-7/weights/best.pt` | Trained YOLO model weights used by the application to make predictions. |

## How It Works

1. Upload an image containing Bahraini currency.

2. The YOLO model processes the image.

3. The system identifies the detected currency denominations.

4. The predicted labels and confidence scores are displayed.

## Live Demo



