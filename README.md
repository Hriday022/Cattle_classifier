# Cattle Breed Classifier — Hosted Demo

Flask app serving `cattle_classifier_ood.keras` (MobileNetV2 transfer-learning model, 15 classes, 224x224 input) with a simple upload UI and out-of-distribution flagging on low-confidence predictions.

## Before you deploy — fill in your class names

Open `app.py` and replace the placeholder `CLASS_NAMES` list with your actual
15 labels, in the exact order used during training (check your training
notebook's `class_indices` or `ImageDataGenerator`/`flow_from_directory` output).
This is the one thing I couldn't fill in without your training data.

## Run locally

```bash
cd cattle-app
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Visit `http://localhost:5000`.

## Deploy to Render (same flow as your Gym Tracker project)

1. Push this folder to a new GitHub repo (include `model/cattle_classifier_ood.keras` — use [Git LFS](https://git-lfs.com/) since it's ~28MB; GitHub is fine with it via LFS or even a normal commit since it's under 100MB).
2. On [render.com](https://render.com): **New → Web Service** → connect the repo.
3. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --timeout 120`
   - **Instance type:** free tier is fine for a demo, but TensorFlow + free tier can be slow on first load (cold start ~30-60s) — mention this if showing it live in an interview, or just link the video walkthrough as backup.
4. Deploy. Your live URL becomes your resume's "Live Demo" link.

## Deploy to Hugging Face Spaces (faster for ML demos, alternative)

If Render's cold-start bugs you, HF Spaces with a Gradio interface is often snappier
for pure model demos and is a platform recruiters recognize. Say the word and I'll
build a `gradio` version of this app instead — same model, different frontend, ~15 min swap.

## Record your video walkthrough (for the BinaryFolks submission)

1. Briefly state the problem (cattle breed classification + OOD detection).
2. Show an upload → prediction → confidence score on a real image.
3. Show one deliberately "wrong" image (e.g. a dog or random object) to demonstrate the OOD flag working.
4. One slide/spoken line on architecture: MobileNetV2 backbone (frozen/fine-tuned base) + dense head, accuracy achieved.
5. Keep it under 5 minutes total.
