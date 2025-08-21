import os
import argparse
import numpy as np
import librosa
import sounddevice as sd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump, load
import datetime

RESULTS_PATH = "./results"
os.makedirs(RESULTS_PATH, exist_ok=True)

# -------------------------------
# Feature extraction
# -------------------------------
def extract_features(file_path, max_pad_len=174):
    try:
        audio, sr = librosa.load(file_path, res_type="kaiser_fast")
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode="constant")
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs.flatten()
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# -------------------------------
# Save realtime results
# -------------------------------
def save_prediction_result(mode, prediction, confidence, classes, probs):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_file = os.path.join(RESULTS_PATH, "realtime_results.txt")

    with open(result_file, "a") as f:
        f.write(f"[{timestamp}] Mode: {mode} | Predicted: {prediction} | Confidence: {confidence:.2f}\n")

    # Save probability plot
    plt.figure(figsize=(8, 4))
    plt.bar(classes, probs, color="skyblue" if mode == "file" else "orange")
    plt.title(f"{mode.capitalize()} Prediction (Predicted: {prediction})")
    plt.ylabel("Probability")
    plt.xticks(rotation=45)
    plt.ylim([0, 1])
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_PATH, f"{mode}_prediction_{timestamp.replace(':','-')}.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Prediction saved to {result_file}")
    print(f"Plot saved to {plot_path}")

# -------------------------------
# Dataset loading
# -------------------------------
def load_dataset(data_dir):
    X, y = [], []
    classes = []
    print("Loading dataset...")
    if not os.path.exists(data_dir):
        print(f"Dataset folder not found: {data_dir}")
        return None, None, None

    for label in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue
        classes.append(label)
        for file in os.listdir(class_dir):
            if file.endswith(".wav"):
                features = extract_features(os.path.join(class_dir, file))
                if features is not None:
                    X.append(features)
                    y.append(label)

    print(f"Loaded {len(X)} files across {len(classes)} classes: {classes}")
    return np.array(X), np.array(y), classes

# -------------------------------
# Train model
# -------------------------------
def train(data_dir, model_path):
    X, y, classes = load_dataset(data_dir)
    if X is None:
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training SVM...")
    clf = SVC(kernel="linear", probability=True)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.3f}\n")
    print("Classification report:")
    report = classification_report(y_test, y_pred)
    print(report)

    dump((clf, classes), model_path)
    print(f"Model saved as {model_path}")

    # Save training results
    with open(os.path.join(RESULTS_PATH, "training_report.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.3f}\n\n")
        f.write(report)

# -------------------------------
# Predict from file
# -------------------------------
def predict_file(model_path, audio_path):
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}. Train first with --mode train.")
        return
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return

    clf, classes = load(model_path)
    features = extract_features(audio_path)
    if features is None:
        return

    features = features.reshape(1, -1)
    pred = clf.predict(features)[0]
    probs = clf.predict_proba(features)[0]
    confidence = np.max(probs)

    print(f"File: {audio_path} → Predicted Emotion: {pred} (confidence: {confidence:.2f})")

    save_prediction_result("file", pred, confidence, classes, probs)

# -------------------------------
# Realtime prediction (mic)
# -------------------------------
def predict_mic(model_path, seconds):
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}. Train first with --mode train.")
        return

    clf, classes = load(model_path)
    print(f"Recording {seconds} seconds of audio... Speak now!")

    audio = sd.rec(int(seconds * 22050), samplerate=22050, channels=1)
    sd.wait()

    audio = audio.flatten()
    mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=40)

    if mfccs.shape[1] < 174:
        pad_width = 174 - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:
        mfccs = mfccs[:, :174]

    features = mfccs.flatten().reshape(1, -1)

    pred = clf.predict(features)[0]
    probs = clf.predict_proba(features)[0]
    confidence = np.max(probs)

    print(f"Predicted Emotion (mic): {pred} (confidence: {confidence:.2f})")

    save_prediction_result("mic", pred, confidence, classes, probs)

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["train", "file", "mic"], help="Operation mode")
    parser.add_argument("--data_dir", type=str, default="./Data", help="Dataset folder (for training)")
    parser.add_argument("--model_path", type=str, default="emotion_model.joblib", help="Path to save/load model")
    parser.add_argument("--audio_path", type=str, help="Path to audio file (for file mode)")
    parser.add_argument("--seconds", type=int, default=4, help="Recording time for mic mode")
    args = parser.parse_args()

    if args.mode == "train":
        train(args.data_dir, args.model_path)
    elif args.mode == "file":
        predict_file(args.model_path, args.audio_path)
    elif args.mode == "mic":
        predict_mic(args.model_path, args.seconds)