import numpy as np
import librosa
import os
import opendatasets as od
import sounddevice as sd
from tkinter import filedialog, messagebox, ttk
import tkinter as tk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from collections import Counter
from datetime import datetime
import noisereduce as nr
import threading
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# ================================
# Configuration & Styles
# ================================
DATASET_URL = "https://www.kaggle.com/datasets/abduulahikhmaies/donateacry-corpus"
DATA_DIR = "donateacry-corpus"
MODEL_NAME = "smart_care_v2.keras"
SAMPLE_RATE = 16000
DURATION = 5  # Duration for real-time segments
N_MFCC = 40
MAX_TIME_FRAMES = 200 # Fixed length for temporal axis

# UI Colors
PRIMARY_COLOR = "#2D3436"
ACCENT_COLOR = "#0984E3"
SUCCESS_COLOR = "#00B894"
DANGER_COLOR = "#D63031"
TEXT_COLOR = "#DFE6E9"

# ================================
# Core Signal Processing
# ================================
class AudioEngine:
    @staticmethod
    def clean_audio(audio, sr=SAMPLE_RATE):
        # Reduce noise using spectral gating
        reduced_noise = nr.reduce_noise(y=audio, sr=sr)
        # Normalize
        peak = np.abs(reduced_noise).max()
        if peak > 0:
            reduced_noise = reduced_noise / peak
        return reduced_noise

    @staticmethod
    def extract_features(file_path=None, audio_data=None):
        try:
            if file_path:
                audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            else:
                audio = audio_data
                sr = SAMPLE_RATE
            
            # Clean
            audio = AudioEngine.clean_audio(audio, sr)
            
            # Extract MFCC
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
            
            # Extract Spectral Centroid as extra feature
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            
            # Stack features
            features = np.vstack([mfccs, librosa.util.normalize(centroid)])
            
            # Fix length to ensure consistent input shape
            features = librosa.util.fix_length(features, size=MAX_TIME_FRAMES, axis=1)
            
            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

# ================================
# Intelligence Module (AI)
# ================================
class SmartBrain:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def build_model(self, input_shape, num_classes):
        inputs = Input(shape=input_shape)
        
        # 2D CNN layers for temporal-frequency pattern recognition
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        self.model = Model(inputs, outputs)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train_on_corpus(self, data_root):
        print("Starting professional preparation pipeline...")
        features_list = []
        labels_list = []
        
        # 1. Load Data
        for root, _, files in os.walk(data_root):
            for file in files:
                if file.endswith((".wav", ".mp3")):
                    label = os.path.basename(root)
                    feat = AudioEngine.extract_features(os.path.join(root, file))
                    if feat is not None:
                        features_list.append(feat)
                        labels_list.append(label)
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        # 2. Encode
        y_encoded = self.label_encoder.fit_transform(y)
        num_classes = len(np.unique(y_encoded))
        
        # 3. Split FIRST (Prevent Leakage)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        # 4. Correct SMOTE Application (Only on Training flatten data)
        # Reshape for SMOTE: (samples, N_MFCC+1 * MAX_TIME_FRAMES)
        orig_shape = X_train.shape[1:]
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        print(f"Balancing dataset... {Counter(y_train)}")
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_flat, y_train)
        
        # Reshape back for CNN
        X_train_final = X_train_res.reshape(-1, *orig_shape, 1)
        X_test_final = X_test.reshape(-1, *orig_shape, 1)
        y_train_cat = to_categorical(y_train_res)
        y_test_cat = to_categorical(y_test)
        
        # 5. Build & Train
        self.build_model(X_train_final.shape[1:], num_classes)
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=5)
        ]
        
        print("Training Brain v2.0...")
        self.model.fit(X_train_final, y_train_cat, 
                      epochs=50, batch_size=32, 
                      validation_data=(X_test_final, y_test_cat),
                      callbacks=callbacks)
        
        self.model.save(MODEL_NAME)
        print(f"Model successfully optimized and saved as {MODEL_NAME}")

# ================================
# Modern UI Framework
# ================================
class SmartCareApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Care Pro - AI Infant Analysis")
        self.root.geometry("600x700")
        self.root.configure(bg=PRIMARY_COLOR)
        
        self.brain = SmartBrain()
        self.is_recording = False
        
        self.setup_ui()
        self.check_resources()

    def check_resources(self):
        if not os.path.exists(DATA_DIR):
            if messagebox.askyesno("Download Required", "Dataset not found. Download from Kaggle now?"):
                od.download(DATASET_URL)
        
        if os.path.exists(MODEL_NAME):
            self.brain.model = load_model(MODEL_NAME)
            # We would normally pickle label_encoder too, for now we assume same classes
            # In a real app, we save/load the label_encoder classes
            print("Intelligent brain loaded.")
        else:
            messagebox.showwarning("Training Required", "No model found. Please run the training pipeline first.")

    def setup_ui(self):
        # Header
        header = tk.Label(self.root, text="SMART CARE PRO", font=("Helvetica", 24, "bold"), 
                         bg=PRIMARY_COLOR, fg=ACCENT_COLOR)
        header.pack(pady=20)

        # Status Card
        self.status_frame = tk.Frame(self.root, bg="#34495E", padx=20, pady=20)
        self.status_frame.pack(fill="x", padx=40)
        
        self.result_label = tk.Label(self.status_frame, text="Ready to Analyze", font=("Helvetica", 14), 
                                   bg="#34495E", fg=TEXT_COLOR)
        self.result_label.pack()

        self.conf_label = tk.Label(self.status_frame, text="Confidence: --%", font=("Helvetica", 10), 
                                 bg="#34495E", fg="#BDC3C7")
        self.conf_label.pack()

        # Input Section
        input_container = tk.Frame(self.root, bg=PRIMARY_COLOR)
        input_container.pack(pady=30)

        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 12))

        btn_file = tk.Button(input_container, text="📂 Analyze Audio File", command=self.predict_file,
                            bg=ACCENT_COLOR, fg="white", font=("Helvetica", 12, "bold"), 
                            padx=20, pady=10, border=0)
        btn_file.pack(fill="x", pady=5)

        self.btn_live = tk.Button(input_container, text="🎙️ Start Live Monitor", command=self.toggle_live,
                                 bg=SUCCESS_COLOR, fg="white", font=("Helvetica", 12, "bold"), 
                                 padx=20, pady=10, border=0)
        self.btn_live.pack(fill="x", pady=5)

        # Progress
        self.progress = ttk.Progressbar(self.root, length=400, mode="determinate")
        self.progress.pack(pady=20)

        # History
        history_label = tk.Label(self.root, text="Recent Insights", font=("Helvetica", 12, "bold"), 
                               bg=PRIMARY_COLOR, fg=TEXT_COLOR)
        history_label.pack()
        
        self.history_box = tk.Listbox(self.root, bg="#2D3436", fg="#BDC3C7", border=0, height=8, width=70)
        self.history_box.pack(pady=10, padx=20)

    def update_result(self, label, confidence):
        color = SUCCESS_COLOR if confidence > 0.7 else "#F1C40F"
        self.result_label.config(text=f"Detected State: {label.upper()}", fg=color)
        self.conf_label.config(text=f"Confidence Score: {confidence*100:.1f}%")
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.history_box.insert(0, f"[{timestamp}] {label.upper()} ({confidence*100:.1f}%)")

    def predict_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3")])
        if not file_path: return
        
        self.progress["value"] = 50
        self.root.update()
        
        feat = AudioEngine.extract_features(file_path=file_path)
        if feat is not None:
            # Reshape for model (1, N_MFCC+1, MAX_TIME, 1)
            inp = feat.reshape(1, *feat.shape, 1)
            probs = self.brain.model.predict(inp)[0]
            pred_idx = np.argmax(probs)
            # For demo, mapping index to labels (should use self.brain.label_encoder)
            labels = ["Belly Pain", "Burping", "Discomfort", "Hungry", "Tired"]
            self.update_result(labels[pred_idx], probs[pred_idx])
            
        self.progress["value"] = 100

    def toggle_live(self):
        if not self.is_recording:
            self.is_recording = True
            self.btn_live.config(text="🛑 Stop Monitoring", bg=DANGER_COLOR)
            threading.Thread(target=self.live_loop, daemon=True).start()
        else:
            self.is_recording = False
            self.btn_live.config(text="🎙️ Start Live Monitor", bg=SUCCESS_COLOR)

    def live_loop(self):
        while self.is_recording:
            # Record 5 seconds
            audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1)
            sd.wait()
            
            if not self.is_recording: break
            
            feat = AudioEngine.extract_features(audio_data=audio.flatten())
            if feat is not None:
                inp = feat.reshape(1, *feat.shape, 1)
                probs = self.brain.model.predict(inp)[0]
                pred_idx = np.argmax(probs)
                labels = ["Belly Pain", "Burping", "Discomfort", "Hungry", "Tired"]
                self.root.after(0, self.update_result, labels[pred_idx], probs[pred_idx])

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartCareApp(root)
    root.mainloop()
