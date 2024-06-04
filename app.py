from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import cv2
import json
import mediapipe as mp


def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    hand_landmarks_list = []

    with mp_hands.Hands(
        min_detection_confidence=0.8, min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Process the image and find hands
            results = hands.process(image)

            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_landmarks_coords = []
                    for lm in hand_landmarks.landmark:
                        h, w, _ = image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        hand_landmarks_coords.extend([cx, cy])
                    hand_landmarks_list.append(hand_landmarks_coords)

    cap.release()
    return hand_landmarks_list


def preprocess_input(input_sequence, num_features):
    input_sequence = np.array([input_sequence])
    input_sequence = pad_sequences(
        input_sequence, padding="post", maxlen=num_features, value=0.0
    )
    return input_sequence


def top_k_sampling(preds, k=5):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def predict_sequence(
    input_sequence, tokenizer, max_seq_length, model, num_features, top_k=5
):
    input_sequence = preprocess_input(input_sequence, num_features)
    target_sequence = np.zeros((1, 1))
    target_sequence[0, 0] = tokenizer.word_index.get("<START>", 0)

    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens = model.predict([input_sequence, target_sequence])
        sampled_token_index = top_k_sampling(output_tokens[0, -1, :], k=top_k)
        sampled_word = tokenizer.index_word.get(sampled_token_index, "<OOV>")

        if sampled_word == "<END>" or len(decoded_sentence.split()) > max_seq_length:
            stop_condition = True
        else:
            decoded_sentence += sampled_word + " "
            target_sequence = np.pad(
                target_sequence,
                ((0, 0), (0, 1)),
                mode="constant",
                constant_values=sampled_token_index,
            )

    return decoded_sentence.strip()


# Exemple d'utilisation de la fonction de prédiction
# input_sequence = features_val[0]  # Séquence d'entrée prétraitée
# predicted_sequence = predict_sequence(input_sequence, tokenizer, max_seq_length=42, model=model, num_features=42)
# print(Prédiction :', predicted_sequence)

app = Flask(__name__)


@app.route("/")
def index():
    return "<h1> Api en cours d'exécution"


# Charger le modèle
model = load_model("wale.h5")

# Charger le tokenizer
with open("tokenizer.json", "r") as f:
    data = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.dumps(data))


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


@app.route("/predict", methods=['POST'])
def predict():
    data = request.json
    features = preprocess_video(data["input_sequence"])
    features = np.array([np.array(feature) for feature in features])

    # max_sequence_length = max([len(elt) for elt in features])

    # newValue = pad_sequences(features, padding='post', value=0.0, maxlen=50)

    predicted_sequence = predict_sequence(
        features, tokenizer, max_seq_length=42, model=model, num_features=42
    )
    return jsonify(predicted_sequence)


if __name__ == "__main__":
    app.run()
