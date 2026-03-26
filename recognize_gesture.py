import cv2
import mediapipe as mp
import json
import os
import numpy as np

# 保存先フォルダ（先ほどデータを記録したフォルダ）
SAVE_DIR = "gesture_data"

# --- 1. データの形状を整える関数 ---
# 手話は毎回スピードが違うため、強制的に「30フレーム」の長さに統一（リサンプリング）して比較しやすくします
def process_frames(frames, target_length=30):
    if len(frames) == 0:
        return np.zeros((target_length, 1))

    processed_seq = []
    for frame in frames:
        feature_vector = []
        # 今回は特に重要な「右手」「左手」「上半身」の順番で座標を1列に並べます
        for part in ["right_hand", "left_hand", "pose"]:
            if part in frame and len(frame[part]) > 0:
                for lm in frame[part]:
                    feature_vector.extend([lm["x"], lm["y"], lm["z"]])
            else:
                # 手が画面に映っていない時は0で埋める
                feature_vector.extend([0.0, 0.0, 0.0] * (21 if "hand" in part else 33))
        processed_seq.append(feature_vector)
    
    processed_seq = np.array(processed_seq)
    
    # フレーム数をターゲット（30）に合わせる
    indices = np.linspace(0, len(processed_seq) - 1, target_length).astype(int)
    return processed_seq[indices]

# --- 2. 保存されたテンプレートデータの読み込み ---
templates = []
print("=== 保存された手話データを読み込み中 ===")
if os.path.exists(SAVE_DIR):
    for filename in os.listdir(SAVE_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(SAVE_DIR, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                label = data.get("label", "unknown")
                frames = data.get("frames", [])
                
                # データを比較しやすい形に変換して保存
                processed_data = process_frames(frames)
                templates.append({"label": label, "data": processed_data})
                print(f"読み込み完了: {filename} (ラベル: {label})")

if len(templates) == 0:
    print("エラー: 比較するデータがありません。先にデータ記録ツールでJSONを作成してください。")
    exit()

# --- 3. リアルタイム判定システム ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
is_recording = False
gesture_frames = []
result_label = "Waiting..."

print("\n=== KoeLink 手話判定プロトタイプ ===")
print("[R]キー : 記録の開始 / 終了 （終了した瞬間に判定します）")
print("[Q]キー または [ESC] : アプリの終了")
print("===================================")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # 骨格の描画（今回は判定メインなので少しシンプルに描画）
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # 記録中の処理
        if is_recording:
            frame_data = {"pose": [], "left_hand": [], "right_hand": []}
            if results.pose_landmarks:
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    frame_data["pose"].append({"id": i, "x": lm.x, "y": lm.y, "z": lm.z})
            if results.left_hand_landmarks:
                for i, lm in enumerate(results.left_hand_landmarks.landmark):
                    frame_data["left_hand"].append({"id": i, "x": lm.x, "y": lm.y, "z": lm.z})
            if results.right_hand_landmarks:
                for i, lm in enumerate(results.right_hand_landmarks.landmark):
                    frame_data["right_hand"].append({"id": i, "x": lm.x, "y": lm.y, "z": lm.z})
            gesture_frames.append(frame_data)
            
            cv2.putText(image, "RECORDING...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        else:
            # 待機中＆結果表示
            cv2.putText(image, f"Result: {result_label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(image, "Press 'R' to record your gesture", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('KoeLink - Recognizer', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            is_recording = not is_recording
            
            if is_recording:
                # 記録開始
                gesture_frames = []
                result_label = "Thinking..."
            else:
                # 記録終了 -> 判定処理の開始
                if len(gesture_frames) > 5: # あまりに短い動きは無視
                    target_data = process_frames(gesture_frames)
                    
                    best_match_label = "Unknown"
                    min_distance = float('inf')
                    
                    # 保存されている全データと「距離（ズレ）」を計算して一番近いものを探す
                    for template in templates:
                        # ユークリッド距離（2つの波形のズレの大きさ）を計算
                        distance = np.linalg.norm(template["data"] - target_data)
                        
                        if distance < min_distance:
                            min_distance = distance
                            best_match_label = template["label"]
                    
                    result_label = best_match_label
                    print(f"判定結果: {result_label} (距離: {min_distance:.2f})")
                else:
                    result_label = "Too short"

cap.release()
cv2.destroyAllWindows()