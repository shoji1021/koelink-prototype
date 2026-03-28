import cv2
import mediapipe as mp
import json
import datetime
import os

print("=== KoeLink データ記録ツール ===")
# 起動時に、今回記録する手話の意味を入力してもらいます
target_label = input("今回記録する手話の意味（例: hello, thanks, こんにちは）を入力してください: ")

# 保存先フォルダの作成
SAVE_DIR = "gesture_data"
os.makedirs(SAVE_DIR, exist_ok=True)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)
is_recording = False
gesture_frames = []

print("\n===========================================")
print(f"現在の記録対象: 【 {target_label} 】")
print("[R]キー : 記録の開始 / 終了 （複数回繰り返せます）")
print("[Q]キー または [ESC] : アプリの終了")
print("===========================================")

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
) as holistic:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic.process(image_rgb)
        
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # 骨格と手の描画
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())

        # 記録処理
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

        # UI表示
        if is_recording:
            cv2.putText(image, f"RECORDING:target_label", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(image, f"Frames: {len(gesture_frames)}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(image, f"Ready:target_label (Press 'R')", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('KoeLink - Data Collector', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            is_recording = not is_recording
            if is_recording:
                print(f"\n[{target_label}] の記録を開始しました！動いてください...")
                gesture_frames = []
            else:
                # ファイル名にラベル（意味）を含めて保存
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{target_label}_{timestamp}.json"
                filepath = os.path.join(SAVE_DIR, filename)
                
                with open(filepath, 'w') as f:
                    json.dump({"label": target_label, "frames": gesture_frames}, f, indent=4)
                
                print(f"--> 保存完了: {filepath} ({len(gesture_frames)} フレーム)")

cap.release()
cv2.destroyAllWindows()