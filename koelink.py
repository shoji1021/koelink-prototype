import cv2
import mediapipe as mp
import json
import datetime
import os
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
import threading

SAVE_DIR = "gesture_data"
os.makedirs(SAVE_DIR, exist_ok=True)

# 閾値
RECOGNITION_THRESHOLD = 19.0

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# データ
def process_frames(frames, target_length=30):
    if len(frames) == 0:
        return np.zeros((target_length, 1))

    processed_seq = []
    for frame in frames:
        feature_vector = []
        for part in ["right_hand", "left_hand", "pose"]:
            if part in frame and len(frame[part]) > 0:
                for lm in frame[part]:
                    feature_vector.extend([lm["x"], lm["y"], lm["z"]])
            else:
                feature_vector.extend([0.0, 0.0, 0.0] * (21 if "hand" in part else 33))
        processed_seq.append(feature_vector)

    processed_seq = np.array(processed_seq)
    indices = np.linspace(0, len(processed_seq) - 1, target_length).astype(int)
    return processed_seq[indices]

# データの読み込み
def load_templates():
    templates = []
    if os.path.exists(SAVE_DIR):
        for filename in os.listdir(SAVE_DIR):
            if filename.endswith(".json"):
                filepath = os.path.join(SAVE_DIR, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        label = data.get("label", "unknown")
                        frames = data.get("frames", [])
                        processed_data = process_frames(frames)
                        templates.append({"label": label, "data": processed_data})
                except:
                    pass
    return templates

# 記録
def record_gesture(target_label, callback):
    cap = cv2.VideoCapture(0)
    is_recording = False
    gesture_frames = []
    record_count = 0
    save_status = "Ready"
    saved_filenames = []

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

            # 骨格描画
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

            # UI
            cv2.putText(image, f"Label: {target_label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if is_recording:
                cv2.putText(image, "RECORDING...", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(image, f"Frames: {len(gesture_frames)}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(image, f"[R] to record  [Q] to exit", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, f"Saved: {record_count} times", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 100), 2)

            cv2.putText(image, save_status, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow('KoeLink - Recording', image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                is_recording = not is_recording
                if is_recording:
                    gesture_frames = []
                    save_status = "Recording..."
                else:
                    if len(gesture_frames) > 5:
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{target_label}_{timestamp}.json"
                        filepath = os.path.join(SAVE_DIR, filename)

                        with open(filepath, 'w') as f:
                            json.dump({"label": target_label, "frames": gesture_frames}, f, indent=4)

                        record_count += 1
                        saved_filenames.append(filename)
                        save_status = f"Saved! ({len(gesture_frames)} frames)"
                    else:
                        save_status = "Too short! Try again."

    cap.release()
    cv2.destroyAllWindows()
    callback(saved_filenames)

# --- 4. 認識モード ---
def recognize_gesture(callback):
    templates = load_templates()

    if len(templates) == 0:
        messagebox.showerror("Error", "テンプレートがありません。\n先にジェスチャーを登録してください。")
        callback("error")
        return

    cap = cv2.VideoCapture(0)
    is_recording = False
    gesture_frames = []
    result_label = "Waiting..."
    result_distance = 0

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

            # 骨格の描画
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

                cv2.putText(image, "RECORDING...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                cv2.putText(image, f"Result: {result_label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.putText(image, f"Distance: {result_distance:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(image, "[R] to recognize  [Q] to exit", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('KoeLink - Recognizer', image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                is_recording = not is_recording

                if is_recording:
                    gesture_frames = []
                    result_label = "Recording..."
                    result_distance = 0
                else:
                    if len(gesture_frames) > 5:
                        target_data = process_frames(gesture_frames)

                        best_match_label = "Unknown"
                        min_distance = float('inf')

                        for template in templates:
                            distance = np.linalg.norm(template["data"] - target_data)

                            if distance < min_distance:
                                min_distance = distance
                                best_match_label = template["label"]

                        # 閾値チェック
                        if min_distance > RECOGNITION_THRESHOLD:
                            result_label = "登録されていません"
                        else:
                            result_label = best_match_label

                        result_distance = min_distance
                    else:
                        result_label = "Too short"
                        result_distance = 0

    cap.release()
    cv2.destroyAllWindows()
    callback((result_label, result_distance))

# --- 5. GUI クラス ---
class KoeLinkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("KoeLink - ジェスチャー認識&記録システム")
        self.root.geometry("500x400")
        self.root.configure(bg="#f0f0f0")

        self.show_main_menu()

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def show_main_menu(self):
        self.clear_window()

        frame = tk.Frame(self.root, bg="#f0f0f0")
        frame.pack(expand=True, fill="both", padx=20, pady=20)

        title = tk.Label(frame, text="🤝 KoeLink", font=("Arial", 28, "bold"), bg="#f0f0f0", fg="#333")
        title.pack(pady=20)

        subtitle = tk.Label(frame, text="ジェスチャー認識＆記録システム", font=("Arial", 14), bg="#f0f0f0", fg="#666")
        subtitle.pack(pady=10)

        btn_frame = tk.Frame(frame, bg="#f0f0f0")
        btn_frame.pack(pady=40)

        btn_record = tk.Button(
            btn_frame,
            text="📝 新しく登録する",
            font=("Arial", 14, "bold"),
            bg="#4CAF50",
            fg="white",
            padx=30,
            pady=15,
            command=self.show_record_menu,
            cursor="hand2"
        )
        btn_record.pack(pady=10, fill="x")

        btn_recognize = tk.Button(
            btn_frame,
            text="🔍 ジェスチャーを認識する",
            font=("Arial", 14, "bold"),
            bg="#2196F3",
            fg="white",
            padx=30,
            pady=15,
            command=self.start_recognize,
            cursor="hand2"
        )
        btn_recognize.pack(pady=10, fill="x")

        btn_exit = tk.Button(
            btn_frame,
            text="❌ 終了",
            font=("Arial", 12, "bold"),
            bg="#f44336",
            fg="white",
            padx=30,
            pady=12,
            command=self.root.quit,
            cursor="hand2"
        )
        btn_exit.pack(pady=10, fill="x")

    def show_record_menu(self):
        label = simpledialog.askstring(
            "ラベル入力",
            "記録するジェスチャーの意味を入力してください\n(例: hello, thanks, こんにちは)",
            parent=self.root
        )

        if label:
            self.root.withdraw()
            thread = threading.Thread(target=lambda: record_gesture(label, self.on_record_complete))
            thread.start()
        else:
            self.show_main_menu()

    def on_record_complete(self, saved_filenames):
        self.root.deiconify()

        if isinstance(saved_filenames, list) and len(saved_filenames) > 0:
            messagebox.showinfo("完了", f"{len(saved_filenames)}件のジェスチャーを保存しました！")

        self.show_post_record_menu()

    def show_post_record_menu(self):
        self.clear_window()

        frame = tk.Frame(self.root, bg="#f0f0f0")
        frame.pack(expand=True, fill="both", padx=20, pady=20)

        title = tk.Label(frame, text="記録完了", font=("Arial", 24, "bold"), bg="#f0f0f0", fg="#4CAF50")
        title.pack(pady=20)

        msg = tk.Label(frame, text="次のアクションを選択してください", font=("Arial", 12), bg="#f0f0f0", fg="#666")
        msg.pack(pady=10)

        btn_frame = tk.Frame(frame, bg="#f0f0f0")
        btn_frame.pack(pady=40)

        btn_record_more = tk.Button(
            btn_frame,
            text="➕ さらに登録する",
            font=("Arial", 14, "bold"),
            bg="#4CAF50",
            fg="white",
            padx=25,
            pady=12,
            command=self.show_record_menu,
            cursor="hand2"
        )
        btn_record_more.pack(pady=10, fill="x")

        btn_recognize = tk.Button(
            btn_frame,
            text="🔍 認識する",
            font=("Arial", 14, "bold"),
            bg="#2196F3",
            fg="white",
            padx=25,
            pady=12,
            command=self.start_recognize,
            cursor="hand2"
        )
        btn_recognize.pack(pady=10, fill="x")

        btn_menu = tk.Button(
            btn_frame,
            text="🏠 メインメニュー",
            font=("Arial", 12, "bold"),
            bg="#FF9800",
            fg="white",
            padx=25,
            pady=12,
            command=self.show_main_menu,
            cursor="hand2"
        )
        btn_menu.pack(pady=10, fill="x")

        btn_exit = tk.Button(
            btn_frame,
            text="❌ 終了",
            font=("Arial", 12, "bold"),
            bg="#f44336",
            fg="white",
            padx=25,
            pady=12,
            command=self.root.quit,
            cursor="hand2"
        )
        btn_exit.pack(pady=10, fill="x")

    def start_recognize(self):
        self.root.withdraw()
        thread = threading.Thread(target=lambda: recognize_gesture(self.on_recognize_complete))
        thread.start()

    def on_recognize_complete(self, result):
        self.root.deiconify()

        if result != "error" and isinstance(result, tuple):
            label, distance = result
            msg = f"結果: {label}\n距離: {distance:.2f}\n(閾値: {RECOGNITION_THRESHOLD})"
            messagebox.showinfo("認識結果", msg)
        elif result != "error":
            messagebox.showinfo("認識結果", f"結果: {result}")

        self.show_post_recognize_menu()

    def show_post_recognize_menu(self):
        self.clear_window()

        frame = tk.Frame(self.root, bg="#f0f0f0")
        frame.pack(expand=True, fill="both", padx=20, pady=20)

        title = tk.Label(frame, text="認識完了", font=("Arial", 24, "bold"), bg="#f0f0f0", fg="#2196F3")
        title.pack(pady=20)

        msg = tk.Label(frame, text="次のアクションを選択してください", font=("Arial", 12), bg="#f0f0f0", fg="#666")
        msg.pack(pady=10)

        btn_frame = tk.Frame(frame, bg="#f0f0f0")
        btn_frame.pack(pady=40)

        btn_recognize_again = tk.Button(
            btn_frame,
            text="🔍 もう一度認識",
            font=("Arial", 14, "bold"),
            bg="#2196F3",
            fg="white",
            padx=25,
            pady=12,
            command=self.start_recognize,
            cursor="hand2"
        )
        btn_recognize_again.pack(pady=10, fill="x")

        btn_record = tk.Button(
            btn_frame,
            text="📝 新しく登録",
            font=("Arial", 14, "bold"),
            bg="#4CAF50",
            fg="white",
            padx=25,
            pady=12,
            command=self.show_record_menu,
            cursor="hand2"
        )
        btn_record.pack(pady=10, fill="x")

        btn_menu = tk.Button(
            btn_frame,
            text="🏠 メインメニュー",
            font=("Arial", 12, "bold"),
            bg="#FF9800",
            fg="white",
            padx=25,
            pady=12,
            command=self.show_main_menu,
            cursor="hand2"
        )
        btn_menu.pack(pady=10, fill="x")

        btn_exit = tk.Button(
            btn_frame,
            text="❌ 終了",
            font=("Arial", 12, "bold"),
            bg="#f44336",
            fg="white",
            padx=25,
            pady=12,
            command=self.root.quit,
            cursor="hand2"
        )
        btn_exit.pack(pady=10, fill="x")

if __name__ == "__main__":
    root = tk.Tk()
    gui = KoeLinkGUI(root)
    root.mainloop()
