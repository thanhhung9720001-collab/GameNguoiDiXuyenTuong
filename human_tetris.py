import cv2
import mediapipe as mp
import numpy as np
import random
import time

# --- KHỞI TẠO ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Hàm tính góc (Dùng chung cho cả 2 tay)
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# --- DỮ LIỆU CÁC TƯ THẾ (LEVEL 2 - 2 TAY) ---
# Quy ước: 170-180 là tay thẳng, 90 là vuông góc, 20-30 là khép nách
poses_dict = {
    "Luc Si (2 Tay Vuong)": {
        "left": 90, 
        "right": 90,
        "tolerance": 20
    },
    "Chao Co (Trai Vuong - Phai Thang)": {
        "left": 90,
        "right": 170,
        "tolerance": 20
    },
    "Chim Bay (2 Tay Thang)": {
        "left": 170,
        "right": 170,
        "tolerance": 20
    },
    "Cheo Canh (Trai Thang - Phai Vuong)": {
        "left": 170,
        "right": 90,
        "tolerance": 20
    }
}

# --- BIẾN GAME ---
score = 0
# Chọn ngẫu nhiên tư thế đầu tiên
current_pose_name = random.choice(list(poses_dict.keys()))
start_time = time.time()
game_duration = 5 # Thời gian mỗi lượt

cap = cv2.VideoCapture(0)
# Đặt tên cửa sổ mới cho ngầu
cv2.namedWindow('Game Tetris - Double Trouble', cv2.WINDOW_NORMAL)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Lật ngược ảnh lại cho giống gương (Mirror) để dễ chơi hơn
        frame = cv2.flip(frame, 1)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        h, w, _ = image.shape
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            # --- LẤY TOẠ ĐỘ TAY TRÁI (LEFT) ---
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            angle_left = calculate_angle(l_shoulder, l_elbow, l_wrist)
            
            # --- LẤY TOẠ ĐỘ TAY PHẢI (RIGHT) ---
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            angle_right = calculate_angle(r_shoulder, r_elbow, r_wrist)
            
            # --- LOGIC GAME ---
            elapsed_time = time.time() - start_time
            time_left = game_duration - elapsed_time
            
            # Lấy mục tiêu (Target)
            target_l = poses_dict[current_pose_name]["left"]
            target_r = poses_dict[current_pose_name]["right"]
            tol = poses_dict[current_pose_name]["tolerance"]
            
            # Kiểm tra từng tay
            left_ok = (target_l - tol) < angle_left < (target_l + tol)
            right_ok = (target_r - tol) < angle_right < (target_r + tol)
            
            # Màu sắc phản hồi (Feedback)
            color_left = (0, 255, 0) if left_ok else (0, 0, 255) # Xanh nếu đúng, Đỏ nếu sai
            color_right = (0, 255, 0) if right_ok else (0, 0, 255)
            
            status_text = "Co len!"
            main_color = (0, 0, 255) # Đỏ
            
            # CẢ HAI TAY ĐỀU PHẢI ĐÚNG
            if left_ok and right_ok:
                status_text = "Giu nguyen!"
                main_color = (0, 255, 0) # Xanh lá
            
            # Xử lý hết giờ
            if time_left <= 0:
                if left_ok and right_ok:
                    score += 1
                    print(f"GHI DIEM! Tong: {score}")
                else:
                    print("HUT ROI!")
                    
                start_time = time.time()
                current_pose_name = random.choice(list(poses_dict.keys()))

            # --- VẼ GIAO DIỆN ---
            # Thanh thời gian
            time_bar_width = int((time_left / game_duration) * w)
            cv2.rectangle(image, (0, h-20), (time_bar_width, h), main_color, -1)
            
            # Hộp thông tin
            cv2.rectangle(image, (0,0), (450, 140), (245,117,16), -1)
            cv2.putText(image, f'NV: {current_pose_name}', (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image, f'DIEM: {score}', (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)
            
            # Đồng hồ đếm ngược
            cv2.putText(image, str(int(time_left)+1), (w//2, h//2), cv2.FONT_HERSHEY_SIMPLEX, 4, main_color, 4, cv2.LINE_AA)

            # Vẽ số đo góc lên 2 khuỷu tay
            # Tay Trái
            cv2.putText(image, str(int(angle_left)), 
                        tuple(np.multiply(l_elbow, [w, h]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color_left, 2, cv2.LINE_AA)
            # Tay Phải
            cv2.putText(image, str(int(angle_right)), 
                        tuple(np.multiply(r_elbow, [w, h]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color_right, 2, cv2.LINE_AA)

        except Exception as e:
            pass

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        cv2.imshow('Game Tetris - Double Trouble', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()