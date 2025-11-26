import cv2
import mediapipe as mp
import numpy as np
import random
import time

# --- CẤU HÌNH ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Danh sách tư thế
poses_dict = {
    "Luc Si (2 Tay Vuong)": {"left": 90, "right": 90, "tolerance": 25},
    "Chao Co (Trai Vuong - Phai Thang)": {"left": 90, "right": 170, "tolerance": 25},
    "Chim Bay (2 Tay Thang)": {"left": 170, "right": 170, "tolerance": 25},
    "Cheo Canh (Trai Thang - Phai Vuong)": {"left": 170, "right": 90, "tolerance": 25},
    "Khep Nach (2 Tay Duoi)": {"left": 25, "right": 25, "tolerance": 25}
}

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# --- KHỞI TẠO BIẾN TOÀN CỤC ---
score = 0
lives = 3
game_active = True # Trạng thái: Đang chơi hay đã thua
base_duration = 5.0 # Thời gian gốc
current_duration = base_duration 
start_time = time.time()
current_pose_name = random.choice(list(poses_dict.keys()))

# Hàm reset game khi chơi lại
def reset_game():
    global score, lives, game_active, current_duration, start_time, current_pose_name
    score = 0
    lives = 3
    game_active = True
    current_duration = 5.0
    start_time = time.time()
    current_pose_name = random.choice(list(poses_dict.keys()))

cap = cv2.VideoCapture(0)
cv2.namedWindow('Game Tetris - Survival Mode', cv2.WINDOW_NORMAL)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Lật ảnh gương + Xử lý màu
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        h, w, _ = image.shape

        # --- TRƯỜNG HỢP 1: ĐANG CHƠI (GAME ACTIVE) ---
        if game_active:
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Tính góc 2 tay
                l_sh = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                l_el = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                l_wr = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                angle_left = calculate_angle(l_sh, l_el, l_wr)
                
                r_sh = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                r_el = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                r_wr = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                angle_right = calculate_angle(r_sh, r_el, r_wr)
                
                # Logic kiểm tra góc
                target_l = poses_dict[current_pose_name]["left"]
                target_r = poses_dict[current_pose_name]["right"]
                tol = poses_dict[current_pose_name]["tolerance"]
                
                left_ok = (target_l - tol) < angle_left < (target_l + tol)
                right_ok = (target_r - tol) < angle_right < (target_r + tol)
                
                # Feedback màu sắc
                c_left = (0, 255, 0) if left_ok else (0, 0, 255)
                c_right = (0, 255, 0) if right_ok else (0, 0, 255)
                
                # Hiển thị số đo góc lên tay
                cv2.putText(image, str(int(angle_left)), tuple(np.multiply(l_el, [w, h]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, c_left, 2)
                cv2.putText(image, str(int(angle_right)), tuple(np.multiply(r_el, [w, h]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, c_right, 2)
                
                # LOGIC THỜI GIAN & TÍNH ĐIỂM
                elapsed = time.time() - start_time
                time_left = current_duration - elapsed
                
                if time_left <= 0:
                    # Hết giờ: Kiểm tra kết quả
                    if left_ok and right_ok:
                        score += 1
                        # TĂNG ĐỘ KHÓ: Cứ 3 điểm giảm 0.5 giây (Tối thiểu 2 giây)
                        if score % 3 == 0 and current_duration > 2.0:
                            current_duration -= 0.5
                            print(f"Tang toc! Thoi gian con: {current_duration}s")
                    else:
                        lives -= 1 # Mất mạng
                        print(f"Hut roi! Con {lives} mang.")
                        if lives == 0:
                            game_active = False # GAME OVER
                    
                    # Reset vòng chơi mới
                    start_time = time.time()
                    current_pose_name = random.choice(list(poses_dict.keys()))

                # --- VẼ UI KHI CHƠI ---
                # Thanh thời gian
                bar_width = int((time_left / current_duration) * w)
                bar_color = (0, 255, 0) if time_left > 2 else (0, 0, 255) # Xanh -> Đỏ khi sắp hết giờ
                cv2.rectangle(image, (0, h-20), (bar_width, h), bar_color, -1)
                
                # Đồng hồ đếm ngược to
                cv2.putText(image, f"{time_left:.1f}", (w//2 - 50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 3, bar_color, 4)

            except:
                pass

        # --- TRƯỜNG HỢP 2: GAME OVER (MÀN HÌNH TỐI) ---
        else:
            # Phủ một lớp màu đen mờ lên ảnh
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            image = cv2.addWeighted(overlay, 0.8, image, 0.2, 0)
            
            # Hiện thông báo thua
            cv2.putText(image, "GAME OVER", (w//2 - 180, h//2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            cv2.putText(image, f"Tong Diem: {score}", (w//2 - 120, h//2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            cv2.putText(image, "Nhan 'R' de choi lai", (w//2 - 160, h//2 + 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

            # Vẽ xương
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
        # --- VẼ UI CỐ ĐỊNH (LUÔN HIỆN) ---
        # Hộp thông tin góc trái
        cv2.rectangle(image, (0,0), (450, 100), (50, 50, 50), -1)
        cv2.putText(image, f"NV: {current_pose_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        # Vẽ trái tim (Mạng sống)
        hearts = "HEART: " + " <3"*lives
        cv2.putText(image, hearts, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2) # Màu đỏ
        
        # Vẽ điểm số góc phải
        cv2.putText(image, f"Diem: {score}", (w - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 215, 0), 2) # Màu vàng

        cv2.imshow('Game Tetris - Survival Mode', image)

        # Xử lý phím bấm
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'): # Thoát
            break
        if key == ord('r') and not game_active: # Chỉ cho reset khi đã Game Over
            reset_game()

cap.release()
cv2.destroyAllWindows()