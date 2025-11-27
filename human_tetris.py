"""
ASSIGNMENT 2: GAME "EXER-GAMING" - FULL BODY WORKOUT
-------------------------------------------------------------------
Đã sửa lỗi:
1. Tư thế CSGT/Chéo Cánh: Cho phép đổi tay trái/phải thoải mái.
2. Dọn dẹp code rác bị lặp.
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import time
import os

# --- 1. HÀM QUẢN LÝ ĐIỂM SỐ (HIGH SCORE) ---
def get_high_score():
    if not os.path.exists("highscore.txt"):
        return 0
    try:
        with open("highscore.txt", "r") as f:
            return int(f.read())
    except:
        return 0

def save_high_score(new_score):
    try:
        with open("highscore.txt", "w") as f:
            f.write(str(new_score))
    except:
        pass

# --- 2. CẤU HÌNH & DỮ LIỆU ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Danh sách tư thế TAY
arm_poses = {
    "Luc Si (2 Tay Vuong)":   {"left": 90,  "right": 90,  "tolerance": 25},
    "Chim Bay (2 Tay Thang)": {"left": 170, "right": 170, "tolerance": 25},
    "CSGT (1 Thang 1 Vuong)": {"left": 170, "right": 90,  "tolerance": 25}, # Đã sửa logic check ở dưới để chấp nhận ngược tay
}

# --- 3. HÀM VẼ NGƯỜI QUE (STICKMAN) ---
def draw_stickman(img, pose_name, x, y, size=80):
    thickness = 3
    color = (255, 255, 255)
    
    # Đầu & Thân
    cv2.circle(img, (x, y - size//2), size//4, color, -1) 
    body_bottom = y + size//2
    cv2.line(img, (x, y), (x, body_bottom), color, thickness)
    
    # Chân
    if "SQUAT" in str(pose_name):
        # Chân chữ M (Squat)
        cv2.line(img, (x, body_bottom), (x - size//3, body_bottom + size//3), color, thickness)
        cv2.line(img, (x - size//3, body_bottom + size//3), (x - size//4, body_bottom + size//2 + 10), color, thickness)
        cv2.line(img, (x, body_bottom), (x + size//3, body_bottom + size//3), color, thickness)
        cv2.line(img, (x + size//3, body_bottom + size//3), (x + size//4, body_bottom + size//2 + 10), color, thickness)
    else:
        # Chân chữ V (Đứng)
        cv2.line(img, (x, body_bottom), (x - size//3, body_bottom + size), color, thickness)
        cv2.line(img, (x, body_bottom), (x + size//3, body_bottom + size), color, thickness)

    # Tay
    l_shoulder = (x - size//4, y)
    r_shoulder = (x + size//4, y)
    cv2.line(img, (x, y), l_shoulder, color, thickness)
    cv2.line(img, (x, y), r_shoulder, color, thickness)

    # Logic tay theo tên
    # Mặc định buông thõng
    l_elbow = (l_shoulder[0] - 10, l_shoulder[1] + 30)
    l_wrist = (l_elbow[0], l_elbow[1] + 20)
    r_elbow = (r_shoulder[0] + 10, r_shoulder[1] + 30)
    r_wrist = (r_elbow[0], r_elbow[1] + 20)

    if pose_name and "Luc Si" in pose_name:
        l_elbow = (l_shoulder[0] - 20, l_shoulder[1])
        l_wrist = (l_elbow[0], l_elbow[1] - 30)
        r_elbow = (r_shoulder[0] + 20, r_shoulder[1])
        r_wrist = (r_elbow[0], r_elbow[1] - 30)

    elif pose_name and "Chim Bay" in pose_name:
        l_elbow = (l_shoulder[0] - 20, l_shoulder[1])
        l_wrist = (l_elbow[0] - 25, l_elbow[1])
        r_elbow = (r_shoulder[0] + 20, r_shoulder[1])
        r_wrist = (r_elbow[0] + 25, r_elbow[1])
        
    elif pose_name and ("Cheo Canh" in pose_name or "CSGT" in pose_name):
        l_elbow = (l_shoulder[0] - 20, l_shoulder[1])
        l_wrist = (l_elbow[0] - 25, l_elbow[1])
        r_elbow = (r_shoulder[0] + 20, r_shoulder[1])
        r_wrist = (r_elbow[0], r_elbow[1] - 30)
        
    elif pose_name and "SQUAT" in pose_name:
        l_elbow = (l_shoulder[0], l_shoulder[1] + 20)
        l_wrist = (l_elbow[0] + 10, l_elbow[1] + 10)
        r_elbow = (r_shoulder[0], r_shoulder[1] + 20)
        r_wrist = (r_elbow[0] - 10, r_elbow[1] + 10)

    cv2.line(img, l_shoulder, l_elbow, color, thickness)
    cv2.line(img, l_elbow, l_wrist, color, thickness)
    cv2.line(img, r_shoulder, r_elbow, color, thickness)
    cv2.line(img, r_elbow, r_wrist, color, thickness)

# --- 4. HÀM TÍNH GÓC ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# --- 5. KHỞI TẠO GAME ---
score = 0
lives = 3
game_active = False 
calibration_frames = 60 
base_y = 0 

base_duration = 5.0
current_duration = base_duration 
start_time = time.time()
high_score = get_high_score()
print(f"Ky luc hien tai: {high_score}")

current_task = None 
task_type = None 

def new_round():
    global current_task, task_type, start_time
    start_time = time.time()
    if random.random() < 0.7:
        task_type = "ARM"
        current_task = random.choice(list(arm_poses.keys()))
    else:
        task_type = "LEG"
        current_task = "SQUAT XUONG (Ne Dan)"

cap = cv2.VideoCapture(0)
cv2.namedWindow('Full Body ExerGame', cv2.WINDOW_NORMAL)

# --- 6. GAME LOOP ---
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h, w, _ = image.shape

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
            right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
            current_hip_y = (left_hip_y + right_hip_y) / 2

            # --- A. LẤY MỐC (CALIBRATION) ---
            if not game_active and lives > 0:
                cv2.rectangle(image, (0,0), (w, h), (50, 50, 50), -1)
                cv2.putText(image, "DUNG THANG DE LAY MOC", (w//2 - 250, h//2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(image, f"Chuan bi: {calibration_frames}", (w//2 - 100, h//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 4)
                
                calibration_frames -= 1
                if calibration_frames <= 0:
                    base_y = current_hip_y
                    game_active = True
                    new_round()

            # --- B. CHƠI GAME ---
            elif game_active:
                success = False
                
                # B.1: KIỂM TRA TAY
                if task_type == "ARM":
                    # Lấy toạ độ
                    l_sh = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    l_el = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    l_wr = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    angle_left = calculate_angle(l_sh, l_el, l_wr)
                    
                    r_sh = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    r_el = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    r_wr = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    angle_right = calculate_angle(r_sh, r_el, r_wr)

                    # Lấy mục tiêu
                    target_l = arm_poses[current_task]["left"]
                    target_r = arm_poses[current_task]["right"]
                    tol = arm_poses[current_task]["tolerance"]
                    
                    # --- SỬA LỖI 1: LOGIC LINH HOẠT CHO PHÉP ĐỔI TAY ---
                    # Trường hợp 1: Tay Trái = Target Trái VÀ Tay Phải = Target Phải (Làm đúng chiều)
                    case_normal = ((target_l - tol) < angle_left < (target_l + tol)) and \
                                  ((target_r - tol) < angle_right < (target_r + tol))
                    
                    # Trường hợp 2: Tay Trái = Target Phải VÀ Tay Phải = Target Trái (Làm ngược chiều vẫn OK)
                    case_swapped = ((target_r - tol) < angle_left < (target_r + tol)) and \
                                   ((target_l - tol) < angle_right < (target_l + tol))
                                   
                    if case_normal or case_swapped:
                        success = True

                    cv2.putText(image, str(int(angle_left)), tuple(np.multiply(l_el, [w, h]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    cv2.putText(image, str(int(angle_right)), tuple(np.multiply(r_el, [w, h]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                # B.2: KIỂM TRA CHÂN
                elif task_type == "LEG":
                    squat_threshold = base_y + 0.15
                    if current_hip_y > squat_threshold:
                        success = True
                        cv2.putText(image, "DA SQUAT!", (w//2 - 100, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    else:
                        cv2.putText(image, "NGOI THAP XUONG!", (w//2 - 150, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    
                    line_y = int(squat_threshold * h)
                    cv2.line(image, (0, line_y), (w, line_y), (0, 255, 255), 2)

                # B.3: XỬ LÝ THỜI GIAN
                elapsed = time.time() - start_time
                time_left = current_duration - elapsed
                status_color = (0, 255, 0) if success else (0, 0, 255)
                
                if time_left <= 0:
                    if success:
                        score += 1
                        if score % 3 == 0 and current_duration > 2.5:
                            current_duration -= 0.5
                    else:
                        lives -= 1
                        if lives == 0:
                            game_active = False 
                    new_round()

                # B.4: VẼ UI (GIAO DIỆN)
                bar_width = int((time_left / current_duration) * w)
                cv2.rectangle(image, (0, h-20), (bar_width, h), status_color, -1)
                
                cv2.rectangle(image, (0,0), (500, 100), (50, 50, 50), -1)
                task_display = current_task if task_type == "ARM" else "!!! SQUAT XUONG !!!"
                cv2.putText(image, f"NV: {task_display}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(image, f"DIEM: {score} | MANG: {lives}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,215,0), 2)
                cv2.putText(image, f"{time_left:.1f}", (w//2, h//2), cv2.FONT_HERSHEY_SIMPLEX, 3, status_color, 4)
                
                # Vẽ Kỷ lục
                cv2.putText(image, f"TOP: {high_score}", (w - 180, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # Vẽ Stickman
                draw_stickman(image, current_task, 80, 180, size=60)
                cv2.rectangle(image, (20, 120), (140, 260), (255, 255, 255), 2)

            # --- C. GAME OVER ---
            elif lives == 0:
                if score > high_score:
                    high_score = score
                    save_high_score(high_score)
                    msg_hscore = "PHA KY LUC MOI!"
                    color_hscore = (0, 255, 0)
                else:
                    msg_hscore = f"Ky luc van la: {high_score}"
                    color_hscore = (200, 200, 200)

                overlay = image.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                image = cv2.addWeighted(overlay, 0.8, image, 0.2, 0)
                
                cv2.putText(image, "GAME OVER", (w//2 - 180, h//2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                cv2.putText(image, f"Tong Diem: {score}", (w//2 - 140, h//2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                cv2.putText(image, msg_hscore, (w//2 - 180, h//2 + 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color_hscore, 2)
                cv2.putText(image, "Nhan 'R' de Reset", (w//2 - 150, h//2 + 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Full Body ExerGame', image)
        
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'): break
        if key == ord('r') and lives == 0:
            lives = 3
            score = 0
            current_duration = 5.0
            calibration_frames = 60
            game_active = False

cap.release()
cv2.destroyAllWindows()