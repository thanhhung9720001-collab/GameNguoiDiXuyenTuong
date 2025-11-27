"""
ASSIGNMENT 2 FINAL: GAME "AI EXER-GAMING"
Phiên bản: Pro Edition (Menu + Sound + High Score)
"""
import cv2
import mediapipe as mp
import numpy as np
import random
import time
import os
import winsound # Thư viện âm thanh có sẵn trên Windows
import threading # Để chạy âm thanh không làm đơ game

# --- 1. HỆ THỐNG ÂM THANH (SOUND FX) ---
def play_sound(type):
    """
    Hàm phát âm thanh chạy luồng riêng để không làm lag game
    """
    def run():
        if type == "score":
            # Tiếng Ting (Tần số 1000Hz, dài 100ms)
            winsound.Beep(1000, 100)
        elif type == "fail":
            # Tiếng Bèooo (Tần số thấp 400Hz, dài 300ms)
            winsound.Beep(400, 300)
        elif type == "gameover":
            # Chuỗi âm thanh Game Over
            winsound.Beep(500, 200)
            winsound.Beep(400, 200)
            winsound.Beep(300, 400)
    
    # Chạy âm thanh trong luồng phụ (Thread)
    threading.Thread(target=run, daemon=True).start()

# --- 2. QUẢN LÝ FILE (HIGH SCORE) ---
def get_high_score():
    if not os.path.exists("highscore.txt"): return 0
    try:
        with open("highscore.txt", "r") as f: return int(f.read())
    except: return 0

def save_high_score(new_score):
    try:
        with open("highscore.txt", "w") as f: f.write(str(new_score))
    except: pass

# --- 3. CẤU HÌNH & DỮ LIỆU ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

arm_poses = {
    "Luc Si (2 Tay Vuong)":   {"left": 90,  "right": 90,  "tolerance": 25},
    "Chim Bay (2 Tay Thang)": {"left": 170, "right": 170, "tolerance": 25},
    "CSGT (1 Thang 1 Vuong)": {"left": 170, "right": 90,  "tolerance": 25},
}

# --- 4. HÀM VẼ & TÍNH TOÁN ---
def draw_stickman(img, pose_name, x, y, size=80):
    thickness = 3
    color = (255, 255, 255)
    
    # Đầu & Thân
    cv2.circle(img, (x, y - size//2), size//4, color, -1) 
    body_bottom = y + size//2
    cv2.line(img, (x, y), (x, body_bottom), color, thickness)
    
    # Chân
    if pose_name and "SQUAT" in str(pose_name):
        cv2.line(img, (x, body_bottom), (x - size//3, body_bottom + size//3), color, thickness)
        cv2.line(img, (x - size//3, body_bottom + size//3), (x - size//4, body_bottom + size//2 + 10), color, thickness)
        cv2.line(img, (x, body_bottom), (x + size//3, body_bottom + size//3), color, thickness)
        cv2.line(img, (x + size//3, body_bottom + size//3), (x + size//4, body_bottom + size//2 + 10), color, thickness)
    else:
        cv2.line(img, (x, body_bottom), (x - size//3, body_bottom + size), color, thickness)
        cv2.line(img, (x, body_bottom), (x + size//3, body_bottom + size), color, thickness)

    # Tay
    l_shoulder = (x - size//4, y)
    r_shoulder = (x + size//4, y)
    
    l_elbow, l_wrist = (l_shoulder[0]-10, l_shoulder[1]+30), (l_shoulder[0]-10, l_shoulder[1]+50)
    r_elbow, r_wrist = (r_shoulder[0]+10, r_shoulder[1]+30), (r_shoulder[0]+10, r_shoulder[1]+50)

    if pose_name and "Luc Si" in pose_name:
        l_elbow, l_wrist = (l_shoulder[0]-20, l_shoulder[1]), (l_shoulder[0]-20, l_shoulder[1]-30)
        r_elbow, r_wrist = (r_shoulder[0]+20, r_shoulder[1]), (r_shoulder[0]+20, r_shoulder[1]-30)
    elif pose_name and "Chim Bay" in pose_name:
        l_elbow, l_wrist = (l_shoulder[0]-20, l_shoulder[1]), (l_shoulder[0]-45, l_shoulder[1])
        r_elbow, r_wrist = (r_shoulder[0]+20, r_shoulder[1]), (r_shoulder[0]+45, r_shoulder[1])
    elif pose_name and ("Cheo Canh" in pose_name or "CSGT" in pose_name):
        l_elbow, l_wrist = (l_shoulder[0]-20, l_shoulder[1]), (l_shoulder[0]-45, l_shoulder[1])
        r_elbow, r_wrist = (r_shoulder[0]+20, r_shoulder[1]), (r_shoulder[0]+20, r_shoulder[1]-30)
    elif pose_name and "SQUAT" in pose_name:
        l_elbow, l_wrist = (l_shoulder[0], l_shoulder[1]+20), (l_shoulder[0]+10, l_shoulder[1]+10)
        r_elbow, r_wrist = (r_shoulder[0], r_shoulder[1]+20), (r_shoulder[0]-10, r_shoulder[1]+10)

    # Vẽ khung xương tay
    cv2.line(img, (x, y), l_shoulder, color, thickness)
    cv2.line(img, (x, y), r_shoulder, color, thickness)
    cv2.line(img, l_shoulder, l_elbow, color, thickness)
    cv2.line(img, l_elbow, l_wrist, color, thickness)
    cv2.line(img, r_shoulder, r_elbow, color, thickness)
    cv2.line(img, r_elbow, r_wrist, color, thickness)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# --- 5. BIẾN GAME TOÀN CỤC ---
score = 0
lives = 3
high_score = get_high_score()
calibration_frames = 60
base_y = 0 
base_duration = 5.0
current_duration = base_duration
start_time = time.time()
current_task = None 
task_type = None 

# TRẠNG THÁI GAME (STATE MACHINE)
# "MENU": Màn hình chờ
# "CALIBRATION": Đang lấy mốc
# "PLAYING": Đang chơi
# "GAMEOVER": Đã thua
game_state = "MENU" 

def new_round():
    global current_task, task_type, start_time
    start_time = time.time()
    if random.random() < 0.7:
        task_type = "ARM"
        current_task = random.choice(list(arm_poses.keys()))
    else:
        task_type = "LEG"
        current_task = "SQUAT XUONG (Ne Dan)"

def reset_game():
    global score, lives, current_duration, calibration_frames, game_state
    score = 0
    lives = 3
    current_duration = 5.0
    calibration_frames = 60
    game_state = "CALIBRATION" # Chuyển sang lấy mốc trước khi chơi

cap = cv2.VideoCapture(0)
cv2.namedWindow('AI ExerGame Pro', cv2.WINDOW_NORMAL)

# --- 6. VÒNG LẶP CHÍNH ---
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        # Tạo hiệu ứng nền tối cho Menu
        if game_state == "MENU":
            frame = cv2.convertScaleAbs(frame, alpha=0.6, beta=0)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h, w, _ = image.shape

        # --- XỬ LÝ THEO TỪNG TRẠNG THÁI (STATE) ---
        
        # 1. TRẠNG THÁI MENU
        if game_state == "MENU":
            cv2.rectangle(image, (0, h//2 - 100), (w, h//2 + 100), (0, 0, 0), -1)
            cv2.putText(image, "AI EXER-GAME", (w//2 - 200, h//2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
            cv2.putText(image, "Nhan 'SPACE' de Bat Dau", (w//2 - 220, h//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f"TOP SCORE: {high_score}", (w//2 - 120, h//2 + 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 215, 0), 2)

        # 2. TRẠNG THÁI LẤY MỐC (CALIBRATION)
        elif game_state == "CALIBRATION":
             if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Lấy trung bình hông
                current_hip_y = (landmarks[23].y + landmarks[24].y) / 2
                
                cv2.rectangle(image, (0,0), (w, h), (50, 50, 50), -1)
                cv2.putText(image, "DUNG THANG NGHIEM!", (w//2 - 250, h//2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
                cv2.putText(image, str(calibration_frames//10), (w//2 - 30, h//2 + 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,255), 5)
                
                calibration_frames -= 1
                if calibration_frames <= 0:
                    base_y = current_hip_y
                    game_state = "PLAYING"
                    new_round()
                    play_sound("score") # Ting một cái báo hiệu bắt đầu

        # 3. TRẠNG THÁI ĐANG CHƠI (PLAYING)
        elif game_state == "PLAYING":
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Lấy tọa độ cần thiết
                l_sh, l_el, l_wr = [landmarks[11].x, landmarks[11].y], [landmarks[13].x, landmarks[13].y], [landmarks[15].x, landmarks[15].y]
                r_sh, r_el, r_wr = [landmarks[12].x, landmarks[12].y], [landmarks[14].x, landmarks[14].y], [landmarks[16].x, landmarks[16].y]
                current_hip_y = (landmarks[23].y + landmarks[24].y) / 2
                
                # Logic Game
                success = False
                if task_type == "ARM":
                    angle_left = calculate_angle(l_sh, l_el, l_wr)
                    angle_right = calculate_angle(r_sh, r_el, r_wr)
                    
                    t_l, t_r, tol = arm_poses[current_task]["left"], arm_poses[current_task]["right"], arm_poses[current_task]["tolerance"]
                    # Kiểm tra xuôi & ngược
                    if (((t_l-tol) < angle_left < (t_l+tol)) and ((t_r-tol) < angle_right < (t_r+tol))) or \
                       (((t_r-tol) < angle_left < (t_r+tol)) and ((t_l-tol) < angle_right < (t_l+tol))):
                        success = True
                    
                    # Hiện số đo
                    cv2.putText(image, str(int(angle_left)), tuple(np.multiply(l_el, [w, h]).astype(int)), 1, 1, (255,255,255), 2)
                    cv2.putText(image, str(int(angle_right)), tuple(np.multiply(r_el, [w, h]).astype(int)), 1, 1, (255,255,255), 2)

                elif task_type == "LEG":
                    squat_thresh = base_y + 0.15
                    if current_hip_y > squat_thresh:
                        success = True
                        cv2.putText(image, "DA SQUAT!", (w//2-100, h-150), 1, 1, (0,255,0), 2)
                    else:
                        cv2.line(image, (0, int(squat_thresh*h)), (w, int(squat_thresh*h)), (0,255,255), 2)

                # Thời gian & Xử lý
                elapsed = time.time() - start_time
                time_left = current_duration - elapsed
                color_st = (0,255,0) if success else (0,0,255)
                
                if time_left <= 0:
                    if success:
                        score += 1
                        play_sound("score") # Âm thanh ghi điểm
                        if score % 3 == 0 and current_duration > 2.0: current_duration -= 0.5
                    else:
                        lives -= 1
                        play_sound("fail") # Âm thanh mất mạng
                        if lives == 0:
                            game_state = "GAMEOVER"
                            play_sound("gameover")
                            if score > high_score:
                                high_score = score
                                save_high_score(high_score)
                    new_round()

                # Vẽ UI
                cv2.rectangle(image, (0, h-20), (int(time_left/current_duration*w), h), color_st, -1)
                cv2.rectangle(image, (0,0), (500, 100), (50, 50, 50), -1)
                task_txt = current_task if task_type == "ARM" else "SQUAT XUONG!"
                cv2.putText(image, f"NV: {task_txt}", (10, 40), 1, 0.7, (255,255,255), 2)
                cv2.putText(image, f"DIEM: {score} | MANG: {lives}", (10, 80), 1, 0.8, (255,215,0), 2)
                cv2.putText(image, f"{time_left:.1f}", (w//2, h//2), 1, 3, color_st, 4)
                cv2.putText(image, f"TOP: {high_score}", (w-180, 40), 1, 1, (0,255,255), 2)
                
                draw_stickman(image, current_task, 80, 180, 60)
                cv2.rectangle(image, (20, 120), (140, 260), (255, 255, 255), 2)

        # 4. TRẠNG THÁI GAME OVER
        elif game_state == "GAMEOVER":
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            image = cv2.addWeighted(overlay, 0.8, image, 0.2, 0)
            
            cv2.putText(image, "GAME OVER", (w//2 - 180, h//2 - 50), 1, 2, (0, 0, 255), 4)
            cv2.putText(image, f"Diem cua ban: {score}", (w//2 - 140, h//2 + 20), 1, 1.5, (255, 255, 255), 2)
            if score == high_score and score > 0:
                cv2.putText(image, "PHA KY LUC MOI!", (w//2 - 150, h//2 + 70), 1, 1, (0, 255, 0), 2)
            cv2.putText(image, "Nhan 'R' de choi lai", (w//2 - 150, h//2 + 120), 1, 1, (200, 200, 200), 2)

        # Vẽ xương (luôn hiện cho ngầu)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('AI ExerGame Pro', image)
        
        # XỬ LÝ PHÍM BẤM
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'): break
        
        # Logic chuyển trạng thái bằng phím
        if game_state == "MENU":
            if key == ord(' '): # Nhấn Space để bắt đầu
                reset_game()
        elif game_state == "GAMEOVER":
            if key == ord('r'): # Nhấn R để chơi lại
                reset_game()

cap.release()
cv2.destroyAllWindows()