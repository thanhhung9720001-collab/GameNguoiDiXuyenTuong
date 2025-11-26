import cv2
import mediapipe as mp
import numpy as np
import random
import time

# --- CẤU HÌNH ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Danh sách tư thế TAY
arm_poses = {
    "Luc Si (2 Tay Vuong)": {"left": 90, "right": 90, "tolerance": 25},
    "Chim Bay (2 Tay Thang)": {"left": 170, "right": 170, "tolerance": 25},
    "Cheo Canh (Trai Thang - Phai Vuong)": {"left": 170, "right": 90, "tolerance": 25},
}

# Danh sách tư thế CHÂN (Dựa trên độ lệch Y của hông)
leg_poses = [
    "SQUAT XUONG (Ne Dan)",
    "DUNG THANG (Nghi ngoi)"
]

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# --- BIẾN GAME ---
score = 0
lives = 3
game_active = False # Chưa chơi ngay, cần Calibrate (lấy mốc) trước
calibration_frames = 60 # Cần đứng im 2 giây (60 frames) để lấy mốc
base_y = 0 # Vị trí hông chuẩn khi đứng

base_duration = 5.0
current_duration = base_duration 
start_time = time.time()

# Nhiệm vụ hiện tại: Có thể là Dict (Tay) hoặc String (Chân)
current_task = None 
task_type = None # "ARM" hoặc "LEG"

def new_round():
    global current_task, task_type, start_time
    start_time = time.time()
    
    # Random: 70% ra tay, 30% ra chân
    if random.random() < 0.7:
        task_type = "ARM"
        current_task = random.choice(list(arm_poses.keys()))
    else:
        task_type = "LEG"
        current_task = "SQUAT XUONG (Ne Dan)" # Hiện tại chỉ làm Squat cho đơn giản

cap = cv2.VideoCapture(0)
cv2.namedWindow('Full Body ExerGame', cv2.WINDOW_NORMAL)

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
            
            # Lấy toạ độ trung bình của 2 bên hông (Hip)
            left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
            right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
            current_hip_y = (left_hip_y + right_hip_y) / 2

            # --- GIAI ĐOẠN 1: CALIBRATION (LẤY MỐC ĐỨNG) ---
            if not game_active and lives > 0:
                cv2.rectangle(image, (0,0), (w, h), (50, 50, 50), -1)
                cv2.putText(image, "DUNG THANG DE LAY MOC", (w//2 - 250, h//2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(image, f"Chuan bi: {calibration_frames}", (w//2 - 100, h//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 4)
                
                calibration_frames -= 1
                if calibration_frames <= 0:
                    base_y = current_hip_y # Lưu vị trí hông chuẩn
                    game_active = True
                    new_round()
                    print(f"Da lay moc HONG: {base_y}")

            # --- GIAI ĐOẠN 2: GAMEPLAY ---
            elif game_active:
                # 1. XỬ LÝ NHIỆM VỤ
                success = False
                
                # Nếu là bài tập TAY
                if task_type == "ARM":
                    # Lấy toạ độ tay
                    l_sh = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    l_el = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    l_wr = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    angle_left = calculate_angle(l_sh, l_el, l_wr)
                    
                    r_sh = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    r_el = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    r_wr = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    angle_right = calculate_angle(r_sh, r_el, r_wr)

                    # Kiểm tra logic
                    target_l = arm_poses[current_task]["left"]
                    target_r = arm_poses[current_task]["right"]
                    tol = arm_poses[current_task]["tolerance"]
                    
                    if (target_l - tol) < angle_left < (target_l + tol) and \
                       (target_r - tol) < angle_right < (target_r + tol):
                        success = True

                    # Vẽ số đo tay
                    cv2.putText(image, str(int(angle_left)), tuple(np.multiply(l_el, [w, h]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    cv2.putText(image, str(int(angle_right)), tuple(np.multiply(r_el, [w, h]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                # Nếu là bài tập CHÂN (SQUAT)
                elif task_type == "LEG":
                    # Logic Squat: Hông hiện tại thấp hơn Hông chuẩn một khoảng (ví dụ 0.1 đơn vị ảnh)
                    # Lưu ý: Trục Y của ảnh đi từ trên xuống (0 ở trên, 1 ở dưới) -> Càng xuống thấp Y càng tăng
                    squat_threshold = base_y + 0.15 # Phải ngồi thấp xuống ít nhất 15% chiều cao ảnh
                    
                    if current_hip_y > squat_threshold: # Đang ngồi
                        success = True
                        cv2.putText(image, "DA SQUAT!", (w//2 - 100, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    else:
                        cv2.putText(image, "NGOI THAP XUONG!", (w//2 - 150, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    
                    # Vẽ vạch chuẩn để người chơi biết phải ngồi qua vạch đó
                    line_y = int(squat_threshold * h)
                    cv2.line(image, (0, line_y), (w, line_y), (0, 255, 255), 2)

                # 2. XỬ LÝ THỜI GIAN
                elapsed = time.time() - start_time
                time_left = current_duration - elapsed
                
                # Feedback màu sắc
                status_color = (0, 255, 0) if success else (0, 0, 255)
                
                if time_left <= 0:
                    if success:
                        score += 1
                        if score % 3 == 0 and current_duration > 2.5:
                            current_duration -= 0.5
                    else:
                        lives -= 1
                        if lives == 0:
                            game_active = False # Game Over -> Chuyển sang màn hình thua
                    
                    new_round()

                # 3. VẼ UI
                # Thanh thời gian
                bar_width = int((time_left / current_duration) * w)
                cv2.rectangle(image, (0, h-20), (bar_width, h), status_color, -1)
                
                # Thông tin
                cv2.rectangle(image, (0,0), (500, 100), (50, 50, 50), -1)
                task_display = current_task if task_type == "ARM" else "!!! SQUAT XUONG !!!"
                cv2.putText(image, f"NV: {task_display}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.putText(image, f"DIEM: {score} | MANG: {lives}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,215,0), 2)
                cv2.putText(image, f"{time_left:.1f}", (w//2, h//2), cv2.FONT_HERSHEY_SIMPLEX, 3, status_color, 4)

            # --- GIAI ĐOẠN 3: GAME OVER ---
            elif lives == 0:
                overlay = image.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                image = cv2.addWeighted(overlay, 0.8, image, 0.2, 0)
                cv2.putText(image, "GAME OVER", (w//2 - 180, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                cv2.putText(image, "Nhan 'R' de Reset", (w//2 - 150, h//2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Full Body ExerGame', image)
        
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'): break
        if key == ord('r') and lives == 0: # Reset
            lives = 3
            score = 0
            current_duration = 5.0
            calibration_frames = 60 # Cần lấy lại mốc vì có thể người chơi đã di chuyển
            game_active = False

cap.release()
cv2.destroyAllWindows()