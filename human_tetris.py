import cv2
import mediapipe as mp
import numpy as np
import random
import time

# --- KHỞI TẠO ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Hàm tính góc (Không đổi)
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# --- DỮ LIỆU CÁC TƯ THẾ (LEVELS) ---
# Bạn có thể tự nghĩ thêm các tư thế khác vào đây
poses_dict = {
    "Chao Co (Tay Thang)": {
        "target": 170,  # Góc mong muốn
        "tolerance": 20 # Dung sai
    },
    "Luc Si (Vuong Goc)": {
        "target": 90,
        "tolerance": 15
    },
    "Khep Nach (Tay Duoi)": {
        "target": 30,
        "tolerance": 20
    }
}

# --- BIẾN GAME ---
score = 0
current_pose_name = random.choice(list(poses_dict.keys())) # Chọn bừa 1 tư thế đầu tiên
start_time = time.time()
game_duration = 5 # Mỗi lượt chơi có 5 giây để chuẩn bị

cap = cv2.VideoCapture(0)
cv2.namedWindow('Game Tetris - Final', cv2.WINDOW_NORMAL)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Xử lý ảnh
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Lấy kích thước màn hình để vẽ chữ cho đẹp
        h, w, _ = image.shape
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Lấy toạ độ tay TRÁI (Làm mẫu tay trái thôi nhé)
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Tính góc hiện tại của người chơi
            current_angle = calculate_angle(shoulder, elbow, wrist)
            
            # --- LOGIC GAME LOOP ---
            # 1. Tính thời gian còn lại
            elapsed_time = time.time() - start_time
            time_left = game_duration - elapsed_time
            
            # Lấy thông tin tư thế hiện tại
            target = poses_dict[current_pose_name]["target"]
            tol = poses_dict[current_pose_name]["tolerance"]
            
            # Màu sắc mặc định
            color_status = (0, 0, 255) # Đỏ
            status_text = "Chuan bi..."
            
            # Kiểm tra xem người chơi có làm đúng không (Real-time feedback)
            if (target - tol) < current_angle < (target + tol):
                color_status = (0, 255, 0) # Xanh lá
                status_text = "Giu nguyen!"
            
            # 2. Xử lý khi hết giờ (Hết 5 giây)
            if time_left <= 0:
                # Kiểm tra kết quả lần cuối
                if (target - tol) < current_angle < (target + tol):
                    score += 1 # Cộng điểm
                    feedback = "GHI DIEM!"
                    box_color = (0, 255, 0)
                else:
                    feedback = "HUT ROI!"
                    box_color = (0, 0, 255)
                    
                # Reset lại thời gian và chọn tư thế mới
                start_time = time.time()
                current_pose_name = random.choice(list(poses_dict.keys()))
                
                # (Mẹo nhỏ) Hiện kết quả trong 1 giây đệm (bạn có thể cải tiến sau)
                print(f"Ket qua: {feedback}. Diem hien tai: {score}")

            # --- VẼ GIAO DIỆN (UI) ---
            
            # Vẽ thanh thời gian chạy ngang màn hình
            time_bar_width = int((time_left / game_duration) * w)
            cv2.rectangle(image, (0, h-20), (time_bar_width, h), color_status, -1)
            
            # Vẽ hộp thông tin góc trái
            cv2.rectangle(image, (0,0), (350, 120), (245,117,16), -1)
            
            # Tên tư thế cần làm
            cv2.putText(image, 'NHIEM VU:', (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, current_pose_name, (15,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
            
            # Điểm số
            cv2.putText(image, f'DIEM: {score}', (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Đồng hồ đếm ngược to đùng ở giữa màn hình
            cv2.putText(image, str(int(time_left)+1), (w//2, h//2), cv2.FONT_HERSHEY_SIMPLEX, 4, color_status, 4, cv2.LINE_AA)

            # Vẽ góc lên khuỷu tay
            cv2.putText(image, str(int(current_angle)), 
                        tuple(np.multiply(elbow, [w, h]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color_status, 2, cv2.LINE_AA)

        except Exception as e:
            pass

        # Vẽ xương
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        cv2.imshow('Game Tetris - Final', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()