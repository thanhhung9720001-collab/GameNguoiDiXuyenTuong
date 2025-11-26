import cv2
import mediapipe as mp
import numpy as np

# Khởi tạo
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Hàm tính góc (Giữ nguyên)
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

cap = cv2.VideoCapture(0)

# MỞ RỘNG CỬA SỔ CHO DỄ NHÌN
cv2.namedWindow('Game Tetris - Sprint 4', cv2.WINDOW_NORMAL)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Lấy toạ độ tay TRÁI
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # --- PHẦN MỚI: LOGIC TRỌNG TÀI ---
            # Mục tiêu: Giữ tay vuông góc (khoảng 90 độ)
            target_angle = 90
            tolerance = 15 # Cho phép sai số +- 15 độ (tức là từ 75 đến 105 là OK)

            # Mặc định là màu đỏ (Sai)
            status_text = "Sai Roi!"
            color = (0, 0, 255) # Red (Lưu ý: OpenCV dùng thứ tự Blue-Green-Red)

            # Kiểm tra xem có đạt yêu cầu không?
            if (target_angle - tolerance) < angle < (target_angle + tolerance):
                status_text = "TUYET VOI!"
                color = (0, 255, 0) # Green

            # Hiển thị lên màn hình
            h, w, _ = image.shape
            # Vị trí in chữ
            text_pos = tuple(np.multiply(elbow, [w, h]).astype(int))
            
            # In số đo góc
            cv2.putText(image, str(int(angle)), text_pos, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
            # In thông báo Đạt/Không Đạt
            cv2.putText(image, status_text, (text_pos[0], text_pos[1] + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
            # Vẽ một cái hộp chỉ dẫn ở góc trái màn hình
            cv2.rectangle(image, (0,0), (250, 73), (245,117,16), -1)
            cv2.putText(image, 'NHIEM VU:', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, 'Vuong Goc Tay Trai', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        except:
            pass

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        cv2.imshow('Game Tetris - Sprint 4', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()