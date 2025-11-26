import cv2
import mediapipe as mp
import numpy as np

# Khởi tạo MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- PHẦN MỚI 1: HÀM TÍNH GÓC ---
def calculate_angle(a, b, c):
    """
    Hàm này nhận vào 3 điểm toạ độ: a (đầu), b (giữa), c (cuối).
    Ví dụ: a=Vai, b=Khuỷu tay, c=Cổ tay.
    Nó sẽ trả về góc tạo bởi 3 điểm này (tính bằng độ).
    """
    a = np.array(a) # Điểm đầu
    b = np.array(b) # Điểm giữa (đỉnh góc)
    c = np.array(c) # Điểm cuối
    
    # Công thức toán học (ArcTangent) để tính góc trong không gian 2D
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi) # Đổi từ radian sang độ (degree)
    
    # Vì cánh tay con người không bẻ ngược được quá 360 độ, 
    # nên ta đưa về khoảng 0-180 độ cho dễ hiểu.
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Bắt đầu Webcam
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Chuyển màu và xử lý
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # --- PHẦN MỚI 2: LẤY TOẠ ĐỘ VÀ TÍNH GÓC ---
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Lấy toạ độ 3 điểm của CÁNH TAY TRÁI
            # MediaPipe đánh số: 11=Vai trái, 13=Khuỷu trái, 15=Cổ tay trái
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Gọi hàm tính góc ở trên
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # Hiện con số góc lên màn hình (ngay chỗ khuỷu tay)
            # Chuyển toạ độ từ tỉ lệ (0-1) sang pixel màn hình để in chữ
            h, w, _ = image.shape
            cv2.putText(image, str(int(angle)), 
                           tuple(np.multiply(elbow, [w, h]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                        )
            
            # In ra Terminal để bạn dễ theo dõi
            # print(f"Goc tay trai: {angle}")
            
        except:
            pass # Nếu không thấy người thì bỏ qua, không báo lỗi

        # Vẽ xương
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        cv2.imshow('Game Tetris - Do Goc', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()