import cv2  # Nhập thư viện OpenCV (Đôi mắt)
import mediapipe as mp  # Nhập thư viện MediaPipe (Trí tuệ nhân tạo)
import numpy as np  # Nhập thư viện toán học

# 1. Khởi tạo các công cụ vẽ và nhận diện
mp_drawing = mp.solutions.drawing_utils # Công cụ để vẽ các đường nối khớp xương
mp_pose = mp.solutions.pose # Công cụ để phát hiện tư thế người

# 2. Bắt đầu đọc Webcam (số 0 thường là webcam mặc định của laptop)
cap = cv2.VideoCapture(0)

# 3. Thiết lập mô hình AI
# min_detection_confidence=0.5 nghĩa là: Nếu AI chắc chắn trên 50% đó là người thì mới nhận.
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    # Bắt đầu vòng lặp game (Chạy liên tục cho đến khi tắt)
    while cap.isOpened():
        ret, frame = cap.read() # Đọc từng khung hình từ camera
        if not ret:
            break
        
        # --- Xử lý hình ảnh ---
        # MediaPipe thích màu RGB, còn OpenCV dùng BGR. Phải chuyển đổi màu.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False # Khóa ảnh lại để xử lý nhanh hơn
        
        # Gửi ảnh cho AI phân tích để tìm xương
        results = pose.process(image)
        
        # Chuyển lại màu về BGR để hiển thị lên màn hình cho mắt người xem
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # --- Vẽ xương lên người ---
        # Nếu tìm thấy người (results.pose_landmarks không rỗng)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS, # Nối các điểm lại thành bộ xương
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), # Màu khớp
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)  # Màu đường nối
            )
            
        # 4. Hiển thị hình ảnh lên cửa sổ
        cv2.imshow('Game Tetris Nguoi - Demo', image)

        # 5. Nhấn phím 'q' để thoát chương trình
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Giải phóng camera và đóng cửa sổ khi xong
cap.release()
cv2.destroyAllWindows()