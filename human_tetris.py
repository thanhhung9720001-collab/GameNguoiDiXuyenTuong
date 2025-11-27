"""
ASSIGNMENT 2 KIOSK EDITION: AI EXER-GAME (FINAL FIX)
Tính năng:
- Touchless UI: Dùng tay ảo để bấm nút Start/Restart.
- Item Manager: Quản lý Tiền/Bom.
- High Score & Sound FX.
"""
import cv2
import mediapipe as mp
import numpy as np
import random
import time
import os
import winsound
import threading
import math

# --- 1. HỆ THỐNG ÂM THANH ---
def play_sound(type):
    def run():
        if type == "score": winsound.Beep(1000, 50)
        elif type == "coin": winsound.Beep(2000, 50)
        elif type == "bomb": winsound.Beep(150, 400)
        elif type == "click": winsound.Beep(800, 100)
        elif type == "hover": winsound.Beep(600, 30)
        elif type == "gameover": 
            winsound.Beep(500, 150); winsound.Beep(400, 150); winsound.Beep(300, 400)
    threading.Thread(target=run, daemon=True).start()

# --- 2. CLASS NÚT BẤM ẢO (ĐÃ SỬA LỖI) ---
class Button:
    def __init__(self, text, x, y, w, h, color=(0, 255, 0)):
        self.text = text
        self.rect = (x, y, w, h)
        self.color = color
        self.hover_timer = 0
        self.required_time = 20
        self.is_triggered = False

    def draw(self, img):
        x, y, w, h = self.rect
        cv2.rectangle(img, (x, y), (x + w, y + h), self.color, 2)
        if self.hover_timer > 0:
            fill_width = int(w * (self.hover_timer / self.required_time))
            overlay = img.copy()
            cv2.rectangle(overlay, (x, y), (x + fill_width, y + h), self.color, -1)
            cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        font_scale = 1.0
        text_size = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(img, self.text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

    # ĐÃ SỬA: Thêm tham số 'img' vào hàm này
    def check_hover(self, img, landmarks, img_w, img_h):
        pointers = [19, 20] # Ngón trỏ
        is_hovering = False
        x, y, w, h = self.rect
        
        for idx in pointers:
            px = int(landmarks[idx].x * img_w)
            py = int(landmarks[idx].y * img_h)
            
            # Kiểm tra toạ độ
            if x < px < x + w and y < py < y + h:
                is_hovering = True
                # Vẽ vòng tròn ngón tay (Giờ đã có biến img để vẽ)
                cv2.circle(img, (px, py), 15, (0, 255, 255), -1)
                break
        
        if is_hovering:
            self.hover_timer += 1
            if self.hover_timer == 1: play_sound("hover")
            if self.hover_timer >= self.required_time:
                self.hover_timer = 0
                self.is_triggered = True
                play_sound("click")
                return True
        else:
            self.hover_timer = max(0, self.hover_timer - 2)
            self.is_triggered = False
        return False

# --- 3. QUẢN LÝ VẬT THỂ & EFFECT ---
floating_texts = [] 
def add_floating_text(text, x, y, color=(0, 255, 0)):
    floating_texts.append({'text': text, 'pos': [x, y], 'timer': 30, 'color': color})

def update_and_draw_effects(img):
    for ft in floating_texts[:]:
        ft['pos'][1] -= 3; ft['timer'] -= 1
        if ft['timer'] <= 0: floating_texts.remove(ft)
        else: cv2.putText(img, ft['text'], tuple(ft['pos']), 1, 1.5, ft['color'], 3)

class ItemManager:
    def __init__(self):
        self.items = []
        self.spawn_timer = 0
    def update(self, img_w, img_h):
        self.spawn_timer += 1
        if self.spawn_timer > 60: 
            self.spawn_timer = 0
            item_type = 'coin' if random.random() < 0.6 else 'bomb'
            radius = 25 if item_type == 'coin' else 30
            self.items.append({'x': random.randint(50, img_w - 50),'y': -50,'type': item_type,'radius': radius,'speed': random.randint(4, 9)})
        for item in self.items[:]:
            item['y'] += item['speed']
            if item['y'] > img_h + 50: self.items.remove(item)
    def draw(self, img):
        for item in self.items:
            if item['type'] == 'coin':
                cv2.circle(img, (item['x'], item['y']), item['radius'], (0, 255, 255), -1); cv2.circle(img, (item['x'], item['y']), item['radius'], (255, 255, 255), 2); cv2.putText(img, "$", (item['x']-8, item['y']+8), 1, 1.2, (0,0,0), 2)
            else:
                cv2.circle(img, (item['x'], item['y']), item['radius'], (0, 0, 255), -1); cv2.circle(img, (item['x'], item['y']), item['radius'], (0, 0, 0), 2); cv2.putText(img, "X", (item['x']-10, item['y']+10), 1, 1.5, (255,255,255), 2)
    def check_collision(self, landmarks, img_w, img_h):
        hit_info = {'score': 0, 'hit_bomb': False}
        hand_points = [15, 16, 19, 20]; body_points = [0, 11, 12] 
        for item in self.items[:]:
            check_points = hand_points if item['type'] == 'coin' else body_points
            for idx in check_points:
                px = int(landmarks[idx].x * img_w); py = int(landmarks[idx].y * img_h)
                if math.sqrt((px - item['x'])**2 + (py - item['y'])**2) < item['radius'] + 10: 
                    self.items.remove(item)
                    if item['type'] == 'coin': hit_info['score'] += 5; play_sound("coin"); add_floating_text("NICE!", item['x'], item['y'], (0, 255, 255))
                    else: hit_info['hit_bomb'] = True; play_sound("bomb"); add_floating_text("OUCH!", item['x'], item['y'], (0, 0, 255))
                    break 
        return hit_info

# --- 4. HÀM VẼ (STICKMAN & SKELETON) ---
def draw_stickman(img, pose_name, x, y, size=80):
    thickness = 3; color = (255, 255, 255)
    cv2.circle(img, (x, y - size//2), size//4, color, -1) 
    body_bottom = y + size//2
    cv2.line(img, (x, y), (x, body_bottom), color, thickness)
    if pose_name and "SQUAT" in str(pose_name):
        cv2.line(img, (x, body_bottom), (x - size//3, body_bottom + size//3), color, thickness)
        cv2.line(img, (x - size//3, body_bottom + size//3), (x - size//4, body_bottom + size//2 + 10), color, thickness)
        cv2.line(img, (x, body_bottom), (x + size//3, body_bottom + size//3), color, thickness)
        cv2.line(img, (x + size//3, body_bottom + size//3), (x + size//4, body_bottom + size//2 + 10), color, thickness)
    else:
        cv2.line(img, (x, body_bottom), (x - size//3, body_bottom + size), color, thickness)
        cv2.line(img, (x, body_bottom), (x + size//3, body_bottom + size), color, thickness)
    l_sh = (x - size//4, y); r_sh = (x + size//4, y)
    l_el, l_wr = (l_sh[0]-10, l_sh[1]+30), (l_sh[0]-10, l_sh[1]+50)
    r_el, r_wr = (r_sh[0]+10, r_sh[1]+30), (r_sh[0]+10, r_sh[1]+50)
    if pose_name and "Luc Si" in pose_name: l_el, l_wr = (l_sh[0]-20, l_sh[1]), (l_sh[0]-20, l_sh[1]-30); r_el, r_wr = (r_sh[0]+20, r_sh[1]), (r_sh[0]+20, r_sh[1]-30)
    elif pose_name and "Chim Bay" in pose_name: l_el, l_wr = (l_sh[0]-20, l_sh[1]), (l_sh[0]-45, l_sh[1]); r_el, r_wr = (r_sh[0]+20, r_sh[1]), (r_sh[0]+45, r_sh[1])
    elif pose_name and ("Cheo Canh" in pose_name or "CSGT" in pose_name): l_el, l_wr = (l_sh[0]-20, l_sh[1]), (l_sh[0]-45, l_sh[1]); r_el, r_wr = (r_sh[0]+20, r_sh[1]), (r_sh[0]+20, r_sh[1]-30)
    elif pose_name and "SQUAT" in pose_name: l_el, l_wr = (l_sh[0], l_sh[1]+20), (l_sh[0]+10, l_sh[1]+10); r_el, r_wr = (r_sh[0], r_sh[1]+20), (r_sh[0]-10, r_sh[1]+10)
    cv2.line(img, (x, y), l_sh, color, thickness); cv2.line(img, (x, y), r_sh, color, thickness); cv2.line(img, l_sh, l_el, color, thickness); cv2.line(img, l_el, l_wr, color, thickness); cv2.line(img, r_sh, r_el, color, thickness); cv2.line(img, r_el, r_wr, color, thickness)

CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28)]
def draw_neon_skeleton(img, landmarks, combo):
    h, w, _ = img.shape
    if combo < 3: color = (0, 255, 0) 
    elif combo < 6: color = (0, 255, 255) 
    elif combo < 10: color = (0, 165, 255) 
    else: color = (255, 0, 255) 
    thickness = 2 + (combo // 3) 
    for start_idx, end_idx in CONNECTIONS:
        start = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        end = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
        cv2.line(img, start, end, (0, 0, 0), thickness + 4)
        cv2.line(img, start, end, color, thickness)
    key_joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]
    for idx in key_joints:
        cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
        cv2.circle(img, (cx, cy), 8, (255, 255, 255), -1); cv2.circle(img, (cx, cy), 8, color, 2)

# --- 5. CẤU HÌNH ---
mp_pose = mp.solutions.pose
arm_poses = { "Luc Si (2 Tay Vuong)": {"left": 90, "right": 90, "tolerance": 25}, "Chim Bay (2 Tay Thang)": {"left": 170, "right": 170, "tolerance": 25}, "CSGT (1 Thang 1 Vuong)": {"left": 170, "right": 90, "tolerance": 25}}
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def get_high_score():
    if not os.path.exists("highscore.txt"): return 0
    try:
        with open("highscore.txt", "r") as f:
            return int(f.read())
    except:
        return 0

def save_high_score(n):
    try:
        with open("highscore.txt", "w") as f:
            f.write(str(n))
    except:
        pass

score = 0; lives = 3; combo = 0; high_score = get_high_score()
game_state = "MENU"; base_y = 0; shake_timer = 0 
current_task = None; task_type = None; start_time = time.time(); current_duration = 5.0
calibration_frames = 60
item_manager = ItemManager()

# KHỞI TẠO NÚT BẤM (BUTTONS)
btn_start = Button("START GAME", 200, 300, 240, 60, (0, 255, 0))
btn_restart = Button("RESTART", 220, 380, 200, 60, (0, 255, 255))

def new_round():
    global current_task, task_type, start_time
    start_time = time.time()
    if random.random() < 0.7: task_type = "ARM"; current_task = random.choice(list(arm_poses.keys()))
    else: task_type = "LEG"; current_task = "SQUAT XUONG (Ne Dan)"

def reset_game():
    global score, lives, combo, current_duration, game_state, calibration_frames, item_manager
    score = 0; lives = 3; combo = 0; current_duration = 5.0
    calibration_frames = 60; game_state = "CALIBRATION"
    item_manager = ItemManager()

# --- 6. MAIN LOOP ---
cap = cv2.VideoCapture(0)
cv2.namedWindow('AI ExerGame Kiosk', cv2.WINDOW_NORMAL)

with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6, model_complexity=1) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        
        if shake_timer > 0:
            shake_x = random.randint(-15, 15); shake_y = random.randint(-15, 15)
            M = np.float32([[1, 0, shake_x], [0, 1, shake_y]])
            frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
            shake_timer -= 1
        
        if game_state == "MENU": frame = cv2.convertScaleAbs(frame, alpha=0.6, beta=0)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h, w, _ = image.shape

        # === MENU ===
        if game_state == "MENU":
            cv2.rectangle(image, (0, h//2-100), (w, h//2+100), (0,0,0), -1)
            cv2.putText(image, "AI EXER-GAME", (w//2-250, h//2-20), 1, 3, (0,255,255), 5)
            cv2.putText(image, f"TOP SCORE: {high_score}", (w//2-120, h//2+150), 1, 1, (255,215,0), 2)
            
            # Vẽ nút START
            btn_start.draw(image)
            if results.pose_landmarks:
                # ĐÃ SỬA: Truyền biến 'image' vào hàm check_hover
                if btn_start.check_hover(image, results.pose_landmarks.landmark, w, h):
                    reset_game()

        # === CALIBRATION ===
        elif game_state == "CALIBRATION":
             if results.pose_landmarks:
                current_hip_y = (results.pose_landmarks.landmark[23].y + results.pose_landmarks.landmark[24].y) / 2
                cv2.putText(image, "CALIBRATING...", (w//2-150, h//2-50), 1, 1.5, (0,255,255), 3)
                cv2.putText(image, str(calibration_frames//10), (w//2-20, h//2+50), 1, 3, (255,255,255), 5)
                calibration_frames -= 1
                if calibration_frames <= 0:
                    base_y = current_hip_y; game_state = "PLAYING"; new_round(); play_sound("score")

        # === PLAYING ===
        elif game_state == "PLAYING":
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                draw_neon_skeleton(image, landmarks, combo)
                draw_stickman(image, current_task, 80, 180, 60)
                cv2.rectangle(image, (20, 120), (140, 260), (255, 255, 255), 2)
                
                l_hip, r_hip = landmarks[23].y, landmarks[24].y
                success = False
                if task_type == "ARM":
                    l_ang = calculate_angle([landmarks[11].x, landmarks[11].y], [landmarks[13].x, landmarks[13].y], [landmarks[15].x, landmarks[15].y])
                    r_ang = calculate_angle([landmarks[12].x, landmarks[12].y], [landmarks[14].x, landmarks[14].y], [landmarks[16].x, landmarks[16].y])
                    t_l, t_r, tol = arm_poses[current_task]["left"], arm_poses[current_task]["right"], arm_poses[current_task]["tolerance"]
                    if (((t_l-tol)<l_ang<(t_l+tol)) and ((t_r-tol)<r_ang<(t_r+tol))) or (((t_r-tol)<l_ang<(t_r+tol)) and ((t_l-tol)<r_ang<(t_l+tol))): success = True
                elif task_type == "LEG":
                    if ((l_hip + r_hip)/2) > (base_y + 0.15): success = True
                    else: cv2.line(image, (0, int((base_y+0.15)*h)), (w, int((base_y+0.15)*h)), (0,255,255), 2)

                item_manager.update(w, h)
                item_manager.draw(image)
                hit_result = item_manager.check_collision(landmarks, w, h)
                
                if hit_result['score'] > 0: score += hit_result['score']
                if hit_result['hit_bomb']:
                    lives -= 1; shake_timer = 10; combo = 0
                    if lives == 0: 
                        game_state = "GAMEOVER"; play_sound("gameover")
                        if score > high_score: high_score = score; save_high_score(high_score)

                time_left = current_duration - (time.time() - start_time)
                if time_left <= 0:
                    if success:
                        score += 1; combo += 1; shake_timer = 2
                        play_sound("combo" if combo > 2 else "score")
                        txt = f"+1" if combo < 3 else f"COMBO x{combo}"
                        color = (0, 255, 0) if combo < 5 else (255, 0, 255)
                        add_floating_text(txt, w//2 - 50, h//2, color)
                        if score % 3 == 0 and current_duration > 2.0: current_duration -= 0.5
                    else:
                        lives -= 1; combo = 0; shake_timer = 5
                        play_sound("fail"); add_floating_text("MISS!", w//2 - 50, h//2, (0, 0, 255))
                        if lives == 0:
                            game_state = "GAMEOVER"; play_sound("gameover")
                            if score > high_score: high_score = score; save_high_score(high_score)
                    new_round()

                cv2.rectangle(image, (0, 0), (w, 20), (50, 50, 50), -1)
                cv2.rectangle(image, (0, 0), (int(max(0, time_left)/current_duration*w), 20), (0, 255, 0) if time_left>2 else (0,0,255), -1)
                cv2.putText(image, f"SCORE: {score}", (20, 60), 1, 1.5, (255, 255, 255), 2)
                if combo > 1: cv2.putText(image, f"{combo} COMBO", (w-250, 60), 1, 1.5, (255, 0, 255), 3)
                task_txt = current_task if task_type == "ARM" else "SQUAT DOWN!"
                cv2.putText(image, task_txt, (w//2 - 150, 100), 1, 1, (0, 255, 255), 2)
                cv2.putText(image, "<3 " * lives, (20, 100), 1, 1, (0, 0, 255), 2)

        # === GAMEOVER ===
        elif game_state == "GAMEOVER":
            overlay = image.copy(); cv2.rectangle(overlay, (0, 0), (w, h), (0,0,0), -1)
            image = cv2.addWeighted(overlay, 0.8, image, 0.2, 0)
            
            cv2.putText(image, "GAME OVER", (w//2-180, h//2-20), 1, 3, (0,0,255), 8)
            cv2.putText(image, "GAME OVER", (w//2-180, h//2-20), 1, 3, (255,255,255), 2)
            cv2.putText(image, f"SCORE: {score}", (w//2-100, h//2+50), 1, 2, (255,255,255), 2)
            if score == high_score and score > 0: cv2.putText(image, "NEW RECORD!", (w//2-150, h//2+100), 1, 1.5, (0,255,0), 3)
            
            # Vẽ nút RESTART
            btn_restart.draw(image)
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                # ĐÃ SỬA: Truyền biến 'image' vào hàm check_hover
                if btn_restart.check_hover(image, results.pose_landmarks.landmark, w, h):
                    reset_game()

        update_and_draw_effects(image)
        cv2.imshow('AI ExerGame Kiosk', image)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'): break
        # Vẫn giữ phím Space/R làm dự phòng
        if key == ord(' ') and game_state == "MENU": reset_game()
        if key == ord('r') and game_state == "GAMEOVER": reset_game()

cap.release()
cv2.destroyAllWindows()