"""
ASSIGNMENT 2 MEGA UPDATE: SHOP & SKINS
Tính năng mới:
- Save System (JSON): Lưu Tiền, Kỷ lục, và Skin đã mua.
- Shop UI: Giao diện cửa hàng mua skin bằng tay ảo.
- Skin System: Đổi màu nhân vật (Fire, Matrix, Gold...).
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
import json # Dùng để lưu dữ liệu phức tạp

# --- 1. HỆ THỐNG ÂM THANH ---
def play_sound(type):
    def run():
        if type == "score": winsound.Beep(1000, 50)
        elif type == "coin": winsound.Beep(2000, 50)
        elif type == "bomb": winsound.Beep(150, 400)
        elif type == "click": winsound.Beep(800, 100)
        elif type == "buy": winsound.Beep(1200, 150); winsound.Beep(1500, 150) # Tiếng mua đồ
        elif type == "error": winsound.Beep(200, 300) # Tiền không đủ
        elif type == "hover": winsound.Beep(600, 30)
        elif type == "shield_up": winsound.Beep(3000, 200)
        elif type == "shield_down": winsound.Beep(500, 500)
        elif type == "heal": winsound.Beep(600, 100); winsound.Beep(800, 100)
        elif type == "gameover": 
            winsound.Beep(500, 150); winsound.Beep(400, 150); winsound.Beep(300, 400)
    threading.Thread(target=run, daemon=True).start()

# --- 2. QUẢN LÝ DỮ LIỆU (SAVE/LOAD JSON) ---
DATA_FILE = "gamedata.json"
DEFAULT_DATA = {
    "high_score": 0,
    "money": 0,
    "inventory": ["default"], # Danh sách skin đã sở hữu
    "equipped": "default"     # Skin đang dùng
}

def load_data():
    if not os.path.exists(DATA_FILE):
        return DEFAULT_DATA.copy()
    try:
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    except:
        return DEFAULT_DATA.copy()

def save_data(data):
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(data, f)
    except:
        pass

# Biến toàn cục dữ liệu
game_data = load_data()

# --- 3. ĐỊNH NGHĨA SKINS ---
SKINS = {
    "default": {"name": "DEFAULT", "price": 0, "color": (255, 255, 255)}, # Trắng
    "fire":    {"name": "FIRE",    "price": 50, "color": (0, 165, 255)},   # Cam
    "matrix":  {"name": "MATRIX",  "price": 100,"color": (0, 255, 0)},     # Xanh lá
    "gold":    {"name": "GOLD",    "price": 200,"color": (0, 215, 255)},   # Vàng
    "neon":    {"name": "NEON",    "price": 500,"color": (255, 0, 255)}    # Tím
}

# --- 4. CLASS NÚT BẤM ẢO ---
class Button:
    def __init__(self, text, x, y, w, h, color=(0, 255, 0), data=None):
        self.text = text
        self.rect = (x, y, w, h)
        self.base_color = color
        self.hover_timer = 0
        self.required_time = 15 # Giảm thời gian giữ xuống 15 frame cho nhanh
        self.is_triggered = False
        self.data = data # Dữ liệu phụ (ví dụ: tên skin)

    def draw(self, img, is_locked=False):
        x, y, w, h = self.rect
        color = (100, 100, 100) if is_locked else self.base_color
        
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        
        # Hiệu ứng Loading
        if self.hover_timer > 0 and not is_locked:
            fill_width = int(w * (self.hover_timer / self.required_time))
            overlay = img.copy()
            cv2.rectangle(overlay, (x, y), (x + fill_width, y + h), color, -1)
            cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

        font_scale = 0.8
        text_size = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(img, self.text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

    def check_hover(self, img, landmarks, img_w, img_h):
        pointers = [19, 20] 
        is_hovering = False
        x, y, w, h = self.rect
        
        for idx in pointers:
            px = int(landmarks[idx].x * img_w)
            py = int(landmarks[idx].y * img_h)
            if x < px < x + w and y < py < y + h:
                is_hovering = True
                cv2.circle(img, (px, py), 15, (0, 255, 255), -1)
                break
        
        if is_hovering:
            self.hover_timer += 1
            if self.hover_timer == 1: play_sound("hover")
            if self.hover_timer >= self.required_time:
                self.hover_timer = 0
                self.is_triggered = True
                return True
        else:
            self.hover_timer = max(0, self.hover_timer - 2)
            self.is_triggered = False
        return False

# --- 5. QUẢN LÝ VẬT THỂ & HIỆU ỨNG ---
floating_texts = [] 
def add_text(text, x, y, color=(0, 255, 0)):
    floating_texts.append({'text': text, 'pos': [x, y], 'timer': 30, 'color': color})

def draw_effects(img):
    for ft in floating_texts[:]:
        ft['pos'][1] -= 3; ft['timer'] -= 1
        if ft['timer'] <= 0: floating_texts.remove(ft)
        else: cv2.putText(img, ft['text'], tuple(ft['pos']), 1, 1.5, ft['color'], 3)

class ItemManager:
    def __init__(self):
        self.items = []; self.timer = 0
    def update(self, w, h):
        self.timer += 1
        if self.timer > 50: 
            self.timer = 0
            rand = random.random()
            if rand < 0.4: type = 'coin'; radius = 25
            elif rand < 0.7: type = 'bomb'; radius = 35
            elif rand < 0.9: type = 'shield'; radius = 30
            else: type = 'heart'; radius = 30
            self.items.append({'x': random.randint(50, w-50),'y': -50,'type': type,'radius': radius,'speed': random.randint(4, 9)})
        for item in self.items[:]:
            item['y'] += item['speed']
            if item['y'] > h + 50: self.items.remove(item)
    def draw(self, img):
        for item in self.items:
            x, y, r, t = item['x'], item['y'], item['radius'], item['type']
            if t == 'coin': c = (0, 255, 255); txt = "$"
            elif t == 'bomb': c = (0, 0, 255); txt = "X"
            elif t == 'shield': c = (255, 0, 0); txt = "S"
            elif t == 'heart': c = (0, 0, 255); txt = "H"
            cv2.circle(img, (x, y), r, c, -1)
            cv2.circle(img, (x, y), r, (255, 255, 255), 2)
            cv2.putText(img, txt, (x-10, y+10), 1, 1.5, (0,0,0) if t=='coin' else (255,255,255), 2)
    def check_hit(self, landmarks, w, h, has_shield):
        hit = {'score': 0, 'damage': False, 'shield': False, 'heal': False}
        hands = [15, 16, 19, 20]; body = [0, 11, 12]
        for item in self.items[:]:
            pts = body if item['type'] in ['bomb', 'shield', 'heart'] else hands
            for idx in pts:
                px = int(landmarks[idx].x * w); py = int(landmarks[idx].y * h)
                if math.sqrt((px - item['x'])**2 + (py - item['y'])**2) < item['radius'] + 15:
                    self.items.remove(item)
                    if item['type'] == 'coin':
                        hit['score'] += 5; play_sound("coin"); add_text("+5", item['x'], item['y'], (0, 255, 255))
                    elif item['type'] == 'bomb':
                        if has_shield: play_sound("shield_down"); add_text("BLOCKED!", item['x'], item['y'], (255, 0, 0))
                        else: hit['damage'] = True; play_sound("bomb"); add_text("OUCH!", item['x'], item['y'], (0, 0, 255))
                    elif item['type'] == 'shield':
                        hit['shield'] = True; play_sound("shield_up"); add_text("SHIELD!", item['x'], item['y'], (255, 0, 0))
                    elif item['type'] == 'heart':
                        hit['heal'] = True; play_sound("heal"); add_text("+1 LIFE", item['x'], item['y'], (0, 0, 255))
                    break
        return hit

# --- 6. HÀM VẼ VISUAL (CÓ SKIN) ---
def draw_stickman(img, pose_name, x, y, size=80, color=(255, 255, 255)):
    thickness = 3
    cv2.circle(img, (x, y - size//2), size//4, color, -1) 
    body_bottom = y + size//2
    cv2.line(img, (x, y), (x, body_bottom), color, thickness)
    # (Giản lược đoạn vẽ chân tay để code ngắn gọn, logic giữ nguyên)
    # Vẽ chân
    if pose_name and "SQUAT" in str(pose_name):
        cv2.line(img, (x, body_bottom), (x - size//3, body_bottom + size//3), color, thickness)
        cv2.line(img, (x - size//3, body_bottom + size//3), (x - size//4, body_bottom + size//2 + 10), color, thickness)
        cv2.line(img, (x, body_bottom), (x + size//3, body_bottom + size//3), color, thickness)
        cv2.line(img, (x + size//3, body_bottom + size//3), (x + size//4, body_bottom + size//2 + 10), color, thickness)
    else:
        cv2.line(img, (x, body_bottom), (x - size//3, body_bottom + size), color, thickness)
        cv2.line(img, (x, body_bottom), (x + size//3, body_bottom + size), color, thickness)
    # Vẽ tay
    l_sh = (x - size//4, y); r_sh = (x + size//4, y)
    l_el, l_wr = (l_sh[0]-10, l_sh[1]+30), (l_sh[0]-10, l_sh[1]+50)
    r_el, r_wr = (r_sh[0]+10, r_sh[1]+30), (r_sh[0]+10, r_sh[1]+50)
    
    if pose_name and "Luc Si" in pose_name: l_el, l_wr = (l_sh[0]-20, l_sh[1]), (l_sh[0]-20, l_sh[1]-30); r_el, r_wr = (r_sh[0]+20, r_sh[1]), (r_sh[0]+20, r_sh[1]-30)
    elif pose_name and "Chim Bay" in pose_name: l_el, l_wr = (l_sh[0]-20, l_sh[1]), (l_sh[0]-45, l_sh[1]); r_el, r_wr = (r_sh[0]+20, r_sh[1]), (r_sh[0]+45, r_sh[1])
    elif pose_name and "CSGT" in pose_name: l_el, l_wr = (l_sh[0]-20, l_sh[1]), (l_sh[0]-45, l_sh[1]); r_el, r_wr = (r_sh[0]+20, r_sh[1]), (r_sh[0]+20, r_sh[1]-30)
    elif pose_name and "SQUAT" in pose_name: l_el, l_wr = (l_sh[0], l_sh[1]+20), (l_sh[0]+10, l_sh[1]+10); r_el, r_wr = (r_sh[0], r_sh[1]+20), (r_sh[0]-10, r_sh[1]+10)
    
    cv2.line(img, (x, y), l_sh, color, thickness); cv2.line(img, (x, y), r_sh, color, thickness)
    cv2.line(img, l_sh, l_el, color, thickness); cv2.line(img, l_el, l_wr, color, thickness)
    cv2.line(img, r_sh, r_el, color, thickness); cv2.line(img, r_el, r_wr, color, thickness)

CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28)]
def draw_neon_skeleton(img, landmarks, combo, skin_color):
    h, w, _ = img.shape
    # Nếu combo cao thì nhấp nháy, nếu không dùng màu skin
    color = skin_color
    if combo >= 10: color = (255, 0, 255) # Fever Mode đè lên skin
    
    thick = 2 + (combo // 3) 
    for s, e in CONNECTIONS:
        start = (int(landmarks[s].x * w), int(landmarks[s].y * h))
        end = (int(landmarks[e].x * w), int(landmarks[e].y * h))
        cv2.line(img, start, end, (0, 0, 0), thick + 4); cv2.line(img, start, end, color, thick)
    for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]:
        cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
        cv2.circle(img, (cx, cy), 8, (255, 255, 255), -1); cv2.circle(img, (cx, cy), 8, color, 2)

# --- 7. CẤU HÌNH & LOGIC ---
mp_pose = mp.solutions.pose
arm_poses = { "Luc Si (2 Tay Vuong)": {"left": 90, "right": 90, "tolerance": 25}, "Chim Bay (2 Tay Thang)": {"left": 170, "right": 170, "tolerance": 25}, "CSGT (1 Thang 1 Vuong)": {"left": 170, "right": 90, "tolerance": 25}}
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

score = 0; lives = 3; combo = 0;
game_state = "MENU"; base_y = 0; shake_timer = 0 
current_task = None; task_type = None; start_time = time.time(); current_duration = 5.0
calib_timer = 60; items = ItemManager(); shield_timer = 0

# BUTTONS
btn_start = Button("PLAY", 220, 200, 200, 60)
btn_shop = Button("SHOP", 220, 280, 200, 60, (0, 165, 255))
btn_back = Button("BACK", 20, 20, 150, 50, (0, 0, 255))
btn_restart = Button("RESTART", 220, 380, 200, 60, (0,255,255))

# SHOP BUTTONS (Tạo động)
shop_buttons = []
y_offset = 100
for key, val in SKINS.items():
    btn = Button(f"{val['name']} ${val['price']}", 150, y_offset, 340, 50, val['color'], data=key)
    shop_buttons.append(btn)
    y_offset += 70

def new_round():
    global current_task, task_type, start_time
    start_time = time.time()
    if random.random() < 0.7: task_type = "ARM"; current_task = random.choice(list(arm_poses.keys()))
    else: task_type = "LEG"; current_task = "SQUAT XUONG (Ne Dan)"

def reset_game():
    global score, lives, combo, current_duration, game_state, calib_timer, items, shield_timer
    score = 0; lives = 3; combo = 0; current_duration = 5.0
    calib_timer = 60; game_state = "CALIBRATION"; items = ItemManager(); shield_timer = 0

# --- 8. MAIN LOOP ---
cap = cv2.VideoCapture(0)
cv2.namedWindow('AI ExerGame Shop', cv2.WINDOW_NORMAL)

with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6, model_complexity=1) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        
        # Shake & Background
        if shake_timer > 0:
            sx = random.randint(-15, 15); sy = random.randint(-15, 15)
            M = np.float32([[1, 0, sx], [0, 1, sy]])
            frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
            shake_timer -= 1
        
        if game_state in ["MENU", "SHOP"]: frame = cv2.convertScaleAbs(frame, alpha=0.6, beta=0)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h, w, _ = image.shape

        # === MENU ===
        if game_state == "MENU":
            cv2.putText(image, "AI EXER-GAME", (w//2-220, 100), 1, 3, (0,255,255), 5)
            cv2.putText(image, f"COINS: ${game_data['money']}", (w-250, 50), 1, 1.5, (0,255,255), 2)
            cv2.putText(image, f"SKIN: {game_data['equipped']}", (20, 50), 1, 1, SKINS[game_data['equipped']]['color'], 2)
            
            btn_start.draw(image); btn_shop.draw(image)
            
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                if btn_start.check_hover(image, lm, w, h): reset_game()
                if btn_shop.check_hover(image, lm, w, h): game_state = "SHOP"

        # === SHOP ===
        elif game_state == "SHOP":
            cv2.putText(image, "SKIN SHOP", (w//2-150, 60), 1, 2, (255,255,255), 3)
            cv2.putText(image, f"MONEY: ${game_data['money']}", (w-250, 50), 1, 1.2, (0,255,255), 2)
            btn_back.draw(image)
            
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                if btn_back.check_hover(image, lm, w, h): game_state = "MENU"
                
                # Logic mua skin
                for btn in shop_buttons:
                    skin_key = btn.data
                    skin_info = SKINS[skin_key]
                    is_owned = skin_key in game_data['inventory']
                    is_equipped = game_data['equipped'] == skin_key
                    
                    # Cập nhật text nút
                    if is_equipped: btn.text = f"{skin_info['name']} (EQUIPPED)"
                    elif is_owned: btn.text = f"{skin_info['name']} (OWNED)"
                    else: btn.text = f"{skin_info['name']} ${skin_info['price']}"
                    
                    btn.draw(image)
                    
                    if btn.check_hover(image, lm, w, h):
                        if is_owned:
                            game_data['equipped'] = skin_key
                            save_data(game_data)
                            play_sound("click")
                        elif game_data['money'] >= skin_info['price']:
                            game_data['money'] -= skin_info['price']
                            game_data['inventory'].append(skin_key)
                            game_data['equipped'] = skin_key
                            save_data(game_data)
                            play_sound("buy")
                        else:
                            play_sound("error")

        # === GAMEPLAY ===
        elif game_state == "CALIBRATION" or game_state == "PLAYING":
            # (Giữ nguyên logic chơi game nhưng thêm +Tiền vào game_data)
            if game_state == "CALIBRATION":
                 if results.pose_landmarks:
                    current_hip_y = (results.pose_landmarks.landmark[23].y + results.pose_landmarks.landmark[24].y) / 2
                    cv2.putText(image, "CALIBRATING...", (w//2-150, h//2-50), 1, 1.5, (0,255,255), 3)
                    cv2.putText(image, str(calib_timer//10), (w//2-20, h//2+50), 1, 3, (255,255,255), 5)
                    calib_timer -= 1
                    if calib_timer <= 0:
                        base_y = current_hip_y; game_state = "PLAYING"; new_round(); play_sound("score")
            
            elif game_state == "PLAYING":
                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    # Vẽ Skin đang trang bị
                    skin_color = SKINS[game_data['equipped']]['color']
                    draw_neon_skeleton(image, lm, combo, skin_color)
                    draw_stickman(image, current_task, 80, 180, 60, skin_color)
                    
                    if shield_timer > 0:
                        shield_timer -= 1
                        hip_x = int((lm[23].x + lm[24].x) * w / 2)
                        hip_y = int((lm[23].y + lm[24].y) * h / 2)
                        if shield_timer > 60 or (shield_timer // 5) % 2 == 0:
                            cv2.circle(image, (hip_x, hip_y), 250, (255, 0, 0), 5)

                    cv2.rectangle(image, (20, 120), (140, 260), (255, 255, 255), 2)
                    l_hip, r_hip = lm[23].y, lm[24].y
                    success = False
                    if task_type == "ARM":
                        l_ang = calculate_angle([lm[11].x, lm[11].y], [lm[13].x, lm[13].y], [lm[15].x, lm[15].y])
                        r_ang = calculate_angle([lm[12].x, lm[12].y], [lm[14].x, lm[14].y], [lm[16].x, lm[16].y])
                        t_l, t_r, tol = arm_poses[current_task]["left"], arm_poses[current_task]["right"], arm_poses[current_task]["tolerance"]
                        if (((t_l-tol)<l_ang<(t_l+tol)) and ((t_r-tol)<r_ang<(t_r+tol))) or (((t_r-tol)<l_ang<(t_r+tol)) and ((t_l-tol)<r_ang<(t_l+tol))): success = True
                    elif task_type == "LEG":
                        if ((l_hip + r_hip)/2) > (base_y + 0.15): success = True
                        else: cv2.line(image, (0, int((base_y+0.15)*h)), (w, int((base_y+0.15)*h)), (0,255,255), 2)

                    items.update(w, h); items.draw(image)
                    hit = items.check_hit(lm, w, h, shield_timer > 0)
                    
                    if hit['score'] > 0: 
                        score += hit['score']
                        # CỘNG TIỀN VÀO TÀI KHOẢN
                        game_data['money'] += hit['score'] 
                    
                    if hit['heal']: lives = min(3, lives + 1)
                    if hit['shield']: shield_timer = 300
                    
                    if hit['damage']:
                        lives -= 1; shake_timer = 10; combo = 0
                        if lives == 0: 
                            game_state = "GAMEOVER"; play_sound("gameover")
                            if score > game_data['high_score']: 
                                game_data['high_score'] = score
                            save_data(game_data) # LƯU GAME

                    time_left = current_duration - (time.time() - start_time)
                    if time_left <= 0:
                        if success:
                            score += 1; combo += 1; shake_timer = 2; play_sound("combo" if combo > 2 else "score")
                            add_text(f"+1" if combo < 3 else f"COMBO x{combo}", w//2 - 50, h//2, (0, 255, 0))
                            if score % 3 == 0 and current_duration > 2.0: current_duration -= 0.5
                        else:
                            lives -= 1; combo = 0; shake_timer = 5; play_sound("fail"); add_text("MISS!", w//2 - 50, h//2, (0, 0, 255))
                            if lives == 0:
                                game_state = "GAMEOVER"; play_sound("gameover")
                                if score > game_data['high_score']: game_data['high_score'] = score
                                save_data(game_data) # LƯU GAME
                        new_round()

                    cv2.rectangle(image, (0, 0), (w, 20), (50, 50, 50), -1)
                    cv2.rectangle(image, (0, 0), (int(max(0, time_left)/current_duration*w), 20), (0, 255, 0) if time_left>2 else (0,0,255), -1)
                    cv2.putText(image, f"SCORE: {score}", (20, 60), 1, 1.5, (255, 255, 255), 2)
                    cv2.putText(image, f"TOP: {game_data['high_score']}", (w - 220, 60), 1, 1.5, (0, 255, 255), 2)
                    if combo > 1: cv2.putText(image, f"{combo} COMBO", (w-250, 100), 1, 1.5, (255, 0, 255), 3)
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
            cv2.putText(image, f"EARNED: ${score} COINS", (w//2-150, h//2+100), 1, 1.5, (0,255,255), 2)
            
            btn_restart.draw(image)
            if results.pose_landmarks and btn_restart.check_hover(image, results.pose_landmarks.landmark, w, h): reset_game()

        draw_effects(image)
        cv2.imshow('AI ExerGame Shop', image)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'): break
        if key == ord(' ') and game_state == "MENU": reset_game()
        if key == ord('r') and game_state == "GAMEOVER": reset_game()

cap.release()
cv2.destroyAllWindows()