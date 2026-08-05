# Kế hoạch phát triển web — Game Người Đi Xuyên Tường

## Mục tiêu

Chuyển prototype Python/OpenCV hiện tại thành một web game có giao diện hiện đại,
bắt mắt và có thể chơi trực tiếp qua đường link. Người chơi cấp quyền webcam,
thực hiện tư thế theo yêu cầu và nhận điểm ngay trong trình duyệt; video không cần
được gửi lên máy chủ.

## Định hướng sản phẩm

- Tên tạm thời: **Người Đi Xuyên Tường — AI ExerGame**.
- Phong cách hình ảnh: game show neon, năng lượng cao, dễ hiểu khi chơi lần đầu.
- Thiết bị ưu tiên: laptop/desktop có webcam; hỗ trợ mobile ở mức phù hợp.
- Quyền riêng tư: xử lý pose cục bộ trong trình duyệt, nêu rõ điều này trên màn
  hình cấp quyền camera.

## Kiến trúc đề xuất

| Hạng mục | Lựa chọn |
| --- | --- |
| Frontend | React + TypeScript + Vite |
| Render game | HTML Canvas (có thể dùng CSS/React cho UI) |
| Pose detection | MediaPipe Pose Landmarker cho Web |
| Lưu tiến trình giai đoạn đầu | `localStorage` |
| Backend giai đoạn sau | Supabase (tài khoản, leaderboard) |
| Hosting | Vercel hoặc Cloudflare Pages |

> Không deploy trực tiếp mã Python hiện tại làm game web. Gameplay và dữ liệu được
> giữ lại, còn camera, nhận diện tư thế và giao diện sẽ được xây dựng lại bằng web.

## Phạm vi MVP

1. Landing page có giới thiệu ngắn, nút **Chơi ngay** và hướng dẫn bật webcam.
2. Màn hình game lấy video webcam, hiển thị skeleton và một tư thế mục tiêu.
3. Ba thử thách ban đầu: hai tay vuông, hai tay thẳng và squat.
4. Đếm ngược, kiểm tra tư thế, điểm số, combo, máu và game over.
5. Hiệu ứng chuyển cảnh, âm thanh web và responsive cơ bản.
6. Lưu high score ở `localStorage`.
7. Deploy HTTPS để API webcam hoạt động trên link công khai.

## Các giai đoạn thực hiện

### Giai đoạn 0 — Chuẩn bị

- [ ] Đọc và đối chiếu gameplay Python với README hiện có.
- [ ] Xác định các pose, ngưỡng góc và luật tính điểm dùng cho MVP.
- [ ] Chuẩn hoá repo: README mới, `.gitignore`, lệnh cài/chạy rõ ràng.
- [ ] Tạo cấu trúc React/Vite mà không xoá prototype Python.

### Giai đoạn 1 — Giao diện và game loop

- [ ] Xây dựng design system: màu neon, typography, panel, nút, trạng thái.
- [ ] Tạo landing page, loading screen và flow xin quyền webcam.
- [ ] Làm màn hình game Canvas: timer, HUD, target pose và skeleton người chơi.
- [ ] Thiết kế trạng thái: menu, calibration, playing, paused và game over.
- [ ] Bảo đảm có thông báo rõ ràng khi không có webcam hoặc người dùng từ chối quyền.

### Giai đoạn 2 — Nhận diện tư thế

- [ ] Tích hợp MediaPipe Pose Landmarker phiên bản web.
- [ ] Chuẩn hoá landmark theo kích thước cơ thể/camera.
- [ ] Implement các bộ chấm pose dựa trên góc khuỷu tay và độ hạ hông.
- [ ] Yêu cầu người chơi giữ pose đúng trong vài frame để tránh nhận diện ngẫu nhiên.
- [ ] Hiển thị phản hồi trực quan: đúng/sai, vùng cần điều chỉnh và độ tự tin.

### Giai đoạn 3 — Gameplay mở rộng

- [ ] Chuyển thử thách thành cơ chế "tường tiến tới" và silhouette mục tiêu.
- [ ] Thêm coin, bomb, shield, heart, combo và hiệu ứng hạt.
- [ ] Thêm shop skin, trang bị skin và lưu tiến trình bằng `localStorage`.
- [ ] Thêm âm thanh qua Web Audio; hỗ trợ tắt âm thanh.
- [ ] Cân bằng độ khó theo thời gian và kết quả người chơi.

### Giai đoạn 4 — Chất lượng và deploy

- [ ] Kiểm thử Chrome/Edge trên Windows và Chrome Safari trên mobile.
- [ ] Kiểm tra webcam, tốc độ khung hình, quyền riêng tư và xử lý lỗi mạng.
- [ ] Tối ưu kích thước bundle và tải model nhận diện pose.
- [ ] Viết README: yêu cầu hệ thống, local setup, cách deploy.
- [ ] Deploy preview, kiểm thử đường link HTTPS, sau đó phát hành chính thức.

### Giai đoạn 5 — Hướng phát triển sau MVP

- [ ] Đăng nhập và leaderboard trực tuyến.
- [ ] Daily challenge, thành tích và chia sẻ điểm.
- [ ] Nhiều tư thế, level/chủ đề tường và accessibility options.
- [ ] Chế độ thi đấu nhóm hoặc sự kiện cho CLB.
- [ ] Analytics tôn trọng quyền riêng tư để cân bằng độ khó.

## Tiêu chí hoàn thành MVP

- Người dùng mở một URL HTTPS, cấp quyền webcam và bắt đầu chơi không cần cài đặt.
- Pose được xử lý cục bộ với phản hồi thời gian thực, mượt ở khoảng 24 FPS trở lên
  trên laptop phổ thông.
- Có vòng chơi hoàn chỉnh từ menu đến game over, cùng high score được lưu lại.
- Có hướng dẫn, fallback lỗi camera và thông tin riêng tư rõ ràng.
- Giao diện nhất quán, đủ chỉn chu để trình diễn cho CLB/người dùng thử.

## Các quyết định cần chốt khi bắt đầu triển khai

1. Tên/nhận diện thương hiệu chính thức và bộ màu mong muốn.
2. Dùng Vercel hay Cloudflare Pages để hosting.
3. MVP có cần leaderboard online ngay hay chỉ lưu cục bộ.
4. Đối tượng ưu tiên: demo cho CLB trên laptop hay public/mobile rộng rãi.
