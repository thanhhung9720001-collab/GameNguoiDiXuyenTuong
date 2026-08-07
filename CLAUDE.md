# Quy tắc làm việc cho dự án này

## 1. Bắt buộc: chia nhỏ giai đoạn và báo trước

Mọi công việc trong dự án này **phải được chia thành từng giai đoạn nhỏ**. Trước khi
bắt đầu **mỗi** giai đoạn, AI phải trình bày rõ:

- **Sẽ làm gì**: liệt kê cụ thể các file/thay đổi trong giai đoạn đó.
- **Cần người dùng làm gì**: các việc AI không tự làm được (thêm file ảnh, cấp quyền
  camera, chạy thử, xác nhận thiết kế…). Nếu không cần gì thì ghi rõ "không cần".
- **Kết quả kiểm chứng**: sau giai đoạn đó chạy lệnh nào / mở màn hình nào để thấy
  được kết quả.

Không gộp nhiều giai đoạn vào một lần làm. Làm xong một giai đoạn thì dừng lại, báo
cáo kết quả, chờ người dùng xác nhận rồi mới sang giai đoạn kế tiếp.

Ngôn ngữ trao đổi: **tiếng Việt**.

## 2. Bối cảnh dự án

Web game điều khiển bằng tư thế cơ thể qua webcam (MediaPipe Pose), phục vụ sự kiện
CLB Developer. Định hướng hiện tại: **làm lại hoàn toàn thành Flappy Bird**.

- Người chơi **quạt hai tay như cánh chim** trước webcam → chim bay lên.
  (Trước đây từng dùng động tác nhảy; đã bỏ vì quạt tay đỡ mệt và dễ nhận diện hơn.)
- Vẫn giữ fallback bằng **phím Space / click chuột** khi không có webcam.
- Có **khung camera nhỏ ở góc dưới bên phải** để người chơi tự thấy mình.
- Giao diện dùng **sprite ảnh gốc Flappy Bird** (người dùng tự thêm file PNG).

## 3. Cấu trúc mã

- Game chạy ở `web/public/game/` (`index.html`, `app.js`, `styles.css`) — HTML/JS
  thuần, không qua React.
- `web/` là app vinext + Vite + Cloudflare Worker; root `/` redirect sang
  `/game/index.html` (xử lý trong app router).
- Kiểm thử: `cd web && npm test` (build rồi chạy `tests/rendered-html.test.mjs`).
  File test này kiểm tra nội dung `public/game/*` — **đổi game thì phải cập nhật test
  tương ứng**, không để test kiểm tra tính năng đã bị xoá.
- `human_tetris.py` là bản prototype Python cũ, không còn nằm trong hướng phát triển.
