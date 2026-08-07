# Flappy Bird — Quạt Tay Để Bay

Flappy Bird chơi bằng cả cơ thể: người chơi **quạt hai tay như cánh chim** trước webcam
thì con chim mới bay lên. Làm cho sự kiện CLB Developer — dễ hiểu, chơi được ngay từ
lượt đầu, và vui khi đứng xem người khác chơi.

- Giao diện dùng đúng sprite gốc Flappy Bird, dựng lại ở khung ngang 16:9.
- Khung webcam nhỏ ở góc dưới bên phải để người chơi tự thấy mình.
- Không có webcam vẫn chơi được bằng **Space / click chuột**.
- Toàn bộ xử lý hình ảnh chạy cục bộ trong trình duyệt; video không gửi đi đâu cả.
- **Chạy được offline**: MediaPipe và mô hình nhận diện đã nhúng sẵn trong repo
  (`web/public/game/vendor/`), không cần mạng — xem README trong thư mục đó.

## Chạy thử

Cần **Node.js ≥ 22.13.0**. Ngoài ra không phải cài gì thêm — không Python, không
tải mô hình AI, không cấu hình.

```bash
cd web
npm install   # cần mạng, chỉ một lần
npm run dev
```

Mở http://localhost:3000 — trang gốc tự chuyển sang `/game/index.html`, rồi cho phép
quyền camera.

Sau khi `npm install` xong thì **game chạy được offline hoàn toàn**, vì MediaPipe và mô
hình nhận diện đã nằm sẵn trong repo. Muốn máy chưa từng có mạng cũng chạy được thì
copy luôn cả thư mục `web/node_modules/` sang.

> **Đừng deploy chỉ để chơi ở sự kiện.** Nhận diện chạy hoàn toàn trong trình duyệt nên
> deploy không làm game mượt hơn, mà lại thêm một cái bẫy: trình duyệt **chỉ cho dùng
> camera trên HTTPS hoặc localhost**. Mở qua `http://192.168.x.x` trong mạng LAN là
> camera bị chặn thẳng, không hỏi quyền gì cả. Cứ chạy `npm run dev` ngay trên máy đặt
> tại booth.

## Kiểm thử

```bash
cd web
npm test
```

Lệnh này build rồi chạy `tests/rendered-html.test.mjs`. Bộ test kiểm tra cả những điều
dễ vỡ mà mắt thường khó bắt: mọi sprite được nạp đều tồn tại thật, ống luôn phủ kín tới
trần và tới mặt đất, và nền ngang phải ghép bằng lát gương để không lộ đường nối.

## Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| `web/public/game/` | Toàn bộ game: `index.html`, `app.js`, `styles.css` — HTML/JS thuần |
| `web/public/game/assets/` | Sprite và âm thanh gốc Flappy Bird |
| `web/app/`, `web/worker/` | Vỏ vinext + Cloudflare Worker, chỉ lo redirect và phục vụ file tĩnh |
| `web/tests/` | Kiểm thử |

## Ghi nhận

Sprite và âm thanh lấy từ [samuelcust/flappy-bird-assets](https://github.com/samuelcust/flappy-bird-assets),
xem `web/public/game/assets/LICENSE`. Bản quyền tác phẩm gốc thuộc về Dong Nguyen —
dự án này chỉ dùng cho hoạt động CLB phi lợi nhuận.
