# Thư viện nhúng sẵn

Những file ở đây **không phải mã của dự án**. Chúng được tải về và cam kết vào repo để
game chạy được **hoàn toàn offline**.

Lý do: game chạy ở booth sự kiện, wifi hội trường hay chập chờn. Nếu tải MediaPipe từ
CDN thì mất mạng là không bật được camera — cả booth đứng hình. Nhúng sẵn thì rút phích
mạng vẫn chơi được.

## Nội dung

| Đường dẫn | Nguồn | Dung lượng |
|---|---|---|
| `mediapipe/vision_bundle.mjs` | `@mediapipe/tasks-vision@1.0.1` | 0.15 MB |
| `mediapipe/wasm/vision_wasm_internal.{js,wasm}` | cùng gói trên | 11.5 MB |
| `mediapipe/wasm/vision_wasm_nosimd_internal.{js,wasm}` | cùng gói trên | 10.8 MB |
| `models/pose_landmarker_lite.task` | [MediaPipe model zoo](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task) | 5.5 MB |

Bản `nosimd` là dự phòng cho máy không hỗ trợ WebAssembly SIMD. Thư viện tự chọn bản
phù hợp lúc chạy; **thiếu bản này thì máy đời cũ sẽ hỏng mà không báo gì rõ ràng**.

## Cập nhật

```bash
cd web/public/game
B="https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1"
curl -fL "$B/vision_bundle.mjs" -o vendor/mediapipe/vision_bundle.mjs
for f in vision_wasm_internal.js vision_wasm_internal.wasm \
         vision_wasm_nosimd_internal.js vision_wasm_nosimd_internal.wasm; do
  curl -fL "$B/wasm/$f" -o "vendor/mediapipe/wasm/$f"
done
curl -fL "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task" \
  -o vendor/models/pose_landmarker_lite.task
```

Đổi phiên bản thì nhớ sửa cả bảng ở trên. `npm test` có kiểm tra các file này tồn tại,
đủ lớn và đúng định dạng nhị phân — tải hụt sẽ bị bắt.

## Lưu ý: MediaPipe có gửi thống kê sử dụng

Trong `vision_bundle.mjs` có một bộ ghi log tự gửi `POST` tới
`https://odml.pa.googleapis.com/v1/log` mỗi 60 giây. Nó được khởi tạo vô điều kiện khi
tạo tác vụ nhận diện, không có tuỳ chọn tắt.

- Nội dung gửi đi là **thống kê sử dụng dạng protobuf, không phải hình ảnh camera**.
- Khi offline, lệnh gửi thất bại ngay và bộ ghi log **tự tắt hẳn** — không ảnh hưởng game.

Chưa chặn vì mọi cách chặn đều phải sửa mã thư viện hoặc thêm CSP, đều có rủi ro làm
hỏng phần nhận diện. Nếu cần chặn, xem lại phần này trước.
