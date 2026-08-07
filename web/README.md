# web

Vỏ vinext + Vite + Cloudflare Worker phục vụ game. Xem [README ở thư mục gốc](../README.md)
để biết cách chạy và kiến trúc tổng thể.

Game nằm ở `public/game/` dưới dạng HTML/JS thuần, không đi qua React. Trang gốc `/`
chỉ làm mỗi việc redirect sang `/game/index.html` (xử lý trong `app/page.tsx`).

Lưu ý: đừng đặt `index.html` ở thư mục này. Vite dev server ưu tiên phục vụ nó cho `/`
và sẽ chặn mất redirect, khiến `npm run dev` hiện ra thứ khác với bản build.
