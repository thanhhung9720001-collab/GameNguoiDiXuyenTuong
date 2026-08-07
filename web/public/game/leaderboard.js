/**
 * Lớp lưu trữ bảng xếp hạng.
 *
 * Hiện lưu trong localStorage của chính máy đang chạy game — không cần mạng, hợp với
 * booth sự kiện dùng một laptop. Muốn đổi sang Cloudflare D1 sau này thì chỉ cần thay
 * phần thân của 4 hàm dưới bằng lệnh `fetch` tới API, mọi chỗ gọi giữ nguyên.
 *
 * Vì vậy toàn bộ hàm ở đây đều `async` dù localStorage vốn đồng bộ: D1 bắt buộc bất
 * đồng bộ, khai báo async ngay từ đầu để lúc chuyển không phải sửa lại chỗ gọi.
 */

const STORAGE_KEY = "flappy-bird-leaderboard";

/** Chỉ giữ 20 người dẫn đầu; ai rớt khỏi top 20 là bị loại khỏi bộ nhớ luôn. */
export const MAX_ENTRIES = 20;

export const MAX_NAME_LENGTH = 12;

/** Ký tự vô hình / ép hướng văn bản — dán vào tên là bố cục bảng vỡ ngay. */
const INVISIBLE = new Set([
  0x200b, 0x200c, 0x200d, 0x200e, 0x200f, // zero-width, đánh dấu hướng
  0x2028, 0x2029, // ngắt dòng, ngắt đoạn
  0x202a, 0x202b, 0x202c, 0x202d, 0x202e, // ép hướng hiển thị
  0xfeff, // zero-width no-break space
]);

/**
 * Làm sạch tên người chơi. Bảng xếp hạng chiếu công khai lên màn lớn nên tên phải
 * ngắn, gọn trong một dòng và không chứa ký tự phá bố cục.
 *
 * Lọc theo mã ký tự thay vì regex: dải ký tự điều khiển viết bằng escape rất dễ hỏng
 * khi file đi qua công cụ khác, mà hỏng thì âm thầm, không ai phát hiện.
 */
export function sanitizeName(raw) {
  const kept = [];
  for (const character of String(raw ?? "")) {
    const code = character.codePointAt(0);
    // Tab, xuống dòng, về đầu dòng là dấu ngăn cách — đổi thành dấu cách, không xoá,
    // nếu không "Nam<xuống dòng>Anh" sẽ dính thành "NamAnh".
    if (code === 0x09 || (code >= 0x0a && code <= 0x0d)) {
      kept.push(" ");
      continue;
    }
    if (code < 0x20 || code === 0x7f) continue; // ký tự điều khiển còn lại: rác, xoá hẳn
    if (INVISIBLE.has(code)) continue;
    kept.push(character);
  }
  return kept.join("").replace(/\s+/g, " ").trim().slice(0, MAX_NAME_LENGTH);
}

function readAll() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    if (!Array.isArray(parsed)) return [];
    // Lọc bỏ bản ghi hỏng để một dòng lỗi không làm sập cả bảng.
    return parsed.filter((entry) => entry && typeof entry.name === "string" && Number.isFinite(entry.score));
  } catch {
    return [];
  }
}

/** Điểm cao đứng trước; cùng điểm thì ai đạt trước đứng trên. */
function compare(a, b) {
  return b.score - a.score || a.at - b.at;
}

function writeAll(entries) {
  try {
    // Sắp xếp trước khi cắt, nếu không thì cắt nhầm người: chỉ `submit` mới sắp xếp
    // sẵn, còn `remove` truyền vào mảng theo thứ tự đang lưu.
    const kept = [...entries].sort(compare).slice(0, MAX_ENTRIES);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(kept));
  } catch (error) {
    // Hết dung lượng hoặc chế độ ẩn danh — không được để game dừng vì chuyện này.
    console.warn("Không lưu được bảng xếp hạng:", error);
  }
}

/** Lấy `limit` người dẫn đầu, đã kèm sẵn thứ hạng. */
export async function top(limit = 10) {
  return readAll()
    .sort(compare)
    .slice(0, limit)
    .map((entry, index) => ({ ...entry, rank: index + 1 }));
}

/**
 * Ghi một lượt chơi — MỘT NGƯỜI MỘT DÒNG, giữ điểm cao nhất.
 *
 * Ghép theo tên đã hạ chữ thường, nên "Hùng" và "hùng" được coi là cùng một người.
 * Hệ quả cần biết: mọi lượt để trống tên đều dồn chung vào một dòng "Ẩn danh".
 *
 * Trả về thứ hạng và `isBest` — kỷ lục của CHÍNH người này, không phải của cả máy.
 * Ở booth nhiều người chơi chung một máy, so với kỷ lục chung là sai: người sau sẽ
 * không bao giờ được ăn mừng dù vừa phá kỷ lục của bản thân.
 */
export async function submit(name, score) {
  const cleanName = sanitizeName(name) || "Ẩn danh";
  const points = Math.max(0, Math.floor(score));
  const entries = readAll();

  const key = cleanName.toLowerCase();
  const existing = entries.find((entry) => entry.name.toLowerCase() === key);
  const previousBest = existing ? existing.score : null;
  const isBest = previousBest === null || points > previousBest;

  let target = existing;
  if (!target) {
    target = { id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`, name: cleanName, score: points, at: Date.now() };
    entries.push(target);
  } else if (isBest) {
    // Chỉ nâng điểm, không bao giờ hạ. `at` cập nhật theo lúc đạt được điểm đó, để
    // luật hoà điểm "ai đạt trước đứng trên" vẫn đúng.
    target.score = points;
    target.at = Date.now();
    target.name = cleanName; // giữ đúng cách viết hoa lần gần nhất
  }

  entries.sort(compare);
  writeAll(entries);

  return {
    ...target,
    rank: entries.findIndex((entry) => entry.id === target.id) + 1,
    isBest,
    previousBest,
  };
}

/** Xoá một dòng — dành cho người trực booth khi có bạn gõ tên bậy. */
export async function remove(id) {
  writeAll(readAll().filter((entry) => entry.id !== id));
}

export async function clear() {
  writeAll([]);
}
