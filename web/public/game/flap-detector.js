/**
 * Nhận diện động tác quạt hai tay như cánh chim.
 *
 * File này cố tình KHÔNG import gì cả — không MediaPipe, không DOM. Nhờ vậy nó chạy
 * và kiểm thử được ngoài trình duyệt, còn `pose.js` chỉ lo phần camera và mô hình.
 */

// Chỉ số điểm mốc của MediaPipe Pose.
export const NOSE = 0;
export const LEFT_SHOULDER = 11, RIGHT_SHOULDER = 12;
export const LEFT_WRIST = 15, RIGHT_WRIST = 16;
export const LEFT_HIP = 23, RIGHT_HIP = 24;
export const MIN_VISIBILITY = 0.5;

/**
 * Đo cổ tay so với vai, quy theo TỈ LỆ chiều dài thân người chứ không theo toạ độ
 * chuẩn hoá. Lý do: toạ độ MediaPipe là tỉ lệ so với khung hình, nên đứng xa camera
 * thì cùng một động tác lại cho số nhỏ hơn hẳn. Chia cho chiều dài thân (vai→hông)
 * thì người cao, người thấp, đứng gần hay đứng xa đều ra cùng một con số.
 *
 * Bắn ở nhịp HẠ tay — đó là lúc con chim thật đạp không khí, nên khớp cảm giác.
 * Phải nâng tay qua ARM_UP rồi hạ qua ARM_DOWN mới tính trọn một nhịp.
 */
export const ARM_UP = 0.30; // cổ tay cao hơn vai 30% chiều dài thân
export const ARM_DOWN = -0.10; // rồi hạ xuống dưới vai

export const FLAP_COOLDOWN = 180; // ms, chặn một nhịp bị đếm thành hai
export const MIN_TORSO = 0.05; // thân ngắn hơn mức này thì phép đo không đáng tin

const IDLE = { visible: false, armsVisible: false, flapped: false, lift: 0 };

export function createFlapDetector() {
  let armsUp = false;
  let lastFlapAt = 0;

  return {
    reset() {
      armsUp = false;
      lastFlapAt = 0;
    },

    /** Trả về `{ visible, armsVisible, flapped, lift }`; `flapped` đúng một khung hình. */
    update(landmarks, now) {
      if (!landmarks) return { ...IDLE };

      const core = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP].map((i) => landmarks[i]);
      if (core.some((point) => !point || point.visibility < MIN_VISIBILITY)) return { ...IDLE };

      const [leftShoulder, rightShoulder, leftHip, rightHip] = core;
      const hipY = (leftHip.y + rightHip.y) / 2;
      const shoulderY = (leftShoulder.y + rightShoulder.y) / 2;
      const torso = Math.abs(hipY - shoulderY);

      // Thân quá ngắn nghĩa là người đang nghiêng, khuất, hoặc đứng quá xa. Đo lúc này
      // không đáng tin, và vì torso nằm dưới mẫu số nên sai số bị khuếch đại.
      if (torso < MIN_TORSO) return { ...IDLE };

      const wrists = [landmarks[LEFT_WRIST], landmarks[RIGHT_WRIST]];
      if (wrists.some((point) => !point || point.visibility < MIN_VISIBILITY)) {
        // Mất dấu tay thì quên nhịp đang dở, đừng để nó bắn khi tay hiện lại.
        armsUp = false;
        return { visible: true, armsVisible: false, flapped: false, lift: 0 };
      }

      // y tăng dần xuống dưới, nên cổ tay cao hơn vai là hiệu số DƯƠNG.
      const lift = wrists.reduce((total, wrist) => total + (shoulderY - wrist.y), 0) / 2 / torso;

      let flapped = false;
      if (!armsUp && lift >= ARM_UP) {
        armsUp = true;
      } else if (armsUp && lift <= ARM_DOWN) {
        armsUp = false;
        if (now - lastFlapAt >= FLAP_COOLDOWN) {
          flapped = true;
          lastFlapAt = now;
        }
      }

      return { visible: true, armsVisible: true, flapped, lift };
    },
  };
}

/* ---------- Kiểm tra người chơi đã đứng đúng chỗ chưa ---------- */

const TORSO_TOO_FAR = 0.10; // thân ngắn hơn: đứng quá xa camera
const TORSO_TOO_CLOSE = 0.46; // thân dài hơn: đứng quá sát
const HEAD_MARGIN = 0.04; // chừa chỗ phía trên đầu để giơ tay lên không bị mất dấu
const FOOT_MARGIN = 0.02;
const CENTER_TOLERANCE = 0.28; // lệch khỏi giữa khung quá mức này thì chưa đạt

/**
 * Trả về `{ ok, reason, fit }`.
 *
 * `reason` là câu hướng dẫn cụ thể; khi sai nhiều thứ cùng lúc thì báo thứ SAI NHẤT,
 * không phải thứ đầu tiên trong danh sách. Báo theo thứ tự cố định khiến người chơi
 * sửa xong một lỗi mà chữ vẫn y nguyên — tưởng máy bị đơ.
 *
 * `fit` từ 0 đến 1 để giao diện hiện được mức "gần đạt", thay vì chỉ đỏ hoặc xanh.
 *
 * Đứng đúng chỗ quan trọng vì động tác quạt tay cần thấy rõ CẢ hai cổ tay lẫn hai
 * vai: đứng lệch hoặc quá sát là tay vung ra ngoài khung và nhịp quạt không được tính.
 */
export function checkFraming(landmarks) {
  if (!landmarks) return { ok: false, reason: "KHÔNG THẤY AI TRONG KHUNG HÌNH", fit: 0 };

  const needed = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP].map((i) => landmarks[i]);
  if (needed.some((point) => !point || point.visibility < MIN_VISIBILITY)) {
    return { ok: false, reason: "ĐỨNG THẲNG, QUAY MẶT VỀ CAMERA", fit: 0 };
  }

  const [nose, leftShoulder, rightShoulder, leftHip, rightHip] = needed;
  const hipY = (leftHip.y + rightHip.y) / 2;
  const shoulderY = (leftShoulder.y + rightShoulder.y) / 2;
  const torso = Math.abs(hipY - shoulderY);
  const centerX = (leftShoulder.x + rightShoulder.x + leftHip.x + rightHip.x) / 4;

  /*
   * Camera nhìn thẳng vào mặt người chơi, nên trục x của MediaPipe lộn so với hướng
   * người chơi cảm nhận: bước sang PHẢI của bản thân thì x GIẢM. Khung camera lại
   * được soi gương khi hiển thị, nên x nhỏ hiện ra ở mép phải màn hình. Kết quả:
   * x < 0.5 nghĩa là đang lệch sang phải, phải dịch sang trái — và câu này khớp luôn
   * với cái người chơi nhìn thấy trong gương.
   */
  const offCenter = centerX < 0.5 ? "DỊCH SANG TRÁI MỘT CHÚT" : "DỊCH SANG PHẢI MỘT CHÚT";

  // Mỗi lỗi quy về một con số "sai bao nhiêu phần", để so được với nhau.
  const problems = [
    { error: (TORSO_TOO_FAR - torso) / TORSO_TOO_FAR, reason: "TIẾN LẠI GẦN CAMERA HƠN" },
    { error: (torso - TORSO_TOO_CLOSE) / TORSO_TOO_CLOSE, reason: "LÙI RA XA CAMERA HƠN" },
    { error: (HEAD_MARGIN - nose.y) / HEAD_MARGIN, reason: "LÙI RA XA CAMERA HƠN" },
    { error: (hipY - (1 - FOOT_MARGIN)) / FOOT_MARGIN, reason: "LÙI RA XA CAMERA HƠN" },
    { error: (Math.abs(centerX - 0.5) - CENTER_TOLERANCE) / CENTER_TOLERANCE, reason: offCenter },
  ];

  const worst = problems.reduce((a, b) => (b.error > a.error ? b : a));
  if (worst.error > 0) {
    return { ok: false, reason: worst.reason, fit: Math.max(0, 1 - Math.min(1, worst.error)) };
  }
  return { ok: true, reason: "ĐỨNG YÊN NHÉ", fit: 1 };
}
