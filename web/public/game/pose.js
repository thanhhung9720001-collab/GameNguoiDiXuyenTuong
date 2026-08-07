/**
 * Camera + MediaPipe Pose. Chỉ lo phần vào/ra: mở webcam, chạy mô hình, vẽ khung
 * xương lên khung nhỏ. Phần quyết định "đây có phải một nhịp quạt tay không" nằm ở
 * `flap-detector.js` để kiểm thử được ngoài trình duyệt.
 */

/*
 * MediaPipe được nhúng thẳng vào repo (xem `vendor/`), KHÔNG tải từ CDN.
 * Lý do: game chạy ở booth sự kiện, wifi hội trường hay chập chờn. Tải từ CDN thì mất
 * mạng là không bật được camera — cả booth đứng hình. Nhúng vào rồi thì rút phích
 * mạng vẫn chơi được.
 */
import { FilesetResolver, PoseLandmarker } from "./vendor/mediapipe/vision_bundle.mjs";
import { checkFraming, createFlapDetector, MIN_VISIBILITY } from "./flap-detector.js";

// Suy ra từ vị trí của chính file này, không phải từ URL của trang: hai đường dẫn dưới
// được thư viện tự fetch, mà đường dẫn tương đối sẽ tính theo trang chứ không theo module.
const WASM_PATH = new URL("./vendor/mediapipe/wasm", import.meta.url).href;
const MODEL_PATH = new URL("./vendor/models/pose_landmarker_lite.task", import.meta.url).href;

const SKELETON = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28],
];

/**
 * Bật camera và bộ nhận diện. Trả về đối tượng có `stop()` và `recalibrate()`.
 * `onFlap` gọi mỗi lần phát hiện một nhịp quạt tay.
 * `onFrame({ visible, armsVisible, lift, framing })`
 * gọi mỗi khung hình để bên ngoài tự quyết định hiển thị gì.
 */
export async function startPose({ video, overlay, onFlap, onFrame, onStatus }) {
  onStatus("ĐANG BẬT CAMERA…", "busy");

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false,
  });

  let landmarker = null;
  try {
    video.srcObject = stream;
    await video.play();

    onStatus("ĐANG TẢI MÔ HÌNH…", "busy");
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
    landmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_PATH, delegate: "GPU" },
      runningMode: "VIDEO",
      numPoses: 1,
    });
  } catch (error) {
    // Tải mô hình hỏng (mất mạng chẳng hạn) vẫn phải trả camera lại cho người dùng,
    // nếu không đèn webcam sáng mãi mà chẳng có gì chạy.
    stream.getTracks().forEach((track) => track.stop());
    video.srcObject = null;
    throw error;
  }

  const detector = createFlapDetector();
  const ctx = overlay.getContext("2d");
  let frame = 0;
  let lastVideoTime = -1;
  let stopped = false;

  function drawSkeleton(landmarks) {
    if (overlay.width !== overlay.clientWidth) overlay.width = overlay.clientWidth;
    if (overlay.height !== overlay.clientHeight) overlay.height = overlay.clientHeight;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (!landmarks) return;

    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    ctx.strokeStyle = "rgba(248, 225, 108, .95)";
    for (const [a, b] of SKELETON) {
      const from = landmarks[a], to = landmarks[b];
      if (!from || !to || from.visibility < MIN_VISIBILITY || to.visibility < MIN_VISIBILITY) continue;
      ctx.beginPath();
      ctx.moveTo(from.x * overlay.width, from.y * overlay.height);
      ctx.lineTo(to.x * overlay.width, to.y * overlay.height);
      ctx.stroke();
    }
  }

  function tick() {
    if (stopped) return;
    frame = requestAnimationFrame(tick);
    if (video.currentTime === lastVideoTime) return;
    lastVideoTime = video.currentTime;

    const now = performance.now();
    const landmarks = landmarker.detectForVideo(video, now).landmarks?.[0] || null;
    drawSkeleton(landmarks);

    const status = detector.update(landmarks, now);
    onFrame({ ...status, framing: checkFraming(landmarks), now });
    if (status.flapped) onFlap();
  }

  tick();

  return {
    stop() {
      stopped = true;
      cancelAnimationFrame(frame);
      stream.getTracks().forEach((track) => track.stop());
      landmarker.close();
      video.srcObject = null;
      ctx.clearRect(0, 0, overlay.width, overlay.height);
    },
    recalibrate: () => detector.reset(),
  };
}
