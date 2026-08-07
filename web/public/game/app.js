/**
 * Flappy Bird — điều khiển bằng động tác quạt hai tay qua webcam.
 *
 * Luồng một lượt chơi:
 *   nhập tên → (tự bật camera) → canh vị trí + đếm ngược 5s → chơi → bảng xếp hạng
 *
 * Không có webcam thì bỏ qua bước canh vị trí và chơi bằng Space/chuột. Mọi cách
 * điều khiển đều đi qua handleInput().
 */

import * as leaderboard from "./leaderboard.js";
import { startPose } from "./pose.js";

// Màn hình ngang 16:9. Chiều cao bằng đúng chiều cao sprite nền gốc (512) nên mọi
// hình được vẽ ở kích thước pixel gốc, không co méo; chỉ bề ngang là mở rộng.
const GAME_WIDTH = 910;
const GAME_HEIGHT = 512;
const BASE_HEIGHT = 112;
const GROUND_Y = GAME_HEIGHT - BASE_HEIGHT;
const BACKGROUND_PARALLAX = 0.35; // nền trôi chậm hơn cho có chiều sâu

/**
 * Số đo lấy trực tiếp từ từng file nền: mỗi ảnh chỉ có một dải hoạ tiết, phía trên và
 * phía dưới là hai vùng màu phẳng. Nhờ vậy nền rộng được dựng lại từ chính art gốc mà
 * không cần thêm file ảnh nào: tô phẳng hai vùng kia, chỉ lát riêng dải hoạ tiết.
 *
 * Hai ảnh có số đo KHÁC nhau — ảnh đêm còn có vùng sao bắt đầu từ y=122, nên dải hoạ
 * tiết của nó cao hơn hẳn. Dùng chung số đo sẽ làm sai màu trời và xoá mất sao.
 */
const THEMES = {
  day: { sprite: "background-day", sky: "#4ec0ca", ground: "#5ee270", bandTop: 304, bandBottom: 390 },
  night: { sprite: "background-night", sky: "#008793", ground: "#00a300", bandTop: 122, bandBottom: 386 },
};
const NIGHT_EVERY = 10; // cứ 10 điểm lại đổi ngày ↔ đêm
const RED_PIPES_FROM = 20; // qua mốc này ống chuyển đỏ — chỉ để ghi nhận, độ khó không đổi
const BIRD_COLORS = ["yellow", "red", "blue"];

// Bước canh vị trí trước khi chơi.
const STEADY_REQUIRED = 1000; // phải đứng yên đúng chỗ chừng này rồi mới bắt đầu đếm
const COUNTDOWN_MS = 5000;

const MAX_FALL_SPEED = 12;
const BIRD_X = 210; // lùi về trái để người chơi có thời gian phản ứng trên màn rộng
const COLLISION_INSET = 3; // nới va chạm một chút cho đỡ ức chế

/**
 * Hai bộ thông số vật lý, đổi theo cách điều khiển.
 *
 * Quạt tay chậm hơn bấm phím, nên chế độ webcam phải rơi chậm hơn, khe ống rộng hơn
 * và cuộn chậm hơn để một nhịp quạt nuôi chim bay đủ lâu tới ống kế tiếp.
 */
const PHYSICS = {
  keyboard: { gravity: 0.45, flap: -8.4, scroll: 2.0, gap: 130, spacing: 260 },
  webcam: { gravity: 0.22, flap: -7.0, scroll: 1.45, gap: 175, spacing: 340 },
};

/*
 * Cố ý KHÔNG có hệ thống độ khó tăng dần. Game điều khiển bằng cả cơ thể, nên càng
 * chơi càng mệt — đó đã là đường cong độ khó tự nhiên, thật hơn bất cứ thứ gì bịa ra
 * bằng cách bóp hẹp khe ống theo điểm số.
 */
let tuning = PHYSICS.keyboard;

const SPRITE_NAMES = [
  "background-day", "background-night", "base", "pipe-green", "pipe-red",
  "yellowbird-upflap", "yellowbird-midflap", "yellowbird-downflap",
  "redbird-upflap", "redbird-midflap", "redbird-downflap",
  "bluebird-upflap", "bluebird-midflap", "bluebird-downflap",
  "message", "gameover",
  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
];
const BIRD_FLAPS = ["upflap", "midflap", "downflap"];
const SOUND_NAMES = ["wing", "point", "hit", "die", "swoosh"];

const LAST_NAME_KEY = "flappy-bird-last-name";
// Hiện đúng bằng số bản ghi được giữ lại, để "lưu bao nhiêu thì thấy bấy nhiêu".
const BOARD_SIZE = leaderboard.MAX_ENTRIES;

const $ = (selector) => document.querySelector(selector);
const stage = $("#stage");
const canvas = $("#game-canvas");
const ctx = canvas.getContext("2d");
const hint = $("#hint");
const menu = $("#menu-screen");
const boardList = $("#board-list");
const boardEmpty = $("#board-empty");
const nameInput = $("#player-name");
const joinForm = $("#join-form");
const resultPanel = $("#result-panel");
const cameraPip = $("#camera-pip");
const cameraBadge = $("#camera-badge");
const cameraButton = $("#camera-button");
const cameraCountdown = $("#camera-countdown");

const sprites = {};
const sounds = {};

const state = {
  phase: "loading", // loading | menu | ready | playing | dying | gameover
  playerName: "",
  lastEntryId: null, // để tô sáng đúng dòng của lượt vừa chơi
  lastRank: null,
  bird: { y: 0, velocity: 0, frame: 0, rotation: 0 },
  birdColor: "yellow",
  theme: "day",
  pipeColor: "green",
  pipes: [],
  score: 0,
  pose: null, // phiên webcam đang chạy, null nghĩa là đang dùng bàn phím/chuột
  steadyMs: 0, // thời gian đã đứng đúng chỗ, dùng ở bước canh vị trí
  countdownMs: 0, // thời gian đã đếm ngược; tạm dừng chứ không reset khi bước ra
  lastFramingAt: 0,
  cameraDeclined: false, // đã từ chối quyền thì thôi, đừng hỏi lại mỗi lượt
  scrollOffset: 0,
  frameCount: 0,
  animation: 0,
  lastFrame: 0,
};

ctx.imageSmoothingEnabled = false;

/** Đang chơi thì giấu nút camera — không ai muốn lỡ tay tắt camera giữa lượt. */
const HIDE_CAMERA_BUTTON_IN = new Set(["playing", "dying"]);

/**
 * Chỗ DUY NHẤT được đổi `state.phase`, để mọi thứ phụ thuộc vào trạng thái luôn đồng
 * bộ. Gán tay ở từng chỗ thì thêm một trạng thái mới là chắc chắn sót một chỗ.
 */
function setPhase(next) {
  state.phase = next;
  cameraButton.hidden = HIDE_CAMERA_BUTTON_IN.has(next);
}

/* ---------- Nạp tài nguyên ---------- */

function loadSprite(name) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve([name, image]);
    image.onerror = () => reject(new Error(`Không nạp được sprite: ${name}.png`));
    image.src = `./assets/sprites/${name}.png`;
  });
}

async function loadSprites() {
  const loaded = await Promise.all(SPRITE_NAMES.map(loadSprite));
  for (const [name, image] of loaded) sprites[name] = image;
}

/** Âm thanh không chặn quá trình khởi động: thiếu file thì game vẫn chạy, chỉ im tiếng. */
function loadSounds() {
  for (const name of SOUND_NAMES) {
    const audio = new Audio(`./assets/audio/${name}.ogg`);
    audio.preload = "auto";
    audio.volume = 0.4;
    sounds[name] = audio;
  }
}

function playSound(name) {
  const audio = sounds[name];
  if (!audio) return;
  audio.currentTime = 0;
  // Trình duyệt chặn phát trước khi người dùng tương tác — bỏ qua cho êm.
  audio.play().catch(() => {});
}

/* ---------- Vẽ nền ---------- */

/**
 * Vẽ một bản sao dải hoạ tiết, có thể lật gương theo trục dọc.
 * Ảnh gốc không nối liền ở mép (58/512 hàng lệch màu), nên lát gương xen kẽ là cách
 * duy nhất ghép được bằng chính art gốc mà không lộ đường nối: mép trái của bản lật
 * luôn trùng khít mép phải của bản đứng cạnh.
 */
function drawBandTile(sprite, theme, x, flipped) {
  const height = theme.bandBottom - theme.bandTop;
  ctx.save();
  if (flipped) {
    ctx.translate(x + sprite.width, 0);
    ctx.scale(-1, 1);
  } else {
    ctx.translate(x, 0);
  }
  ctx.drawImage(sprite, 0, theme.bandTop, sprite.width, height, 0, theme.bandTop, sprite.width, height);
  ctx.restore();
}

function drawBackground() {
  const theme = THEMES[state.theme];
  const sprite = sprites[theme.sprite];
  const offset = Math.floor(state.scrollOffset * BACKGROUND_PARALLAX);

  ctx.fillStyle = theme.sky;
  ctx.fillRect(0, 0, GAME_WIDTH, theme.bandTop);
  ctx.fillStyle = theme.ground;
  ctx.fillRect(0, theme.bandBottom, GAME_WIDTH, GROUND_Y - theme.bandBottom);

  // Chu kỳ là 2 bản (đứng + lật), nên lấy dư theo 2×chiều rộng sprite.
  const period = sprite.width * 2;
  const start = -(((offset % period) + period) % period);
  for (let x = start, index = 0; x < GAME_WIDTH; x += sprite.width, index += 1) {
    drawBandTile(sprite, theme, Math.round(x), index % 2 === 1);
  }
}

function drawBase() {
  const base = sprites.base;
  // scrollOffset ngừng tăng khi chim chết, nên mặt đất tự đứng yên như bản gốc.
  const offset = state.scrollOffset;
  const start = -(((offset % base.width) + base.width) % base.width);
  for (let x = start; x < GAME_WIDTH; x += base.width) {
    ctx.drawImage(base, Math.round(x), GROUND_Y);
  }
}

/* ---------- Ống nước ---------- */

/** Khoảng đặt tâm khe hở, tính sao cho ống luôn phủ kín tới mép trên và mặt đất. */
function gapCenterRange() {
  const pipeHeight = pipeSprite().height;
  return {
    min: Math.max(tuning.gap / 2 + 40, GROUND_Y - pipeHeight + tuning.gap / 2),
    max: Math.min(GROUND_Y - tuning.gap / 2 - 40, pipeHeight - tuning.gap / 2),
  };
}

function spawnPipe(x) {
  const { min, max } = gapCenterRange();
  state.pipes.push({ x, gapCenter: min + Math.random() * (max - min), scored: false });
}

function resetPipes() {
  state.pipes = [];
  // Ống đầu tiên đặt ngoài mép phải để người chơi kịp vào nhịp.
  for (let i = 0; i < 5; i++) spawnPipe(GAME_WIDTH + 120 + i * tuning.spacing);
}

function drawPipes() {
  const pipe = pipeSprite();
  for (const item of state.pipes) {
    const x = Math.round(item.x);
    const gapTop = Math.round(item.gapCenter - tuning.gap / 2);
    const gapBottom = Math.round(item.gapCenter + tuning.gap / 2);

    // Ống trên: lật dọc để miệng ống quay xuống.
    ctx.save();
    ctx.translate(x, gapTop);
    ctx.scale(1, -1);
    ctx.drawImage(pipe, 0, 0);
    ctx.restore();

    ctx.drawImage(pipe, x, gapBottom);
  }
}

/* ---------- Chim ---------- */

function resetBird() {
  state.bird.y = GROUND_Y / 2 - 40;
  state.bird.velocity = 0;
  state.bird.rotation = 0;
  state.bird.frame = 0;
}

/** Ống và chim đổi màu theo lượt/độ khó; mọi biến thể cùng kích thước nên vật lý không đổi. */
function pipeSprite() {
  return sprites[`pipe-${state.pipeColor}`];
}

function birdSprite() {
  return sprites[`${state.birdColor}bird-${BIRD_FLAPS[state.bird.frame % BIRD_FLAPS.length]}`];
}

function birdBox() {
  const sprite = birdSprite();
  return {
    left: BIRD_X + COLLISION_INSET,
    right: BIRD_X + sprite.width - COLLISION_INSET,
    top: state.bird.y + COLLISION_INSET,
    bottom: state.bird.y + sprite.height - COLLISION_INSET,
  };
}

function drawBird() {
  const sprite = birdSprite();
  // Ở màn chờ, chim nhấp nhô nhẹ cho sinh động.
  const bob = state.phase === "ready" ? Math.sin(state.frameCount / 9) * 5 : 0;
  ctx.save();
  ctx.translate(BIRD_X + sprite.width / 2, state.bird.y + bob + sprite.height / 2);
  ctx.rotate(state.bird.rotation);
  ctx.drawImage(sprite, -sprite.width / 2, -sprite.height / 2);
  ctx.restore();
}

/* ---------- Điểm số ---------- */

function measureDigits(value) {
  return String(value).split("").reduce((total, digit) => total + sprites[digit].width, 0);
}

function drawDigits(value, centerX, y, scale = 1) {
  const digits = String(value).split("");
  let x = centerX - (measureDigits(value) * scale) / 2;
  for (const digit of digits) {
    const sprite = sprites[digit];
    ctx.drawImage(sprite, Math.round(x), y, sprite.width * scale, sprite.height * scale);
    x += sprite.width * scale;
  }
}

function drawCentered(sprite, y) {
  ctx.drawImage(sprite, Math.round((GAME_WIDTH - sprite.width) / 2), y);
}


/* ---------- Bảng xếp hạng ---------- */

/**
 * Dựng lại danh sách. Tên do người chơi nhập nên dựng bằng textContent, tuyệt đối
 * không nối chuỗi vào innerHTML.
 */
async function renderBoard() {
  const entries = await leaderboard.top(BOARD_SIZE);
  boardList.textContent = "";
  boardList.hidden = entries.length === 0;
  boardEmpty.hidden = entries.length > 0;

  for (const entry of entries) {
    const row = document.createElement("li");
    row.className = "board-row";
    if (entry.id === state.lastEntryId) row.classList.add("is-you");

    row.dataset.rank = String(entry.rank); // top 3 được tô màu huy chương qua CSS

    const rank = document.createElement("span");
    rank.className = "board-rank";
    rank.textContent = String(entry.rank);

    const name = document.createElement("span");
    name.className = "board-name";
    name.textContent = entry.name;
    name.title = entry.name;

    const score = document.createElement("span");
    score.className = "board-score";
    score.textContent = String(entry.score);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "board-remove";
    remove.textContent = "×";
    remove.title = `Xoá "${entry.name}"`;
    remove.setAttribute("aria-label", `Xoá ${entry.name}`);
    remove.addEventListener("click", async () => {
      await leaderboard.remove(entry.id);
      await renderBoard();
    });

    row.append(rank, name, score, remove);
    boardList.append(row);
  }
}

/** Mở khung ở chế độ nhập tên. */
async function openMenu() {
  setPhase("menu");
  menu.hidden = false;
  joinForm.hidden = false;
  resultPanel.hidden = true;
  hint.textContent = "";
  await renderBoard();
  nameInput.value = state.playerName || localStorage.getItem(LAST_NAME_KEY) || "";
  nameInput.focus();
  nameInput.select();
}

/** Mở đúng khung đó ở chế độ kết quả, ngay khi vừa thua. */
async function openResult(entry) {
  menu.hidden = false;
  joinForm.hidden = true;
  resultPanel.hidden = false;

  $("#result-banner").src = sprites.gameover.src;
  $("#result-name").textContent = state.playerName;
  $("#result-score").textContent = String(state.score);
  $("#result-rank").textContent = `#${entry.rank}`;

  // `isBest` là kỷ lục của chính người này, do bảng xếp hạng tính.
  let note = "";
  if (entry.isBest && entry.rank === 1) note = "🏆 DẪN ĐẦU BẢNG!";
  else if (entry.isBest && entry.previousBest !== null) note = `KỶ LỤC MỚI CỦA BẠN! (cũ: ${entry.previousBest})`;
  else if (!entry.isBest) note = `Điểm cao nhất của bạn vẫn là ${entry.previousBest}`;
  $("#result-note").textContent = note;

  await renderBoard();
  // Space bấm luôn nút này, khỏi phải rời tay khỏi bàn phím.
  $("#replay-button").focus();
}

function closeMenu() {
  menu.hidden = true;
  nameInput.blur();
}

/* ---------- Vòng đời ---------- */

/* ---------- Bước canh vị trí trước khi chơi ---------- */

/** Mở camera to, yêu cầu người chơi đứng vào khung, rồi đếm ngược 5 giây. */
function toFraming() {
  setPhase("framing");
  state.steadyMs = 0;
  state.countdownMs = 0;
  closeMenu();
  cameraPip.classList.add("framing");
  cameraCountdown.textContent = "";
  hint.textContent = "";
  // Mốc đứng yên phải đo lại từ đầu cho đúng người và đúng chỗ vừa đứng.
  state.pose?.recalibrate();
}

/**
 * Gọi mỗi khung hình camera trong lúc canh vị trí.
 * Bước ra ngoài thì đồng hồ TẠM DỪNG chứ không reset — bị đẩy về 5 giây mỗi lần
 * nhúc nhích sẽ khiến người chơi không bao giờ vào được game.
 */
function updateFraming({ framing, now }) {
  const delta = state.lastFramingAt ? Math.min(now - state.lastFramingAt, 100) : 0;
  state.lastFramingAt = now;

  cameraPip.classList.toggle("aligned", framing.ok);
  // Sắp đạt thì khung chuyển vàng — người chơi thấy mình đang ấm dần chứ không phải
  // chỉ có đỏ với xanh, nên biết mình đang đi đúng hướng.
  cameraPip.classList.toggle("near", !framing.ok && framing.fit >= 0.55);

  if (!framing.ok) {
    state.steadyMs = 0;
    setCameraStatus(framing.reason, "warn");
    cameraCountdown.textContent = state.countdownMs > 0 ? "…" : "";
    return;
  }

  state.steadyMs += delta;
  if (state.steadyMs < STEADY_REQUIRED) {
    setCameraStatus("ĐỨNG YÊN NHÉ…", "busy");
    cameraCountdown.textContent = "";
    return;
  }

  state.countdownMs += delta;
  const remaining = COUNTDOWN_MS - state.countdownMs;
  if (remaining <= 0) {
    cameraCountdown.textContent = "";
    cameraPip.classList.remove("framing", "aligned", "near");
    setCameraStatus("QUẠT TAY ĐI!", "ready");
    toReady();
    return;
  }

  setCameraStatus("SẴN SÀNG!", "ready");
  cameraCountdown.textContent = String(Math.ceil(remaining / 1000));
}

/** Đồng bộ chủ đề nền và màu ống theo điểm hiện tại. */
function applyProgression() {
  state.theme = Math.floor(state.score / NIGHT_EVERY) % 2 === 1 ? "night" : "day";
  state.pipeColor = state.score >= RED_PIPES_FROM ? "red" : "green";
}

function toReady() {
  setPhase("ready");
  state.score = 0;
  closeMenu();
  // Mỗi lượt một màu chim cho đỡ nhàm — chỉ là thẩm mỹ, không đổi lối chơi.
  state.birdColor = BIRD_COLORS[Math.floor(Math.random() * BIRD_COLORS.length)];
  applyProgression();
  resetBird();
  resetPipes();
  hint.textContent = "Nhấn Space / chạm màn hình để vỗ cánh";
}

function startPlaying() {
  setPhase("playing");
  hint.textContent = "";
  flap();
}

function flap() {
  state.bird.velocity = tuning.flap;
  playSound("wing");
}

function die() {
  setPhase("dying");
  playSound("hit");
  if (state.bird.y < GROUND_Y) playSound("die");
}

async function toGameOver() {
  setPhase("gameover");
  playSound("swoosh");
  hint.textContent = "";

  // Kỷ lục do bảng xếp hạng tự tính, theo từng người. Không giữ thêm một "kỷ lục của
  // cả máy" song song nữa: hai nguồn sự thật thì kiểu gì cũng lệch nhau.
  const entry = await leaderboard.submit(state.playerName, state.score);
  state.lastEntryId = entry.id;
  state.lastRank = entry.rank;
  await openResult(entry);
}

/** Một hành động điều khiển duy nhất — sau này webcam cũng gọi đúng hàm này. */
function handleInput() {
  if (state.phase === "ready") startPlaying();
  else if (state.phase === "playing") flap();
  // Chơi lại giữ nguyên tên: ở booth đông người, bắt gõ lại tên mỗi lượt là tắc hàng.
  else if (state.phase === "gameover") toReady();
}

/* ---------- Cập nhật mỗi khung hình ---------- */

function updateBird(delta) {
  const bird = state.bird;
  bird.velocity = Math.min(bird.velocity + tuning.gravity * delta, MAX_FALL_SPEED);
  bird.y += bird.velocity * delta;

  // Ngóc lên khi bay lên, chúi dần xuống khi rơi.
  const target = bird.velocity < 0 ? -0.35 : Math.min(Math.PI / 2, bird.velocity / 14);
  bird.rotation += (target - bird.rotation) * 0.15 * delta;

  if (state.phase === "playing" && state.frameCount % 6 === 0) bird.frame += 1;
}

function updatePipes(delta) {
  for (const pipe of state.pipes) pipe.x -= tuning.scroll * delta;

  // Ống ra khỏi màn thì đẩy về cuối hàng, khỏi phải cấp phát mới. `last` phải cập
  // nhật ngay trong vòng lặp, nếu không hai ống tái sử dụng cùng frame sẽ chồng nhau.
  let last = state.pipes.reduce((max, pipe) => Math.max(max, pipe.x), -Infinity);
  const { min, max } = gapCenterRange();
  for (const pipe of state.pipes) {
    if (pipe.x + pipeSprite().width < 0) {
      last += tuning.spacing;
      pipe.x = last;
      pipe.gapCenter = min + Math.random() * (max - min);
      pipe.scored = false;
    }
  }
}

function updateScore() {
  const box = birdBox();
  for (const pipe of state.pipes) {
    if (!pipe.scored && box.left > pipe.x + pipeSprite().width) {
      pipe.scored = true;
      state.score += 1;
      playSound("point");
      applyProgression();
    }
  }
}

function hitsAnything() {
  const box = birdBox();
  if (box.bottom >= GROUND_Y) return true;
  if (box.top <= 0) return true;

  const width = pipeSprite().width;
  for (const pipe of state.pipes) {
    const overlapsX = box.right > pipe.x && box.left < pipe.x + width;
    if (!overlapsX) continue;
    const gapTop = pipe.gapCenter - tuning.gap / 2;
    const gapBottom = pipe.gapCenter + tuning.gap / 2;
    if (box.top < gapTop || box.bottom > gapBottom) return true;
  }
  return false;
}

function update(delta) {
  state.frameCount += 1;

  // Màn chờ và màn menu cùng chạy hoạt cảnh nền cho sinh động khi không ai chơi.
  if (state.phase === "menu" || state.phase === "framing" || state.phase === "ready") {
    state.scrollOffset += tuning.scroll * delta;
    if (state.frameCount % 8 === 0) state.bird.frame += 1;
    return;
  }

  if (state.phase === "playing") {
    state.scrollOffset += tuning.scroll * delta;
    updateBird(delta);
    updatePipes(delta);
    updateScore();
    if (hitsAnything()) die();
    return;
  }

  if (state.phase === "dying") {
    // Ống và đất đứng yên, chim rơi nốt xuống đất rồi mới hiện bảng kết quả.
    updateBird(delta);
    const sprite = birdSprite();
    if (state.bird.y + sprite.height >= GROUND_Y) {
      state.bird.y = GROUND_Y - sprite.height;
      toGameOver();
    }
  }
}

/* ---------- Vẽ ---------- */

function render() {
  drawBackground();
  drawPipes();
  drawBase();
  drawBird();

  if (state.phase === "ready") {
    drawCentered(sprites.message, Math.round((GROUND_Y - sprites.message.height) / 2));
  } else if (state.phase !== "menu" && state.phase !== "framing") {
    // Ở menu, lớp phủ HTML đã che hết — vẽ thêm điểm chỉ gây rối.
    drawDigits(state.score, GAME_WIDTH / 2, 30);
  }

}

function loop(now) {
  // Chuẩn hoá theo 60fps và chặn bước nhảy lớn khi tab bị treo/ẩn.
  const delta = state.lastFrame ? Math.min((now - state.lastFrame) / (1000 / 60), 3) : 1;
  state.lastFrame = now;
  update(delta);
  render();
  state.animation = requestAnimationFrame(loop);
}

/* ---------- Khởi động ---------- */

/** Phóng sân khấu to nhất có thể mà vẫn vừa cửa sổ, ưu tiên bội số nguyên cho nét pixel. */
function fitStage() {
  const raw = Math.min(window.innerWidth / GAME_WIDTH, window.innerHeight / GAME_HEIGHT);
  const scale = raw >= 1 ? Math.max(1, Math.floor(raw * 2) / 2) : raw;
  stage.style.transform = `scale(${scale})`;
}

function showFatalError(message) {
  ctx.fillStyle = "#533847";
  ctx.fillRect(0, 0, GAME_WIDTH, GAME_HEIGHT);
  ctx.fillStyle = "#fff";
  ctx.font = "14px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(message, GAME_WIDTH / 2, GAME_HEIGHT / 2);
}

async function boot() {
  fitStage();
  window.addEventListener("resize", fitStage);
  try {
    await loadSprites();
  } catch (error) {
    console.error(error);
    showFatalError(error.message);
    return;
  }
  loadSounds();
  toReady();
  await openMenu();
  state.animation = requestAnimationFrame(loop);
}

window.addEventListener("keydown", (event) => {
  if (event.code !== "Space" && event.code !== "ArrowUp" && event.code !== "KeyW") return;
  // Đang nhập tên: Space là dấu cách trong ô tên, không phải cú vỗ cánh.
  if (!menu.hidden && !joinForm.hidden) return;
  // Khi con trỏ đang ở một nút (nút CHƠI LẠI được focus sẵn sau khi thua), để trình
  // duyệt tự bấm nút đó — nếu không Space sẽ vừa bấm nút vừa gọi handleInput.
  if (event.target.closest?.("button, input")) return;
  event.preventDefault(); // Space mặc định cuộn trang
  if (event.repeat) return; // giữ phím không được bắn liên tục
  handleInput();
});

stage.addEventListener("pointerdown", (event) => {
  // Khung bảng xếp hạng nằm trong #stage nên mọi cú bấm vào nút của nó đều nổi bọt
  // lên đây. Chặn theo "khung có đang mở không", không theo phase: ở màn kết quả
  // phase vẫn là "gameover", nên bấm ĐỔI NGƯỜI hay XOÁ sẽ bị hiểu thành chơi lại.
  if (!menu.hidden) return;
  event.preventDefault();
  handleInput();
});

/**
 * Vào lượt chơi. Có camera thì đi qua bước canh vị trí; không có thì vào thẳng.
 * Lượt đầu sẽ tự xin quyền camera để người trực booth không phải nhắc từng bạn bấm
 * thêm nút; từ chối quyền thì rơi về chơi bằng Space, không chặn ai lại.
 */
async function startRound() {
  if (!state.pose && !state.cameraDeclined) await enableCamera({ silent: true });
  if (state.pose) toFraming();
  else toReady();
}

$("#join-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const name = leaderboard.sanitizeName(nameInput.value);
  if (!name) {
    nameInput.focus();
    return;
  }
  state.playerName = name;
  state.lastEntryId = null;
  state.lastRank = null;
  localStorage.setItem(LAST_NAME_KEY, name);
  closeMenu();
  startRound();
});

/** Chơi lại giữ nguyên tên — đường nhanh nhất, tránh tắc hàng ở booth. */
$("#replay-button").addEventListener("click", () => {
  closeMenu();
  startRound();
});

/** Người khác lên chơi: quay lại ô nhập tên. */
$("#change-player").addEventListener("click", () => {
  openMenu();
});

// Xoá tất cả cần bấm hai lần — tránh mất sạch bảng vì một cú chạm nhầm ở booth.
const clearButton = $("#clear-scores");
let clearArmed = false;
let clearTimer = 0;
clearButton.addEventListener("click", async () => {
  if (!clearArmed) {
    clearArmed = true;
    clearButton.textContent = "CHẮC CHƯA?";
    clearTimer = setTimeout(() => {
      clearArmed = false;
      clearButton.textContent = "XOÁ TẤT CẢ";
    }, 4000);
    return;
  }
  clearTimeout(clearTimer);
  clearArmed = false;
  clearButton.textContent = "XOÁ TẤT CẢ";
  await leaderboard.clear();
  state.lastEntryId = null;
  await renderBoard();
});

// Khung camera bật ở giai đoạn 3; hiện tại chỉ giữ chỗ.
/* ---------- Webcam ---------- */

function setCameraStatus(text, kind = "") {
  cameraBadge.textContent = text;
  cameraBadge.dataset.kind = kind;
}

/** Nhận mọi khung hình camera; ở bước canh vị trí thì lái luôn đồng hồ đếm ngược. */
function onPoseFrame(status) {
  if (state.phase === "framing") {
    updateFraming(status);
    return;
  }
  state.lastFramingAt = 0;

  // Tách riêng trường hợp thấy người nhưng mất dấu tay: đó là lúc người chơi quạt
  // mãi mà chim không bay, mà nếu chỉ báo chung chung thì không ai đoán ra tại sao.
  if (!status.visible) setCameraStatus("ĐỨNG VÀO KHUNG HÌNH", "warn");
  else if (!status.armsVisible) setCameraStatus("ĐƯA HAI TAY VÀO KHUNG HÌNH", "warn");
  else setCameraStatus("QUẠT TAY ĐI!", "ready");
}

function disableCamera() {
  state.pose?.stop();
  state.pose = null;
  tuning = PHYSICS.keyboard;
  cameraPip.classList.remove("live", "framing", "aligned", "near");
  cameraButton.textContent = "BẬT CAMERA";
  setCameraStatus("CAMERA TẮT");
  hint.textContent = "Nhấn Space / chạm màn hình để vỗ cánh";
}

async function enableCamera({ silent = false } = {}) {
  cameraButton.disabled = true;
  try {
    state.pose = await startPose({
      video: $("#camera"),
      overlay: $("#camera-overlay"),
      // Đi qua đúng cửa mà bàn phím và chuột vẫn dùng, nên không phải nhân đôi logic.
      onFlap: handleInput,
      onFrame: onPoseFrame,
      onStatus: setCameraStatus,
    });
    // Nhảy chậm hơn bấm phím rất nhiều, nên phải nới vật lý ra cho chơi được.
    tuning = PHYSICS.webcam;
    cameraPip.classList.add("live");
    cameraButton.textContent = "TẮT CAMERA";
    hint.textContent = "Quạt hai tay như cánh chim — Space vẫn dùng được";
  } catch (error) {
    console.error(error);
    // Nhớ lại việc bị từ chối, nếu không mỗi lượt chơi lại xin quyền một lần nữa.
    if (/denied|NotAllowed/i.test(String(error))) state.cameraDeclined = true;
    setCameraStatus("CAMERA TẮT", "warn");
    if (!silent) {
      hint.textContent = state.cameraDeclined
        ? "Bạn đã từ chối quyền camera. Cấp lại quyền trong trình duyệt rồi thử lại."
        : "Không bật được camera. Cần chạy trên localhost hoặc HTTPS.";
    }
  } finally {
    cameraButton.disabled = false;
  }
}

cameraButton.addEventListener("click", (event) => {
  event.stopPropagation(); // đừng để bấm nút thành một cú vỗ cánh
  if (state.pose) {
    disableCamera();
    return;
  }
  state.cameraDeclined = false; // bấm tay là cố ý thử lại
  enableCamera();
});

window.addEventListener("beforeunload", () => {
  cancelAnimationFrame(state.animation);
  state.pose?.stop();
});

boot();
