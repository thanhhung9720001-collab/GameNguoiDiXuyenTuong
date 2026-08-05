export default function Home() {
  return (
    <main className="game-shell">
      <iframe
        className="game-frame"
        src="/game/index.html"
        title="Người Đi Xuyên Tường — AI ExerGame"
        allow="camera; autoplay"
      />
    </main>
  );
}
