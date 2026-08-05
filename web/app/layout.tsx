import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Người Đi Xuyên Tường — AI ExerGame",
  description: "Game vận động AI nhận diện tư thế qua webcam ngay trong trình duyệt.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="vi"><body>{children}</body></html>;
}
