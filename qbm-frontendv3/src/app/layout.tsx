import type { Metadata } from "next";
import "./globals.css";
import "@crayonai/react-ui/styles/index.css";
import { Navigation } from "./components/Navigation";
import { Providers } from "./providers";

export const metadata: Metadata = {
  title: "QBM - Quranic Behavioral Matrix",
  description:
    "AI-powered research platform for exploring behavioral classifications across the Holy Quran",
  keywords: [
    "Quran",
    "Islamic Studies",
    "Behavioral Analysis",
    "Tafsir",
    "AI Research",
    "Natural Language Processing",
  ],
  openGraph: {
    title: "Quranic Behavioral Matrix",
    description:
      "World's first AI-powered platform for Quranic behavioral research",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gray-50 antialiased">
        <Providers>
          <Navigation />
          {children}
        </Providers>
      </body>
    </html>
  );
}
