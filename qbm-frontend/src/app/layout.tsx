import type { Metadata } from "next";
import "./globals.css";
import "@crayonai/react-ui/styles/index.css";
import { Navigation } from "./components/Navigation";

export const metadata: Metadata = {
  title: "QBM - Quranic Behavioral Matrix",
  description: "Research interface for the Quranic Human-Behavior Classification Matrix",
  keywords: ["Quran", "Behavior", "Islamic Studies", "Research", "Tafsir"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" dir="ltr">
      <body className="min-h-screen bg-gray-50">
        <Navigation />
        {children}
      </body>
    </html>
  );
}
