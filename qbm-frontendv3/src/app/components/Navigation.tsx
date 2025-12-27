"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BookOpen,
  BarChart3,
  MessageSquare,
  Sparkles,
  Globe,
  Home,
  Menu,
  X,
  Search,
  FileCheck,
  Database,
  ClipboardCheck,
} from "lucide-react";
import { useState } from "react";
import { useLanguage } from "../contexts/LanguageContext";

const navItems = [
  { href: "/", labelKey: "nav.home", icon: Home },
  { href: "/research", labelKey: "nav.research", icon: MessageSquare },
  { href: "/proof", labelKey: "nav.proof", icon: FileCheck },
  { href: "/explorer", labelKey: "nav.explorer", icon: Globe },
  { href: "/genome", labelKey: "nav.genome", icon: Database },
  { href: "/reviews", labelKey: "nav.reviews", icon: ClipboardCheck },
  { href: "/discovery", labelKey: "nav.discovery", icon: Search },
  { href: "/dashboard", labelKey: "nav.dashboard", icon: BarChart3 },
];

export function Navigation() {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const { language, setLanguage, t, isRTL } = useLanguage();

  return (
    <nav className={`bg-emerald-800 text-white sticky top-0 z-50 ${isRTL ? 'rtl' : 'ltr'}`}>
      <div className="max-w-7xl mx-auto px-4 lg:px-6">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-3 group">
            <div className="w-10 h-10 bg-gradient-to-br from-emerald-500 to-emerald-600 rounded-xl flex items-center justify-center shadow-lg group-hover:scale-105 transition-transform">
              <span className="text-xl font-bold">Q</span>
            </div>
            <div className="hidden sm:block">
              <span className="font-bold text-lg">QBM</span>
              <span className="text-emerald-300 text-xs block -mt-1">
                Behavioral Matrix
              </span>
            </div>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive =
                pathname === item.href ||
                (item.href !== "/" && pathname.startsWith(item.href));

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`nav-link ${isActive ? "active" : ""}`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{t(item.labelKey)}</span>
                </Link>
              );
            })}
          </div>

          {/* Right side */}
          <div className="flex items-center gap-4">
            <div className="hidden lg:flex items-center gap-2 text-sm">
              <button
                onClick={() => setLanguage("ar")}
                className={`px-2 py-1 rounded ${language === "ar" ? "bg-emerald-600 text-white font-medium" : "text-emerald-300 hover:text-white"}`}
              >
                العربية
              </button>
              <span className="text-emerald-500">|</span>
              <button
                onClick={() => setLanguage("en")}
                className={`px-2 py-1 rounded ${language === "en" ? "bg-emerald-600 text-white font-medium" : "text-emerald-300 hover:text-white"}`}
              >
                English
              </button>
            </div>

            {/* Mobile menu button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-2 hover:bg-emerald-700 rounded-lg"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <div className="md:hidden py-4 border-t border-emerald-700">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive =
                pathname === item.href ||
                (item.href !== "/" && pathname.startsWith(item.href));

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={() => setMobileMenuOpen(false)}
                  className={`flex items-center gap-3 px-4 py-3 rounded-lg ${
                    isActive ? "bg-emerald-700" : "hover:bg-emerald-700/50"
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span>{t(item.labelKey)}</span>
                </Link>
              );
            })}
          </div>
        )}
      </div>
    </nav>
  );
}
