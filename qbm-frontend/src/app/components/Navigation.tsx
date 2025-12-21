"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BookOpen, LayoutDashboard, PenTool, Home } from "lucide-react";

const navItems = [
  { href: "/", label: "Home", icon: Home },
  { href: "/research", label: "Research", icon: BookOpen },
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/annotate", label: "Annotate", icon: PenTool },
];

export function Navigation() {
  const pathname = usePathname();

  return (
    <nav className="bg-emerald-800 text-white shadow-lg">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-emerald-600 rounded-lg flex items-center justify-center">
              <span className="text-xl font-bold">Q</span>
            </div>
            <div>
              <span className="font-bold text-lg">QBM</span>
              <span className="text-emerald-300 text-sm block -mt-1">
                Behavioral Matrix
              </span>
            </div>
          </Link>

          {/* Navigation Links */}
          <div className="flex items-center space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;
              
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`
                    flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors
                    ${isActive 
                      ? "bg-emerald-700 text-white" 
                      : "text-emerald-200 hover:bg-emerald-700 hover:text-white"
                    }
                  `}
                >
                  <Icon size={18} />
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </div>

          {/* Arabic Toggle (placeholder) */}
          <div className="flex items-center space-x-2">
            <span className="text-emerald-300 text-sm">العربية</span>
            <span className="text-emerald-300">|</span>
            <span className="text-white text-sm">English</span>
          </div>
        </div>
      </div>
    </nav>
  );
}
