"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "MIC Trend" },
  { href: "/countries", label: "Countries" },
];

export default function Nav() {
  const pathname = usePathname();
  return (
    <nav className="border-b border-gray-200 bg-white">
      <div className="max-w-5xl mx-auto px-4 flex items-center gap-6 h-12">
        <span className="text-sm font-bold text-gray-900 mr-2">MIC Creep Watch</span>
        {LINKS.map((l) => (
          <Link
            key={l.href}
            href={l.href}
            className={`text-sm font-medium transition-colors ${
              pathname === l.href
                ? "text-blue-600 border-b-2 border-blue-600 pb-0.5"
                : "text-gray-500 hover:text-gray-800"
            }`}
          >
            {l.label}
          </Link>
        ))}
      </div>
    </nav>
  );
}
