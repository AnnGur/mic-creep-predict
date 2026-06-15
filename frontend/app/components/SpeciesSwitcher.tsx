"use client";

import { useRouter, usePathname } from "next/navigation";

const SPECIES = [
  { key: "kpneumoniae", label: "K. pneumoniae" },
  { key: "abaumannii",  label: "A. baumannii" },
];

export default function SpeciesSwitcher({ current }: { current: string }) {
  const router   = useRouter();
  const pathname = usePathname();

  return (
    <div className="inline-flex rounded-lg border border-gray-200 bg-gray-50 p-1 gap-1">
      {SPECIES.map(({ key, label }) => (
        <button
          key={key}
          onClick={() => router.push(`${pathname}?species=${key}`)}
          className={`rounded-md px-4 py-1.5 text-sm font-medium italic transition-colors ${
            current === key
              ? "bg-white shadow-sm text-blue-600 border border-gray-200"
              : "text-gray-500 hover:text-gray-800"
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
