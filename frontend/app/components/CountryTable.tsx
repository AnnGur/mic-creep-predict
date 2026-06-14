"use client";

import { useState } from "react";

type CountryRow = {
  country: string;
  n: number;
  pct_resistant: number;
  pct_resistant_pred: number;
  mic90_actual: number;
  mic90_predicted: number;
};

type SortKey = keyof CountryRow;

function ResistanceBadge({ pct }: { pct: number }) {
  const colour =
    pct >= 50
      ? "bg-red-100 text-red-700"
      : pct >= 20
      ? "bg-orange-100 text-orange-700"
      : pct >= 5
      ? "bg-yellow-100 text-yellow-700"
      : "bg-green-100 text-green-700";
  return (
    <span className={`inline-block rounded-full px-2 py-0.5 text-xs font-semibold ${colour}`}>
      {pct.toFixed(1)}%
    </span>
  );
}

export default function CountryTable({ data }: { data: CountryRow[] }) {
  const [sortKey, setSortKey] = useState<SortKey>("pct_resistant");
  const [asc, setAsc] = useState(false);

  function handleSort(key: SortKey) {
    if (key === sortKey) setAsc((a) => !a);
    else { setSortKey(key); setAsc(false); }
  }

  const sorted = [...data].sort((a, b) => {
    const va = a[sortKey];
    const vb = b[sortKey];
    const cmp = typeof va === "string" ? (va as string).localeCompare(vb as string) : (va as number) - (vb as number);
    return asc ? cmp : -cmp;
  });

  function Th({ label, k }: { label: string; k: SortKey }) {
    const active = sortKey === k;
    return (
      <th
        className="px-3 py-2 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide cursor-pointer hover:text-gray-800 select-none whitespace-nowrap"
        onClick={() => handleSort(k)}
      >
        {label}{" "}
        <span className={active ? "text-blue-600" : "text-gray-300"}>
          {active ? (asc ? "▲" : "▼") : "▲▼"}
        </span>
      </th>
    );
  }

  return (
    <div className="overflow-x-auto rounded-xl border border-gray-200">
      <table className="min-w-full text-sm">
        <thead className="bg-gray-50 border-b border-gray-200">
          <tr>
            <th className="px-3 py-2 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide w-6">#</th>
            <Th label="Country" k="country" />
            <Th label="Isolates" k="n" />
            <Th label="% Resistant (actual)" k="pct_resistant" />
            <Th label="% Resistant (model)" k="pct_resistant_pred" />
            <Th label="MIC₉₀ actual (mg/L)" k="mic90_actual" />
            <Th label="MIC₉₀ model (mg/L)" k="mic90_predicted" />
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {sorted.map((row, i) => (
            <tr key={row.country} className="hover:bg-gray-50 transition-colors">
              <td className="px-3 py-2 text-gray-400 text-xs">{i + 1}</td>
              <td className="px-3 py-2 font-medium text-gray-800">{row.country}</td>
              <td className="px-3 py-2 text-gray-600">{row.n.toLocaleString()}</td>
              <td className="px-3 py-2"><ResistanceBadge pct={row.pct_resistant} /></td>
              <td className="px-3 py-2 text-gray-600">{row.pct_resistant_pred.toFixed(1)}%</td>
              <td className="px-3 py-2 text-gray-600">{row.mic90_actual}</td>
              <td className="px-3 py-2 text-gray-600">{row.mic90_predicted.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
