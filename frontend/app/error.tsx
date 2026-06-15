"use client";

export default function Error({
  reset,
}: {
  error: Error;
  reset: () => void;
}) {
  return (
    <main className="max-w-5xl mx-auto px-4 py-20 text-center">
      <div className="rounded-2xl border border-amber-200 bg-amber-50 p-10 inline-block">
        <p className="text-4xl mb-4">⏳</p>
        <h2 className="text-xl font-semibold text-amber-900 mb-2">
          API is waking up
        </h2>
        <p className="text-sm text-amber-700 mb-6 max-w-sm">
          The prediction server runs on Render free tier and sleeps after inactivity.
          First request takes 30-60 seconds. Please try again.
        </p>
        <button
          onClick={reset}
          className="rounded-lg bg-amber-600 px-5 py-2 text-sm font-medium text-white hover:bg-amber-700 transition-colors"
        >
          Reload
        </button>
      </div>
    </main>
  );
}
