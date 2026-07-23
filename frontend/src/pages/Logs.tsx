import { useEffect, useMemo, useState } from 'react'
import { clearLogs, getLogs } from '../api/client'

export default function Logs() {
  const [logs, setLogs] = useState<any[]>([])
  const [filter, setFilter] = useState<'all' | 'classifier' | 'annotator' | 'pipeline'>('all')

  const refresh = () => getLogs().then((d) => setLogs([...(d.logs || [])].reverse()))

  useEffect(() => {
    refresh().catch(console.error)
  }, [])

  const filtered = useMemo(() => {
    if (filter === 'all') return logs
    return logs.filter((l) => l.task === filter)
  }, [logs, filter])

  const wipe = async () => {
    if (!confirm('Clear all prediction logs?')) return
    await clearLogs()
    setLogs([])
  }

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-display text-3xl text-sand">Prediction Logs</h1>
          <p className="text-[var(--muted)]">Append-only local JSON history</p>
        </div>
        <button
          type="button"
          onClick={wipe}
          className="rounded-md border border-ember/40 px-3 py-1.5 text-sm text-ember"
        >
          Clear all
        </button>
      </header>

      <div className="flex flex-wrap gap-2">
        {(['all', 'classifier', 'annotator', 'pipeline'] as const).map((f) => (
          <button
            key={f}
            type="button"
            onClick={() => setFilter(f)}
            className={`rounded-md px-3 py-1.5 text-sm capitalize ${
              filter === f ? 'bg-sky/30 text-white' : 'bg-white/5 text-[var(--muted)]'
            }`}
          >
            {f}
          </button>
        ))}
      </div>

      <div className="overflow-x-auto rounded-xl border border-[var(--line)]">
        <table className="min-w-full text-left text-sm">
          <thead className="bg-black/40 text-xs uppercase text-[var(--muted)]">
            <tr>
              <th className="px-4 py-3">Timestamp</th>
              <th className="px-4 py-3">Task</th>
              <th className="px-4 py-3">Image</th>
              <th className="px-4 py-3">Result</th>
              <th className="px-4 py-3">Confidence</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r, i) => (
              <tr key={i} className="border-t border-white/10 bg-black/20">
                <td className="whitespace-nowrap px-4 py-3 font-mono text-xs text-[var(--muted)]">
                  {r.timestamp ? new Date(r.timestamp).toLocaleString() : '—'}
                </td>
                <td className="px-4 py-3 font-mono text-xs uppercase text-sky">{r.task}</td>
                <td className="px-4 py-3">
                  {r.image ? (
                    <img
                      src={`/files/uploads/${r.image}`}
                      alt=""
                      className="h-12 w-12 rounded object-cover"
                      onError={(e) => {
                        ;(e.target as HTMLImageElement).style.display = 'none'
                      }}
                    />
                  ) : (
                    '—'
                  )}
                </td>
                <td className="max-w-xs truncate px-4 py-3">{r.result}</td>
                <td className="px-4 py-3 font-mono text-xs">
                  {r.confidence != null ? `${(Number(r.confidence) * 100).toFixed(1)}%` : '—'}
                </td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={5} className="px-4 py-8 text-center text-[var(--muted)]">
                  No logs yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
