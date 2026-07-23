type Props = {
  value: number
  label?: string
  color?: string
}

export default function ConfidenceBar({ value, label, color = '#4a90a4' }: Props) {
  const pct = Math.max(0, Math.min(100, value * 100))
  return (
    <div className="w-full">
      {(label || label === '') && (
        <div className="mb-1 flex items-center justify-between gap-3 text-sm">
          <span className="truncate">{label}</span>
          <span className="font-mono text-xs text-[var(--muted)]">{pct.toFixed(1)}%</span>
        </div>
      )}
      <div className="h-2 overflow-hidden rounded-full bg-white/10">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  )
}
