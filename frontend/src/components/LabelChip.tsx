type Props = {
  name: string
  active?: boolean
  /** Confidence 0–1 (shown as %). */
  score?: number
  /** Dataset image count for this label. */
  count?: number
  onClick?: () => void
}

export default function LabelChip({ name, active, score, count, onClick }: Props) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`inline-flex items-center gap-2 rounded-md border px-3 py-1.5 text-sm transition ${
        active
          ? 'border-moss/60 bg-moss/25 text-[#d8f3dc]'
          : 'border-white/10 bg-white/5 text-[var(--muted)]'
      } ${onClick ? 'cursor-pointer hover:border-moss/40' : 'cursor-default'}`}
    >
      <span>{name.replace(/_/g, ' ')}</span>
      {typeof count === 'number' && (
        <span className="font-mono text-xs opacity-80">{count}</span>
      )}
      {typeof score === 'number' && (
        <span className="font-mono text-xs opacity-80">{(score * 100).toFixed(0)}%</span>
      )}
    </button>
  )
}
