type Props = {
  value: number
  size?: number
  stroke?: number
  color?: string
  label?: string
}

/** Animated circular confidence meter (0–1). */
export default function ConfidenceRing({
  value,
  size = 148,
  stroke = 10,
  color = '#c9a66b',
  label,
}: Props) {
  const pct = Math.max(0, Math.min(1, Number(value) || 0))
  const r = (size - stroke) / 2
  const circ = 2 * Math.PI * r
  const offset = circ * (1 - pct)

  return (
    <div className="relative inline-flex items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="rgba(255,255,255,0.08)"
          strokeWidth={stroke}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circ}
          strokeDashoffset={offset}
          className="anim-ring"
          style={{ ['--ring-circ' as string]: String(circ) }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-display text-3xl font-bold text-sand">{(pct * 100).toFixed(1)}%</span>
        {label && <span className="mt-0.5 text-[10px] uppercase tracking-wider text-[var(--muted)]">{label}</span>}
      </div>
    </div>
  )
}
