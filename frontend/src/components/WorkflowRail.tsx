type Step = { label: string; highlight?: boolean }

export default function WorkflowRail({ steps, accent = 'sky' }: { steps: Step[]; accent?: 'sky' | 'moss' | 'sand' }) {
  const line =
    accent === 'moss' ? 'border-moss/40' : accent === 'sand' ? 'border-sand/40' : 'border-sky/40'
  const dot =
    accent === 'moss' ? 'bg-moss' : accent === 'sand' ? 'bg-sand' : 'bg-sky'
  const hi =
    accent === 'moss' ? 'text-[#b7e4c7]' : accent === 'sand' ? 'text-sand' : 'text-sky'

  return (
    <ol className="space-y-0">
      {steps.map((s, i) => (
        <li key={s.label} className="flex gap-3">
          <div className="flex w-4 flex-col items-center">
            <span className={`mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full ${s.highlight ? dot : 'bg-white/25'}`} />
            {i < steps.length - 1 && <span className={`my-1 w-px flex-1 border-l ${line}`} />}
          </div>
          <p
            className={`pb-3 text-sm ${s.highlight ? `font-medium ${hi}` : 'text-[var(--muted)]'}`}
          >
            {s.label}
          </p>
        </li>
      ))}
    </ol>
  )
}
