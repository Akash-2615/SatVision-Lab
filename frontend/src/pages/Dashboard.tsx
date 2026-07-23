import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getAnnotateStatus, getClassifyStatus, getLogs, getHealth } from '../api/client'

export default function Dashboard() {
  const [health, setHealth] = useState<any>(null)
  const [clf, setClf] = useState<any>(null)
  const [ann, setAnn] = useState<any>(null)
  const [recent, setRecent] = useState<any[]>([])

  useEffect(() => {
    Promise.all([getHealth(), getClassifyStatus(), getAnnotateStatus(), getLogs()]).then(
      ([h, c, a, l]) => {
        setHealth(h)
        setClf(c)
        setAnn(a)
        setRecent((l.logs || []).slice(-5).reverse())
      },
    )
  }, [])

  return (
    <div className="space-y-8">
      <section className="relative overflow-hidden rounded-2xl border border-[var(--line)] bg-gradient-to-br from-[#132533] via-[#0f2230] to-[#1a3040] p-8 md:p-12">
        <div
          className="pointer-events-none absolute inset-0 opacity-40"
          style={{
            backgroundImage:
              'radial-gradient(circle at 20% 30%, rgba(74,144,164,0.35), transparent 40%), radial-gradient(circle at 80% 70%, rgba(45,106,79,0.3), transparent 35%)',
          }}
        />
        <div className="relative max-w-2xl">
          <h1 className="font-display text-4xl font-extrabold tracking-tight text-sand md:text-5xl">
            SatVision Lab
          </h1>
          <p className="mt-3 text-lg text-[var(--muted)]">
            Classify tiny objects in aerial patches and annotate full satellite scenes — local,
            offline, no cloud.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Link to="/classify" className="rounded-md bg-sky px-4 py-2 text-sm font-medium text-ink">
              Object Classification
            </Link>
            <Link
              to="/annotate"
              className="rounded-md bg-moss px-4 py-2 text-sm font-medium text-white"
            >
              Scene Annotation
            </Link>
            <Link
              to="/training"
              className="rounded-md border border-sand/40 px-4 py-2 text-sm text-sand"
            >
              Go to Train
            </Link>
            <Link
              to="/dataset"
              className="rounded-md border border-white/15 px-4 py-2 text-sm text-white/80"
            >
              Manage Dataset
            </Link>
          </div>
          {health && (
            <p className="mt-4 font-mono text-xs text-[var(--muted)]">
              device={health.device} · classifier={String(health.models_loaded.classifier)} ·
              annotator={String(health.models_loaded.annotator)}
            </p>
          )}
        </div>
      </section>

      <div className="grid gap-5 md:grid-cols-2">
        <StatusCard
          title="Object Classification"
          trained={clf?.trained}
          metricLabel="Val accuracy"
          metric={clf?.accuracy != null ? `${(clf.accuracy * 100).toFixed(1)}%` : '—'}
          detail={`${(clf?.class_list || []).length} classes · single label`}
          to="/classify"
        />
        <StatusCard
          title="Scene Annotation"
          trained={ann?.trained}
          metricLabel="Val mAP"
          metric={ann?.map != null ? Number(ann.map).toFixed(3) : '—'}
          detail={`${(ann?.labels || []).length} labels · multi-label`}
          to="/annotate"
        />
      </div>

      <section>
        <h2 className="mb-3 font-display text-xl text-white">Recent predictions</h2>
        {recent.length === 0 ? (
          <p className="text-sm text-[var(--muted)]">No predictions yet.</p>
        ) : (
          <ul className="divide-y divide-white/10 overflow-hidden rounded-xl border border-[var(--line)] bg-black/20">
            {recent.map((r, i) => (
              <li key={i} className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 text-sm">
                <span className="font-mono text-xs uppercase text-sky">{r.task}</span>
                <span className="flex-1 truncate">{r.result}</span>
                <span className="font-mono text-xs text-[var(--muted)]">
                  {r.confidence != null ? `${(r.confidence * 100).toFixed(1)}%` : ''}
                </span>
                <span className="text-xs text-[var(--muted)]">
                  {r.timestamp ? new Date(r.timestamp).toLocaleString() : ''}
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  )
}

function StatusCard({
  title,
  trained,
  metricLabel,
  metric,
  detail,
  to,
}: {
  title: string
  trained?: boolean
  metricLabel: string
  metric: string
  detail: string
  to: string
}) {
  return (
    <Link
      to={to}
      className="block rounded-xl border border-[var(--line)] bg-[#132533]/80 p-6 transition hover:border-sky/40"
    >
      <div className="flex items-start justify-between gap-3">
        <h3 className="font-display text-lg text-white">{title}</h3>
        <span
          className={`rounded px-2 py-0.5 font-mono text-xs ${
            trained ? 'bg-moss/30 text-[#b7e4c7]' : 'bg-ember/20 text-[#ffd6a5]'
          }`}
        >
          {trained ? 'trained' : 'untrained'}
        </span>
      </div>
      <p className="mt-4 text-sm text-[var(--muted)]">{metricLabel}</p>
      <p className="font-display text-3xl text-sand">{metric}</p>
      <p className="mt-2 text-xs text-[var(--muted)]">{detail}</p>
    </Link>
  )
}
