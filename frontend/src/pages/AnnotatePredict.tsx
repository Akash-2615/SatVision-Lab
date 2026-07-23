import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { annotatePredict, getAnnotateStatus } from '../api/client'
import ImageUploader from '../components/ImageUploader'
import ConfidenceBar from '../components/ConfidenceBar'
import GradCAMOverlay from '../components/GradCAMOverlay'
import LabelChip from '../components/LabelChip'
import WorkflowRail from '../components/WorkflowRail'

export default function AnnotatePredict() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [raw, setRaw] = useState<any>(null)
  const [threshold, setThreshold] = useState(0.45)
  const [trained, setTrained] = useState(true)
  const [showCam, setShowCam] = useState(true)

  useEffect(() => {
    getAnnotateStatus()
      .then((s) => {
        setTrained(!!s.trained)
        if (s.threshold != null) setThreshold(Number(s.threshold))
      })
      .catch(() => setTrained(false))
  }, [])

  const labels = useMemo(() => {
    if (!raw?.labels) return []
    return [...raw.labels]
      .map((l: any) => ({
        ...l,
        above_threshold: Number(l.score) >= threshold,
      }))
      .sort((a: any, b: any) => Number(b.score) - Number(a.score))
  }, [raw, threshold])

  const accepted = useMemo(() => labels.filter((l: any) => l.above_threshold), [labels])
  const rejected = useMemo(() => labels.filter((l: any) => !l.above_threshold), [labels])
  const topLabel = accepted[0] || labels[0]

  const onFile = (f: File) => {
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setRaw(null)
    setError('')
  }

  const run = async () => {
    if (!file) return
    setLoading(true)
    setError('')
    try {
      const data = await annotatePredict(file, threshold)
      setRaw(data)
    } catch (e: any) {
      setError(e?.response?.data?.detail || e.message || 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <header className="anim-fade-up">
        <p className="font-mono text-xs uppercase tracking-[0.2em] text-moss">Scene annotation</p>
        <h1 className="font-display text-3xl text-sand md:text-4xl">Scene Annotation</h1>
        <p className="mt-1 max-w-2xl text-[var(--muted)]">
          Multi-label AI annotation — answers{' '}
          <span className="text-white/90">“What semantic labels describe this entire satellite scene?”</span>
        </p>
      </header>

      {!trained && (
        <div className="rounded-xl border border-ember/40 bg-ember/10 px-4 py-3 text-sm">
          Model not trained.{' '}
          <Link to="/training" className="text-sand underline">
            Go to Training
          </Link>
        </div>
      )}

      {raw && (
        <section className="glass-card-moss anim-fade-up grid gap-4 p-5 sm:grid-cols-2 lg:grid-cols-4">
          <div>
            <p className="text-xs uppercase tracking-wider text-[var(--muted)]">Scene Annotation Summary</p>
            <p className="mt-1 font-display text-2xl text-[#b7e4c7]">
              {accepted.length}{' '}
              <span className="text-base font-normal text-[var(--muted)]">annotations</span>
            </p>
          </div>
          <div>
            <p className="text-xs uppercase tracking-wider text-[var(--muted)]">Highest confidence</p>
            <p className="mt-1 capitalize text-white">
              {topLabel ? String(topLabel.name).replace(/_/g, ' ') : '—'}
            </p>
            <p className="font-mono text-sm text-sand">
              {topLabel ? `${(Number(topLabel.score) * 100).toFixed(1)}%` : ''}
            </p>
          </div>
          <div>
            <p className="text-xs uppercase tracking-wider text-[var(--muted)]">Threshold</p>
            <p className="mt-1 font-mono text-xl text-white">{(threshold * 100).toFixed(0)}%</p>
          </div>
          <div>
            <p className="text-xs uppercase tracking-wider text-[var(--muted)]">Inference</p>
            <p className="mt-1 font-mono text-xl text-white">
              {raw.inference_ms != null ? `${raw.inference_ms} ms` : '—'}
            </p>
          </div>
        </section>
      )}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
        {/* LEFT */}
        <div className="space-y-4">
          <div className="glass-card p-4">
            <h2 className="mb-3 font-display text-lg text-white">Satellite scene</h2>
            <ImageUploader onFile={onFile} label="Upload satellite image" />
            {file && (
              <p className="mt-3 truncate font-mono text-xs text-[var(--muted)]">File · {file.name}</p>
            )}

            <div className="mt-4">
              <div className="mb-1 flex justify-between text-sm">
                <span>Annotation threshold</span>
                <span className="font-mono text-moss">{(threshold * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min={0.05}
                max={0.95}
                step={0.05}
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
                className="w-full accent-[var(--moss)]"
              />
              <p className="mt-1 text-xs text-[var(--muted)]">
                Only scores ≥ threshold become generated annotations.
              </p>
            </div>

            <button
              type="button"
              disabled={!file || loading || !trained}
              onClick={run}
              className="mt-4 w-full rounded-xl bg-moss px-4 py-3.5 text-sm font-semibold text-white transition hover:brightness-110 disabled:opacity-40"
            >
              {loading ? 'Generating annotations…' : 'Generate annotations'}
            </button>
            {error && <p className="mt-2 text-sm text-ember">{error}</p>}
          </div>

          <div className="glass-card p-4">
            <h2 className="mb-3 font-display text-lg text-white">Annotation workflow</h2>
            <WorkflowRail
              accent="moss"
              steps={[
                { label: 'Satellite scene' },
                { label: 'EfficientNet-B3 + CBAM' },
                { label: 'Predict confidence for every label' },
                { label: `Threshold filtering (${(threshold * 100).toFixed(0)}%)` },
                {
                  label: raw
                    ? `Generate image annotations → ${accepted.length} labels`
                    : 'Generate image annotations',
                  highlight: !!raw,
                },
              ]}
            />
          </div>
        </div>

        {/* RIGHT */}
        <div className="space-y-4">
          {loading && (
            <div className="glass-card space-y-4 p-6">
              <div className="h-6 w-2/3 animate-pulse rounded bg-white/10" />
              <div className="flex flex-wrap gap-2">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="h-8 w-28 animate-pulse rounded-md bg-white/10" />
                ))}
              </div>
              <div className="h-3 animate-pulse rounded bg-white/10" />
              <div className="h-3 w-5/6 animate-pulse rounded bg-white/10" />
            </div>
          )}

          {!loading && !raw && (
            <div className="glass-card flex min-h-[280px] flex-col items-center justify-center p-8 text-center">
              <p className="font-display text-xl text-[#b7e4c7]">AI Scene Annotation Platform</p>
              <p className="mt-2 max-w-sm text-sm text-[var(--muted)]">
                Upload a full scene to generate multiple land-cover labels filtered by your threshold.
              </p>
            </div>
          )}

          {!loading && raw && (
            <>
              {/* Generated annotations */}
              <section className="glass-card-moss anim-fade-up p-5">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <div>
                    <h3 className="font-display text-lg text-white">Generated annotations</h3>
                    <p className="mt-1 text-sm text-[var(--muted)]">
                      Labels automatically assigned to this satellite image.
                    </p>
                  </div>
                  <p className="font-mono text-sm text-[#b7e4c7]">
                    {accepted.length} annotation{accepted.length === 1 ? '' : 's'} generated
                  </p>
                </div>
                {accepted.length ? (
                  <div className="mt-4 flex flex-wrap gap-2">
                    {accepted.map((l: any) => (
                      <LabelChip key={l.name} name={l.name} active score={l.score} />
                    ))}
                  </div>
                ) : (
                  <p className="mt-4 text-sm text-[var(--muted)]">
                    No labels cleared the threshold — lower it to accept more annotations.
                  </p>
                )}
              </section>

              {/* All scores */}
              <section className="glass-card anim-fade-up p-5" style={{ animationDelay: '60ms' }}>
                <h3 className="font-display text-lg text-white">AI prediction scores</h3>
                <p className="mt-1 text-sm text-[var(--muted)]">
                  Confidence for every land-cover category. Green = above threshold, grey = below.
                </p>
                <div className="mt-4 max-h-80 space-y-2.5 overflow-auto pr-1">
                  {labels.map((l: any) => (
                    <ConfidenceBar
                      key={l.name}
                      label={String(l.name).replace(/_/g, ' ')}
                      value={l.score}
                      color={l.above_threshold ? '#2d6a4f' : '#5a6d78'}
                    />
                  ))}
                </div>
              </section>

              {/* Threshold decision */}
              <section className="glass-card anim-fade-up p-5" style={{ animationDelay: '100ms' }}>
                <h3 className="font-display text-lg text-white">Threshold decision</h3>
                <p className="mt-1 text-sm text-[var(--muted)]">
                  Threshold = {(threshold * 100).toFixed(0)}%. Annotations are generated only from accepted
                  predictions.
                </p>
                <div className="mt-4 grid gap-4 sm:grid-cols-2">
                  <div className="rounded-xl border border-moss/30 bg-moss/10 p-3">
                    <p className="mb-2 text-xs font-medium uppercase tracking-wider text-[#b7e4c7]">
                      Accepted
                    </p>
                    {accepted.length ? (
                      <ul className="space-y-1.5">
                        {accepted.map((l: any) => (
                          <li key={l.name} className="flex items-center gap-2 text-sm capitalize text-white">
                            <span className="text-moss">✔</span>
                            {String(l.name).replace(/_/g, ' ')}
                            <span className="ml-auto font-mono text-xs text-[var(--muted)]">
                              {(Number(l.score) * 100).toFixed(1)}%
                            </span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-sm text-[var(--muted)]">None</p>
                    )}
                  </div>
                  <div className="rounded-xl border border-white/10 bg-black/25 p-3">
                    <p className="mb-2 text-xs font-medium uppercase tracking-wider text-[var(--muted)]">
                      Rejected
                    </p>
                    {rejected.length ? (
                      <ul className="max-h-48 space-y-1.5 overflow-auto">
                        {rejected.map((l: any) => (
                          <li
                            key={l.name}
                            className="flex items-center gap-2 text-sm capitalize text-[var(--muted)]"
                          >
                            <span className="text-ember/80">✖</span>
                            {String(l.name).replace(/_/g, ' ')}
                            <span className="ml-auto font-mono text-xs">
                              {(Number(l.score) * 100).toFixed(1)}%
                            </span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-sm text-[var(--muted)]">None</p>
                    )}
                  </div>
                </div>
              </section>

              {/* GradCAM */}
              <section className="glass-card anim-fade-up p-5" style={{ animationDelay: '140ms' }}>
                <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <h3 className="font-display text-lg text-white">Explainability · GradCAM</h3>
                    <p className="mt-1 max-w-xl text-sm text-[var(--muted)]">
                      Highlighted regions indicate where the model focused while generating semantic
                      annotations.
                    </p>
                  </div>
                  <label className="flex items-center gap-2 text-xs text-[var(--muted)]">
                    <input
                      type="checkbox"
                      checked={showCam}
                      onChange={(e) => setShowCam(e.target.checked)}
                    />
                    Show heatmap
                  </label>
                </div>
                <GradCAMOverlay
                  src={preview || undefined}
                  overlayBase64={raw.gradcam_base64}
                  showOverlay={showCam}
                />
              </section>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
