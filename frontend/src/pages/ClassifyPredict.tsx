import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { classifyPredict, getClassifyStatus } from '../api/client'
import ImageUploader from '../components/ImageUploader'
import ConfidenceBar from '../components/ConfidenceBar'
import ConfidenceRing from '../components/ConfidenceRing'
import GradCAMOverlay from '../components/GradCAMOverlay'
import WorkflowRail from '../components/WorkflowRail'

function confidenceTier(c: number) {
  if (c >= 0.8) return { label: 'High Confidence', className: 'bg-moss/30 text-[#b7e4c7] border-moss/40' }
  if (c >= 0.55) return { label: 'Medium Confidence', className: 'bg-sand/20 text-sand border-sand/35' }
  return { label: 'Low Confidence', className: 'bg-ember/20 text-[#ffd6a5] border-ember/40' }
}

export default function ClassifyPredict() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState<any>(null)
  const [trained, setTrained] = useState(true)
  const [showCam, setShowCam] = useState(true)

  useEffect(() => {
    getClassifyStatus().then((s) => setTrained(!!s.trained)).catch(() => setTrained(false))
  }, [])

  const onFile = (f: File) => {
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
    setError('')
  }

  const run = async () => {
    if (!file) return
    setLoading(true)
    setError('')
    try {
      const data = await classifyPredict(file)
      setResult(data)
    } catch (e: any) {
      setError(e?.response?.data?.detail || e.message || 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  const tier = useMemo(
    () => (result ? confidenceTier(Number(result.confidence) || 0) : null),
    [result],
  )

  const top3 = (result?.top3 || []).slice(0, 3)

  return (
    <div className="space-y-6">
      <header className="anim-fade-up">
        <p className="font-mono text-xs uppercase tracking-[0.2em] text-sky">Object recognition</p>
        <h1 className="font-display text-3xl text-sand md:text-4xl">Object Classification</h1>
        <p className="mt-1 max-w-2xl text-[var(--muted)]">
          Single-label AI recognition — answers <span className="text-white/90">“What is the primary object in this patch?”</span>
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

      {result && (
        <section className="glass-card-gold anim-fade-up grid gap-4 p-5 sm:grid-cols-2 lg:grid-cols-4">
          <div>
            <p className="text-xs uppercase tracking-wider text-[var(--muted)]">Object Recognition Summary</p>
            <p className="mt-1 font-display text-2xl capitalize text-sand">{result.class}</p>
          </div>
          <div>
            <p className="text-xs uppercase tracking-wider text-[var(--muted)]">Confidence</p>
            <p className="mt-1 font-mono text-xl text-white">{(Number(result.confidence) * 100).toFixed(1)}%</p>
          </div>
          <div>
            <p className="text-xs uppercase tracking-wider text-[var(--muted)]">Inference Time</p>
            <p className="mt-1 font-mono text-xl text-white">
              {result.inference_ms != null ? `${result.inference_ms} ms` : '—'}
            </p>
          </div>
          <div>
            <p className="text-xs uppercase tracking-wider text-[var(--muted)]">Model</p>
            <p className="mt-1 text-sm text-white">{result.model || 'EfficientNet-B3 + CBAM'}</p>
          </div>
        </section>
      )}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
        {/* LEFT */}
        <div className="space-y-4">
          <div className="glass-card p-4">
            <h2 className="mb-3 font-display text-lg text-white">Input patch</h2>
            <ImageUploader onFile={onFile} label="Upload object patch" />
            {file && (
              <p className="mt-3 truncate font-mono text-xs text-[var(--muted)]">File · {file.name}</p>
            )}
            <button
              type="button"
              disabled={!file || loading || !trained}
              onClick={run}
              className="mt-4 w-full rounded-xl bg-sky px-4 py-3.5 text-sm font-semibold text-ink transition hover:brightness-110 disabled:opacity-40"
            >
              {loading ? 'Recognizing object…' : 'Classify object'}
            </button>
            {error && <p className="mt-2 text-sm text-ember">{error}</p>}
          </div>

          <div className="glass-card p-4">
            <h2 className="mb-3 font-display text-lg text-white">Recognition workflow</h2>
            <WorkflowRail
              accent="sand"
              steps={[
                { label: 'Satellite patch' },
                { label: 'CNN feature extraction' },
                { label: 'Attention module (CBAM)' },
                { label: 'Object classification' },
                {
                  label: result?.class
                    ? `Primary object → ${String(result.class)}`
                    : 'Primary object → awaiting run',
                  highlight: !!result,
                },
              ]}
            />
          </div>
        </div>

        {/* RIGHT */}
        <div className="space-y-4">
          {loading && (
            <div className="glass-card space-y-4 p-6">
              <div className="h-8 w-1/2 animate-pulse rounded bg-white/10" />
              <div className="mx-auto h-36 w-36 animate-pulse rounded-full bg-white/10" />
              <div className="h-3 animate-pulse rounded bg-white/10" />
              <div className="h-3 w-4/5 animate-pulse rounded bg-white/10" />
            </div>
          )}

          {!loading && !result && (
            <div className="glass-card flex min-h-[280px] flex-col items-center justify-center p-8 text-center">
              <p className="font-display text-xl text-sand">AI Object Recognition Engine</p>
              <p className="mt-2 max-w-sm text-sm text-[var(--muted)]">
                Upload a patch and run classification to see one primary prediction with Top-3 alternatives.
              </p>
            </div>
          )}

          {!loading && result && (
            <>
              {/* Hero */}
              <section className="glass-card-gold anim-fade-up p-6">
                <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">Predicted object</p>
                <div className="mt-4 flex flex-wrap items-center gap-8">
                  <div className="min-w-0 flex-1">
                    <h2 className="font-display text-4xl font-extrabold capitalize tracking-tight text-sand md:text-5xl">
                      {result.class}
                    </h2>
                    <p className="mt-2 text-sm text-[var(--muted)]">Prediction confidence</p>
                    {tier && (
                      <span
                        className={`mt-3 inline-flex rounded-full border px-3 py-1 text-xs font-medium ${tier.className}`}
                      >
                        {tier.label}
                      </span>
                    )}
                  </div>
                  <ConfidenceRing value={Number(result.confidence) || 0} label="confidence" />
                </div>
              </section>

              {/* Top-3 */}
              <section className="glass-card anim-fade-up p-5" style={{ animationDelay: '60ms' }}>
                <h3 className="font-display text-lg text-white">Top-3 alternative predictions</h3>
                <p className="mt-1 text-sm text-[var(--muted)]">
                  Other possible object classes considered by the model.
                </p>
                <div className="mt-4 space-y-3">
                  {top3.map((t: any) => (
                    <ConfidenceBar
                      key={t.class}
                      label={String(t.class).replace(/_/g, ' ')}
                      value={t.confidence}
                      color="#4a90a4"
                    />
                  ))}
                </div>
              </section>

              {/* Details */}
              <section className="glass-card anim-fade-up p-5" style={{ animationDelay: '100ms' }}>
                <h3 className="font-display text-lg text-white">Prediction details</h3>
                <dl className="mt-3 grid gap-3 sm:grid-cols-2">
                  {[
                    ['Model', result.model || 'EfficientNet-B3 + CBAM'],
                    ['Input size', result.input_size || '128×128'],
                    ['Inference time', result.inference_ms != null ? `${result.inference_ms} ms` : '—'],
                    ['Confidence', `${(Number(result.confidence) * 100).toFixed(1)}%`],
                    ['Prediction status', 'Accepted'],
                    ['Output type', 'Single label'],
                  ].map(([k, v]) => (
                    <div key={k} className="rounded-xl border border-white/10 bg-black/25 px-3 py-2.5">
                      <dt className="text-[10px] uppercase tracking-wider text-[var(--muted)]">{k}</dt>
                      <dd className="mt-0.5 text-sm capitalize text-white">{v}</dd>
                    </div>
                  ))}
                </dl>
              </section>

              {/* GradCAM */}
              <section className="glass-card anim-fade-up p-5" style={{ animationDelay: '140ms' }}>
                <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <h3 className="font-display text-lg text-white">Explainability · GradCAM</h3>
                    <p className="mt-1 max-w-xl text-sm text-[var(--muted)]">
                      Highlighted regions show where the model focused to recognize the detected object.
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
                  overlayBase64={result.gradcam_base64}
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
