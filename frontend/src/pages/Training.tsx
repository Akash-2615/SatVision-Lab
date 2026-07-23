import {
  Component,
  useEffect,
  useMemo,
  useState,
  type ReactElement,
  type ReactNode,
} from 'react'
import {
  adminLogin,
  adminLogout,
  adminMe,
  getAnnotateStatus,
  getClassifyStatus,
  getTrainingLogs,
  trainAnnotator,
  trainClassifier,
} from '../api/client'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  BarChart,
  Bar,
  Legend,
} from 'recharts'

type ChartPoint = { epoch: number; loss?: number; acc?: number; map?: number }

class ChartErrorBoundary extends Component<
  { children: ReactNode; fallback?: ReactNode },
  { error: string | null }
> {
  state = { error: null as string | null }
  static getDerivedStateFromError(err: Error) {
    return { error: err.message || 'Chart failed to render' }
  }
  render() {
    if (this.state.error) {
      return (
        <p className="rounded border border-ember/40 bg-ember/10 px-3 py-2 text-xs text-[#ffd6a5]">
          Chart error: {this.state.error}
        </p>
      )
    }
    return this.props.children
  }
}

function SafeChart({ children, height = 220 }: { children: ReactElement; height?: number }) {
  return (
    <ChartErrorBoundary>
      <div className="w-full min-w-0" style={{ height, minHeight: height }}>
        <ResponsiveContainer width="100%" height="100%" minWidth={200} minHeight={height}>
          {children}
        </ResponsiveContainer>
      </div>
    </ChartErrorBoundary>
  )
}

function pct(v: number | null | undefined) {
  if (v == null || Number.isNaN(Number(v))) return '—'
  return `${(Number(v) * 100).toFixed(1)}%`
}

function num(v: number | null | undefined, dig = 3) {
  if (v == null || Number.isNaN(Number(v))) return '—'
  return Number(v).toFixed(dig)
}

export default function Training() {
  const [clf, setClf] = useState<any>(null)
  const [ann, setAnn] = useState<any>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState('')

  const [isAdmin, setIsAdmin] = useState(false)
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [authMsg, setAuthMsg] = useState('')
  const [authBusy, setAuthBusy] = useState(false)

  const [busy, setBusy] = useState<'clf' | 'ann' | null>(null)
  const [msg, setMsg] = useState('')

  const [clfEpochs, setClfEpochs] = useState(10)
  const [clfLr, setClfLr] = useState(0.00008)
  const [clfBatch, setClfBatch] = useState(16)
  const [annEpochs, setAnnEpochs] = useState(8)
  const [annLr, setAnnLr] = useState(0.0002)
  const [annBatch, setAnnBatch] = useState(8)
  const [annThresh, setAnnThresh] = useState(0.5)

  const refresh = async () => {
    setLoadError('')
    try {
      const [c, a, t, me] = await Promise.all([
        getClassifyStatus(),
        getAnnotateStatus(),
        getTrainingLogs(120),
        adminMe().catch(() => ({ admin: false })),
      ])
      setClf(c)
      setAnn(a)
      setLogs(Array.isArray(t.lines) ? t.lines : [])
      setIsAdmin(!!me.admin)
      if (a?.threshold != null) setAnnThresh(Number(a.threshold))
    } catch (e: any) {
      setLoadError(e?.response?.data?.detail || e?.message || 'Failed to load metrics')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    refresh()
  }, [])

  useEffect(() => {
    if (!busy) return
    const id = setInterval(() => {
      getTrainingLogs(120)
        .then((t) => setLogs(Array.isArray(t.lines) ? t.lines : []))
        .catch(() => {})
    }, 2000)
    return () => clearInterval(id)
  }, [busy])

  const unlock = async () => {
    setAuthBusy(true)
    setAuthMsg('')
    try {
      await adminLogin(password)
      setPassword('')
      setIsAdmin(true)
      setAuthMsg('Admin unlocked — training controls enabled.')
    } catch (e: any) {
      setAuthMsg(String(e?.response?.data?.detail || e.message || 'Login failed'))
      setIsAdmin(false)
    } finally {
      setAuthBusy(false)
    }
  }

  const lock = async () => {
    await adminLogout()
    setIsAdmin(false)
    setAuthMsg('Admin session locked.')
  }

  const runClf = async () => {
    setBusy('clf')
    setMsg('')
    try {
      const res = await trainClassifier({
        epochs: clfEpochs,
        lr: clfLr,
        batch_size: clfBatch,
      })
      setMsg(`Classifier trained — accuracy ${pct(res.accuracy)}`)
      await refresh()
    } catch (e: any) {
      setMsg(String(e?.response?.data?.detail || e.message || 'Training failed'))
    } finally {
      setBusy(null)
    }
  }

  const runAnn = async () => {
    setBusy('ann')
    setMsg('')
    try {
      const res = await trainAnnotator({
        epochs: annEpochs,
        lr: annLr,
        batch_size: annBatch,
        threshold: annThresh,
      })
      setMsg(`Annotator trained — mAP ${num(res.map)}`)
      await refresh()
    } catch (e: any) {
      setMsg(String(e?.response?.data?.detail || e.message || 'Training failed'))
    } finally {
      setBusy(null)
    }
  }

  const classes: string[] = clf?.class_list || []

  const clfCurve: ChartPoint[] = useMemo(() => {
    const loss = clf?.loss_curve || []
    const acc = clf?.accuracy_curve || []
    if (!Array.isArray(loss) || !loss.length) return []
    return loss.map((v: number, i: number) => ({
      epoch: i + 1,
      loss: Number(v),
      acc: acc[i] != null ? Number(acc[i]) : undefined,
    }))
  }, [clf])

  const annCurve: ChartPoint[] = useMemo(() => {
    const loss = ann?.loss_curve || []
    const mapc = ann?.map_curve || []
    if (!Array.isArray(loss) || !loss.length) return []
    return loss.map((v: number, i: number) => ({
      epoch: i + 1,
      loss: Number(v),
      map: mapc[i] != null ? Number(mapc[i]) : undefined,
    }))
  }, [ann])

  const perClassRows = useMemo(() => {
    const pc = clf?.per_class || {}
    return classes.map((name) => ({
      name,
      precision: pc[name]?.precision ?? 0,
      recall: pc[name]?.recall ?? 0,
      f1: pc[name]?.f1 ?? 0,
      support: pc[name]?.support ?? 0,
    }))
  }, [clf, classes])

  const labelMetricRows = useMemo(() => {
    const labels: string[] = ann?.labels || Object.keys(ann?.f1_per_label || {})
    return labels.map((name) => ({
      name: name.replace(/_/g, ' '),
      precision: Number(ann?.precision_per_label?.[name] ?? 0),
      recall: Number(ann?.recall_per_label?.[name] ?? 0),
      f1: Number(ann?.f1_per_label?.[name] ?? 0),
    }))
  }, [ann])

  const confusion = Array.isArray(clf?.confusion_matrix) ? clf.confusion_matrix : null

  return (
    <div className="space-y-8">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="font-display text-3xl text-sand">Training & Metrics</h1>
          <p className="text-[var(--muted)]">
            View trained weights, curves, and metrics. Configure / retrain requires admin unlock.
          </p>
        </div>
        <div
          className={`rounded-md px-3 py-1.5 font-mono text-xs ${
            isAdmin ? 'bg-moss/30 text-[#b7e4c7]' : 'bg-white/10 text-[var(--muted)]'
          }`}
        >
          {isAdmin ? 'ADMIN UNLOCKED' : 'VIEWER (read-only)'}
        </div>
      </header>

      {loading && <p className="text-sm text-sky">Loading metrics…</p>}
      {loadError && (
        <div className="rounded-lg border border-ember/40 bg-ember/10 px-4 py-3 text-sm">
          {loadError}{' '}
          <button type="button" className="underline" onClick={() => refresh()}>
            Retry
          </button>
        </div>
      )}
      {msg && <p className="rounded-lg border border-sand/30 bg-sand/10 px-4 py-2 text-sm">{msg}</p>}

      {/* Split: classifier (left) | annotator (right) */}
      <div className="grid gap-6 lg:grid-cols-2 lg:items-start">
        {/* LEFT — Classifier */}
        <div className="space-y-5 rounded-xl border border-sky/25 bg-black/20 p-4 sm:p-5">
          <div className="border-b border-sky/20 pb-3">
            <h2 className="font-display text-2xl text-sky">Classifier</h2>
            <p className="text-sm text-[var(--muted)]">Small-object CNN metrics</p>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <MetricCard title="Accuracy" value={pct(clf?.accuracy)} sub={clf?.test_accuracy != null ? `test ${pct(clf.test_accuracy)}` : undefined} />
            <MetricCard title="Precision (macro)" value={pct(clf?.precision_macro)} />
            <MetricCard title="Recall (macro)" value={pct(clf?.recall_macro)} />
            <MetricCard title="F1 (macro)" value={pct(clf?.f1_macro)} />
          </div>

          <WeightsCard
            title="Small Object Classifier"
            modelName="EfficientNet-B3 + CBAM + MLP"
            architecture={[
              'Input: 3×128×128 RGB patch',
              'Backbone: EfficientNet-B3 (timm, ImageNet pretrained, features_only)',
              'Attention: CBAM (channel + spatial, reduction=8)',
              'Head: GAP → Dropout → Linear→SiLU→BN → Dropout → FC(num_classes)',
              'Loss: CrossEntropy + label_smoothing=0.05 · AdamW + CosineAnnealingLR',
            ]}
            trained={!!clf?.trained}
            weights={clf?.weights}
            lastTrained={clf?.last_trained}
            config={clf?.config}
          />

          <section className="rounded-xl border border-[var(--line)] bg-black/20 p-4">
            <h3 className="font-display text-lg text-white">Training curves</h3>
            <p className="mt-1 font-mono text-xs text-sky">
              EfficientNet-B3 + CBAM + MLP · CrossEntropy (label smoothing)
            </p>
            {clfCurve.length ? (
              <div className="mt-4 space-y-5">
                <div>
                  <p className="mb-2 text-xs uppercase text-[var(--muted)]">Training loss</p>
                  <SafeChart height={180}>
                    <LineChart data={clfCurve}>
                      <CartesianGrid stroke="rgba(255,255,255,0.08)" />
                      <XAxis dataKey="epoch" stroke="#8aa0ad" fontSize={11} />
                      <YAxis stroke="#8aa0ad" fontSize={11} />
                      <Tooltip contentStyle={{ background: '#132533', border: '1px solid #333' }} />
                      <Legend />
                      <Line type="monotone" dataKey="loss" name="Loss" stroke="#c45c26" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
                    </LineChart>
                  </SafeChart>
                </div>
                <div>
                  <p className="mb-2 text-xs uppercase text-[var(--muted)]">Validation accuracy</p>
                  <SafeChart height={180}>
                    <LineChart data={clfCurve}>
                      <CartesianGrid stroke="rgba(255,255,255,0.08)" />
                      <XAxis dataKey="epoch" stroke="#8aa0ad" fontSize={11} />
                      <YAxis domain={[0, 1]} stroke="#8aa0ad" fontSize={11} />
                      <Tooltip contentStyle={{ background: '#132533', border: '1px solid #333' }} />
                      <Legend />
                      <Line type="monotone" dataKey="acc" name="Val acc" stroke="#c9a66b" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
                    </LineChart>
                  </SafeChart>
                </div>
              </div>
            ) : (
              <p className="mt-3 text-sm text-[var(--muted)]">No classifier curves yet.</p>
            )}
          </section>

          <section className="rounded-xl border border-[var(--line)] bg-black/20 p-4">
            <h3 className="mb-3 font-display text-lg text-white">Confusion matrix</h3>
            {confusion ? (
              <div className="overflow-auto">
                <table className="font-mono text-xs">
                  <thead>
                    <tr>
                      <th className="px-2 py-1 text-[var(--muted)]">true\pred</th>
                      {classes.map((c) => (
                        <th key={c} className="px-2 py-1 text-[var(--muted)]">
                          {c.slice(0, 4)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {confusion.map((row: number[], i: number) => (
                      <tr key={i}>
                        <td className="px-2 py-1 text-[var(--muted)]">{classes[i] || i}</td>
                        {(Array.isArray(row) ? row : []).map((v: number, j: number) => (
                          <td
                            key={j}
                            className="border border-white/10 px-2 py-1 text-center"
                            style={{
                              background:
                                i === j
                                  ? `rgba(45,106,79,${Math.min(0.85, 0.15 + Number(v) / 40)})`
                                  : Number(v) > 0
                                    ? 'rgba(196,92,38,0.25)'
                                    : undefined,
                            }}
                          >
                            {v}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-sm text-[var(--muted)]">No confusion matrix available.</p>
            )}
          </section>

          <section className="rounded-xl border border-[var(--line)] bg-black/20 p-4">
            <h3 className="mb-3 font-display text-lg text-white">Per-class precision / recall / F1</h3>
            {perClassRows.length ? (
              <>
                <div className="overflow-auto">
                  <table className="min-w-full text-left text-sm">
                    <thead className="text-xs uppercase text-[var(--muted)]">
                      <tr>
                        <th className="px-2 py-1">Class</th>
                        <th className="px-2 py-1">Precision</th>
                        <th className="px-2 py-1">Recall</th>
                        <th className="px-2 py-1">F1</th>
                        <th className="px-2 py-1">Support</th>
                      </tr>
                    </thead>
                    <tbody>
                      {perClassRows.map((r) => (
                        <tr key={r.name} className="border-t border-white/10">
                          <td className="px-2 py-1.5 capitalize">{r.name}</td>
                          <td className="px-2 py-1.5 font-mono text-xs">{pct(r.precision)}</td>
                          <td className="px-2 py-1.5 font-mono text-xs">{pct(r.recall)}</td>
                          <td className="px-2 py-1.5 font-mono text-xs">{pct(r.f1)}</td>
                          <td className="px-2 py-1.5 font-mono text-xs">{r.support}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="mt-4">
                  <SafeChart height={200}>
                    <BarChart data={perClassRows}>
                      <CartesianGrid stroke="rgba(255,255,255,0.08)" />
                      <XAxis dataKey="name" stroke="#8aa0ad" fontSize={10} />
                      <YAxis domain={[0, 1]} stroke="#8aa0ad" fontSize={11} />
                      <Tooltip contentStyle={{ background: '#132533', border: '1px solid #333' }} />
                      <Legend />
                      <Bar dataKey="precision" fill="#4a90a4" isAnimationActive={false} />
                      <Bar dataKey="recall" fill="#c9a66b" isAnimationActive={false} />
                      <Bar dataKey="f1" fill="#2d6a4f" isAnimationActive={false} />
                    </BarChart>
                  </SafeChart>
                </div>
              </>
            ) : (
              <p className="text-sm text-[var(--muted)]">No per-class metrics yet.</p>
            )}
          </section>
        </div>

        {/* RIGHT — Annotator */}
        <div className="space-y-5 rounded-xl border border-moss/25 bg-black/20 p-4 sm:p-5">
          <div className="border-b border-moss/20 pb-3">
            <h2 className="font-display text-2xl text-moss">Annotator</h2>
            <p className="text-sm text-[var(--muted)]">Satellite multi-label metrics</p>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <MetricCard
              title="Accuracy"
              value={pct(ann?.accuracy ?? (ann?.hamming_loss != null ? 1 - Number(ann.hamming_loss) : null))}
              sub={ann?.subset_f1 != null ? `active-label F1 ${pct(ann.subset_f1)}` : undefined}
            />
            <MetricCard title="mAP" value={num(ann?.map)} sub={ann?.test_map != null ? `test ${num(ann.test_map)}` : undefined} />
            <MetricCard title="Precision" value={pct(ann?.subset_precision ?? ann?.precision_macro)} />
            <MetricCard title="Recall" value={pct(ann?.subset_recall ?? ann?.recall_macro)} />
          </div>

          <WeightsCard
            title="Satellite Multi-Label Annotator"
            modelName="EfficientNet-B3 + CBAM + MLP (multi-label)"
            architecture={[
              'Input: 3×224×224 RGB scene',
              'Backbone: EfficientNet-B3 (timm, ImageNet pretrained)',
              'Attention: CBAM on last feature map',
              'Head: GAP → MLP (Linear→SiLU→BN) → FC(num_labels) → Sigmoid',
              'Loss: BCEWithLogits + pos_weight balancing · AdamW + CosineAnnealingLR',
            ]}
            trained={!!ann?.trained}
            weights={ann?.weights}
            lastTrained={ann?.last_trained}
            config={ann?.config}
          />

          <section className="rounded-xl border border-[var(--line)] bg-black/20 p-4">
            <h3 className="font-display text-lg text-white">Training curves</h3>
            <p className="mt-1 font-mono text-xs text-moss">
              EfficientNet-B3 + CBAM + MLP · BCEWithLogits + pos_weight
            </p>
            {annCurve.length ? (
              <div className="mt-4 space-y-5">
                <div>
                  <p className="mb-2 text-xs uppercase text-[var(--muted)]">
                    Training loss · {annCurve.length} epochs · start {num(annCurve[0]?.loss)} → end{' '}
                    {num(annCurve[annCurve.length - 1]?.loss)}
                  </p>
                  <SafeChart height={180}>
                    <LineChart data={annCurve}>
                      <CartesianGrid stroke="rgba(255,255,255,0.08)" />
                      <XAxis dataKey="epoch" stroke="#8aa0ad" fontSize={11} />
                      <YAxis stroke="#8aa0ad" fontSize={11} />
                      <Tooltip contentStyle={{ background: '#132533', border: '1px solid #333' }} />
                      <Legend />
                      <Line type="monotone" dataKey="loss" name="Loss" stroke="#c45c26" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
                    </LineChart>
                  </SafeChart>
                </div>
                <div>
                  <p className="mb-2 text-xs uppercase text-[var(--muted)]">
                    Validation mAP · start {num(annCurve[0]?.map)} → end{' '}
                    {num(annCurve[annCurve.length - 1]?.map)}
                  </p>
                  <SafeChart height={180}>
                    <LineChart data={annCurve}>
                      <CartesianGrid stroke="rgba(255,255,255,0.08)" />
                      <XAxis dataKey="epoch" stroke="#8aa0ad" fontSize={11} />
                      <YAxis domain={[0, 1]} stroke="#8aa0ad" fontSize={11} />
                      <Tooltip contentStyle={{ background: '#132533', border: '1px solid #333' }} />
                      <Legend />
                      <Line type="monotone" dataKey="map" name="Val mAP" stroke="#4a90a4" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
                    </LineChart>
                  </SafeChart>
                </div>
                <div>
                  <p className="mb-2 text-xs uppercase text-[var(--muted)]">Combined — loss vs mAP</p>
                  <SafeChart height={200}>
                    <LineChart data={annCurve}>
                      <CartesianGrid stroke="rgba(255,255,255,0.08)" />
                      <XAxis dataKey="epoch" stroke="#8aa0ad" fontSize={11} />
                      <YAxis yAxisId="left" stroke="#c45c26" fontSize={11} />
                      <YAxis yAxisId="right" orientation="right" domain={[0, 1]} stroke="#4a90a4" fontSize={11} />
                      <Tooltip contentStyle={{ background: '#132533', border: '1px solid #333' }} />
                      <Legend />
                      <Line yAxisId="left" type="monotone" dataKey="loss" name="Loss" stroke="#c45c26" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
                      <Line yAxisId="right" type="monotone" dataKey="map" name="Val mAP" stroke="#4a90a4" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
                    </LineChart>
                  </SafeChart>
                </div>
              </div>
            ) : (
              <p className="mt-3 text-sm text-[var(--muted)]">No annotator curves yet.</p>
            )}
          </section>

          <section className="rounded-xl border border-[var(--line)] bg-black/20 p-4">
            <h3 className="mb-3 font-display text-lg text-white">Per-label precision / recall / F1</h3>
            {labelMetricRows.length ? (
              <div className="space-y-4">
                <div className="max-h-80 overflow-auto">
                  <table className="min-w-full text-left text-sm">
                    <thead className="sticky top-0 bg-[#132533] text-xs uppercase text-[var(--muted)]">
                      <tr>
                        <th className="px-2 py-1">Label</th>
                        <th className="px-2 py-1">Precision</th>
                        <th className="px-2 py-1">Recall</th>
                        <th className="px-2 py-1">F1</th>
                      </tr>
                    </thead>
                    <tbody>
                      {labelMetricRows.map((r) => (
                        <tr key={r.name} className="border-t border-white/10">
                          <td className="px-2 py-1.5">{r.name}</td>
                          <td className="px-2 py-1.5 font-mono text-xs">{pct(r.precision)}</td>
                          <td className="px-2 py-1.5 font-mono text-xs">{pct(r.recall)}</td>
                          <td className="px-2 py-1.5 font-mono text-xs">{pct(r.f1)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <SafeChart height={260}>
                  <BarChart data={labelMetricRows} margin={{ bottom: 50 }}>
                    <CartesianGrid stroke="rgba(255,255,255,0.08)" />
                    <XAxis dataKey="name" stroke="#8aa0ad" fontSize={8} angle={-35} textAnchor="end" interval={0} height={60} />
                    <YAxis domain={[0, 1]} stroke="#8aa0ad" fontSize={11} />
                    <Tooltip contentStyle={{ background: '#132533', border: '1px solid #333' }} />
                    <Legend />
                    <Bar dataKey="f1" name="F1" fill="#2d6a4f" isAnimationActive={false} />
                  </BarChart>
                </SafeChart>
              </div>
            ) : (
              <p className="text-sm text-[var(--muted)]">No multi-label metrics yet.</p>
            )}
          </section>
        </div>
      </div>

      {/* Admin lock panel */}
      <section className="rounded-xl border border-sand/30 bg-black/30 p-5">
        <h2 className="font-display text-xl text-sand">Admin — configure & train</h2>
        <p className="mt-1 text-sm text-[var(--muted)]">
          Training and dataset write APIs are locked. Default password:{' '}
          <span className="font-mono text-sand">admin123</span> (change in{' '}
          <span className="font-mono">backend/data/admin_config.json</span>).
        </p>

        {!isAdmin ? (
          <div className="mt-4 flex flex-wrap items-end gap-3">
            <label className="block text-sm">
              <span>Admin password</span>
              <div className="relative mt-1">
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && unlock()}
                  className="block w-56 rounded-md border border-white/10 bg-black/40 py-2 pl-3 pr-10 font-mono text-sm"
                  placeholder="••••••••"
                  autoComplete="current-password"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((v) => !v)}
                  className="absolute inset-y-0 right-0 flex items-center px-3 text-[var(--muted)] hover:text-sand"
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                  title={showPassword ? 'Hide password' : 'Show password'}
                >
                  {showPassword ? (
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" />
                      <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" />
                      <line x1="1" y1="1" x2="23" y2="23" />
                    </svg>
                  ) : (
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                      <circle cx="12" cy="12" r="3" />
                    </svg>
                  )}
                </button>
              </div>
            </label>
            <button
              type="button"
              disabled={authBusy || !password}
              onClick={unlock}
              className="rounded-md bg-sand px-4 py-2 text-sm font-medium text-ink disabled:opacity-40"
            >
              {authBusy ? 'Checking…' : 'Unlock training'}
            </button>
          </div>
        ) : (
          <div className="mt-4 space-y-6">
            <div className="flex flex-wrap items-center gap-3">
              <button
                type="button"
                onClick={lock}
                className="rounded-md border border-ember/40 px-3 py-1.5 text-sm text-ember"
              >
                Lock admin session
              </button>
              {busy && (
                <span className="animate-pulse text-sm text-sky">
                  Training {busy === 'clf' ? 'classifier' : 'annotator'}…
                </span>
              )}
            </div>

            <div className="grid gap-6 lg:grid-cols-2">
              <div className="rounded-lg border border-white/10 bg-black/20 p-4">
                <h3 className="font-display text-lg text-white">Train classifier</h3>
                <div className="mt-3 space-y-3">
                  <Slider label="Epochs" value={clfEpochs} min={1} max={30} onChange={setClfEpochs} />
                  <Num label="Learning rate" value={clfLr} onChange={setClfLr} step={0.0001} />
                  <Select label="Batch size" value={clfBatch} options={[4, 8, 16, 32]} onChange={setClfBatch} />
                  <button
                    type="button"
                    disabled={!!busy}
                    onClick={runClf}
                    className="w-full rounded-md bg-sky px-4 py-2.5 text-sm font-medium text-ink disabled:opacity-40"
                  >
                    {busy === 'clf' ? 'Training…' : 'Start classifier training'}
                  </button>
                </div>
              </div>
              <div className="rounded-lg border border-white/10 bg-black/20 p-4">
                <h3 className="font-display text-lg text-white">Train annotator</h3>
                <div className="mt-3 space-y-3">
                  <Slider label="Epochs" value={annEpochs} min={1} max={30} onChange={setAnnEpochs} />
                  <Num label="Learning rate" value={annLr} onChange={setAnnLr} step={0.0001} />
                  <Select label="Batch size" value={annBatch} options={[2, 4, 8, 16]} onChange={setAnnBatch} />
                  <Slider label="Threshold" value={annThresh} min={0.1} max={0.9} step={0.05} onChange={setAnnThresh} />
                  <button
                    type="button"
                    disabled={!!busy}
                    onClick={runAnn}
                    className="w-full rounded-md bg-moss px-4 py-2.5 text-sm font-medium text-white disabled:opacity-40"
                  >
                    {busy === 'ann' ? 'Training…' : 'Start annotator training'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
        {authMsg && <p className="mt-3 text-sm text-[var(--muted)]">{authMsg}</p>}
      </section>

      <section className="rounded-xl border border-[var(--line)] bg-black/30 p-4">
        <div className="mb-2 flex items-center justify-between">
          <h2 className="font-display text-lg text-white">Training log</h2>
          <button type="button" onClick={() => refresh()} className="text-xs text-sky hover:underline">
            Refresh
          </button>
        </div>
        <pre className="max-h-72 overflow-auto whitespace-pre-wrap font-mono text-xs leading-relaxed text-[var(--muted)]">
          {logs.length ? logs.join('\n') : 'No log lines yet.'}
        </pre>
      </section>
    </div>
  )
}

function MetricCard({ title, value, sub }: { title: string; value: string; sub?: string }) {
  return (
    <div className="rounded-xl border border-[var(--line)] bg-[#132533]/80 px-4 py-3">
      <p className="text-xs text-[var(--muted)]">{title}</p>
      <p className="font-display text-2xl text-sand">{value}</p>
      {sub && <p className="font-mono text-xs text-[var(--muted)]">{sub}</p>}
    </div>
  )
}

function WeightsCard({
  title,
  modelName,
  architecture,
  trained,
  weights,
  lastTrained,
  config,
}: {
  title: string
  modelName: string
  architecture: string[]
  trained: boolean
  weights?: { exists?: boolean; file?: string; mb?: number; bytes?: number }
  lastTrained?: string
  config?: Record<string, unknown>
}) {
  return (
    <div className="rounded-xl border border-[var(--line)] bg-black/20 p-4">
      <div className="flex items-center justify-between gap-2">
        <h3 className="font-display text-lg text-white">{title}</h3>
        <span className={`rounded px-2 py-0.5 font-mono text-xs ${trained ? 'bg-moss/30 text-[#b7e4c7]' : 'bg-ember/20 text-[#ffd6a5]'}`}>
          {trained ? 'loaded' : 'missing'}
        </span>
      </div>
      <p className="mt-2 font-display text-base text-sand">{modelName}</p>
      <ul className="mt-2 space-y-1 text-xs leading-relaxed text-[var(--muted)]">
        {architecture.map((line) => (
          <li key={line} className="flex gap-2">
            <span className="text-sky">▸</span>
            <span>{line}</span>
          </li>
        ))}
      </ul>
      <dl className="mt-4 space-y-1 border-t border-white/10 pt-3 font-mono text-xs text-[var(--muted)]">
        <div className="flex justify-between gap-3">
          <dt>file</dt>
          <dd className="text-white/80">{weights?.file || '—'}</dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt>size</dt>
          <dd className="text-white/80">{weights?.mb != null ? `${weights.mb} MB` : '—'}</dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt>last trained</dt>
          <dd className="text-white/80">
            {lastTrained ? new Date(lastTrained).toLocaleString() : '—'}
          </dd>
        </div>
        {config && (
          <div className="flex justify-between gap-3">
            <dt>config</dt>
            <dd className="max-w-[60%] truncate text-right text-white/80">
              {JSON.stringify(config)}
            </dd>
          </div>
        )}
      </dl>
    </div>
  )
}

function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (v: number) => void
}) {
  return (
    <label className="block text-sm">
      <span className="flex justify-between">
        <span>{label}</span>
        <span className="font-mono text-[var(--muted)]">{value}</span>
      </span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} className="mt-1 w-full accent-sky" />
    </label>
  )
}

function Num({
  label,
  value,
  onChange,
  step,
}: {
  label: string
  value: number
  onChange: (v: number) => void
  step: number
}) {
  return (
    <label className="block text-sm">
      <span>{label}</span>
      <input
        type="number"
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-1 w-full rounded-md border border-white/10 bg-black/30 px-3 py-2 font-mono text-sm"
      />
    </label>
  )
}

function Select({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: number
  options: number[]
  onChange: (v: number) => void
}) {
  return (
    <label className="block text-sm">
      <span>{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-1 w-full rounded-md border border-white/10 bg-black/30 px-3 py-2 text-sm"
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  )
}
