import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { getClassifyStatus, getAnnotateStatus, pipelineAnalyze } from '../api/client'
import ImageUploader from '../components/ImageUploader'
import LabelChip from '../components/LabelChip'
import ConfidenceBar from '../components/ConfidenceBar'

type Patch = {
  patch_id: number
  bbox: [number, number, number, number]
  class: string
  confidence: number
  top3: { class: string; confidence: number }[]
  thumbnail_base64?: string
}

export default function PipelineAnalyze() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState<any>(null)
  const [selected, setSelected] = useState<Patch | null>(null)
  const [ready, setReady] = useState(true)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement | null>(null)

  useEffect(() => {
    Promise.all([getClassifyStatus(), getAnnotateStatus()]).then(([c, a]) => {
      setReady(!!(c.trained || a.trained))
    })
  }, [])

  const onFile = (f: File) => {
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
    setSelected(null)
    setError('')
  }

  const draw = (img: HTMLImageElement, patches: Patch[], active?: Patch | null) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const maxW = 720
    const scale = Math.min(1, maxW / img.naturalWidth)
    canvas.width = img.naturalWidth * scale
    canvas.height = img.naturalHeight * scale
    const ctx = canvas.getContext('2d')!
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
    for (const p of patches) {
      const [x1, y1, x2, y2] = p.bbox
      const isSel = active && active.patch_id === p.patch_id
      ctx.strokeStyle = isSel ? '#c9a66b' : '#4a90a4'
      ctx.lineWidth = isSel ? 3 : 2
      ctx.strokeRect(x1 * scale, y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale)
      const label = `${p.class} ${(p.confidence * 100).toFixed(0)}%`
      ctx.font = '12px IBM Plex Mono, monospace'
      const tw = ctx.measureText(label).width + 8
      ctx.fillStyle = 'rgba(7,16,24,0.85)'
      ctx.fillRect(x1 * scale, Math.max(0, y1 * scale - 18), tw, 16)
      ctx.fillStyle = isSel ? '#c9a66b' : '#e8eef2'
      ctx.fillText(label, x1 * scale + 4, Math.max(12, y1 * scale - 6))
    }
  }

  useEffect(() => {
    if (!preview || !result) return
    const img = new Image()
    img.onload = () => {
      imgRef.current = img
      draw(img, result.patch_results || [], selected)
    }
    img.src = preview
  }, [preview, result, selected])

  const run = async () => {
    if (!file) return
    setLoading(true)
    setError('')
    try {
      const data = await pipelineAnalyze(file, 0.5)
      setResult(data)
      setSelected(null)
    } catch (e: any) {
      setError(e?.response?.data?.detail || e.message || 'Pipeline failed')
    } finally {
      setLoading(false)
    }
  }

  const onCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!result || !imgRef.current) return
    const canvas = canvasRef.current!
    const rect = canvas.getBoundingClientRect()
    const scaleX = imgRef.current.naturalWidth / canvas.width
    const scaleY = imgRef.current.naturalHeight / canvas.height
    const x = ((e.clientX - rect.left) / rect.width) * canvas.width * scaleX
    const y = ((e.clientY - rect.top) / rect.height) * canvas.height * scaleY
    const hit = [...(result.patch_results || [])]
      .reverse()
      .find((p: Patch) => {
        const [x1, y1, x2, y2] = p.bbox
        return x >= x1 && x <= x2 && y >= y1 && y <= y2
      })
    if (hit) setSelected(hit)
  }

  return (
    <div className="space-y-6">
      <header>
        <h1 className="font-display text-3xl text-sand">Unified Pipeline</h1>
        <p className="text-[var(--muted)]">
          Two models: <span className="text-moss">scene labels</span> (agriculture / water / …) then
          optional <span className="text-sky">object boxes</span> (ship / vehicle / aircraft / building)
          on larger images only.
        </p>
      </header>

      {!ready && (
        <div className="rounded-lg border border-ember/40 bg-ember/10 px-4 py-3 text-sm">
          No models trained.{' '}
          <Link to="/training" className="text-sand underline">
            Train first
          </Link>
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
        <div className="space-y-4">
          <ImageUploader onFile={onFile} label="Upload full satellite image" />
          <button
            type="button"
            disabled={!file || loading || !ready}
            onClick={run}
            className="w-full rounded-md bg-ember px-4 py-3 font-medium text-white disabled:opacity-40"
          >
            {loading ? 'Analyzing…' : 'Analyze'}
          </button>
          {error && <p className="text-sm text-ember">{error}</p>}
          {result && (
            <div className="overflow-auto rounded-xl border border-[var(--line)] bg-black/30 p-2">
              <canvas ref={canvasRef} className="mx-auto max-w-full cursor-crosshair" onClick={onCanvasClick} />
            </div>
          )}
        </div>

        <aside className="space-y-4 rounded-xl border border-[var(--line)] bg-black/20 p-4">
          <div>
            <h2 className="font-display text-lg text-moss">Scene labels</h2>
            <p className="mb-2 text-xs text-[var(--muted)]">
              Multi-label annotator — land cover for the whole image (not clickable objects).
            </p>
            {!result && <p className="text-sm text-[var(--muted)]">Run analysis to see labels.</p>}
            {result && (
              <div className="space-y-3">
                <div className="rounded-lg border border-moss/35 bg-moss/10 p-3">
                  <p className="mb-2 text-xs font-medium text-[#b7e4c7]">
                    Annotations
                    <span className="ml-2 font-mono font-normal text-[var(--muted)]">
                      ≥ threshold ·{' '}
                      {(result.scene_labels || []).filter((l: any) => l.above_threshold).length}
                    </span>
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {(result.scene_labels || [])
                      .filter((l: any) => l.above_threshold)
                      .sort((a: any, b: any) => Number(b.score) - Number(a.score))
                      .map((l: any) => (
                        <LabelChip key={l.name} name={l.name} active score={l.score} />
                      ))}
                    {!(result.scene_labels || []).some((l: any) => l.above_threshold) && (
                      <p className="text-sm text-[var(--muted)]">No labels above threshold.</p>
                    )}
                  </div>
                </div>
                <div className="rounded-lg border border-sand/30 bg-sand/5 p-3">
                  <p className="mb-2 text-xs font-medium text-sand">
                    Below threshold
                    <span className="ml-2 font-mono font-normal text-[var(--muted)]">
                      {(result.scene_labels || []).filter((l: any) => !l.above_threshold).length}
                    </span>
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {(result.scene_labels || [])
                      .filter((l: any) => !l.above_threshold)
                      .sort((a: any, b: any) => Number(b.score) - Number(a.score))
                      .slice(0, 8)
                      .map((l: any) => (
                        <LabelChip key={l.name} name={l.name} score={l.score} />
                      ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="border-t border-[var(--line)] pt-4">
            <h2 className="font-display text-lg text-sky">Object patches</h2>
            <p className="mb-2 text-xs text-[var(--muted)]">
              Separate classifier: only aircraft / ship / vehicle / building. Click a box on the image —
              this is not the scene label.
            </p>
            {result?.patch_note && (
              <p className="mb-3 rounded-md border border-sand/30 bg-sand/10 px-3 py-2 text-xs text-sand">
                {result.patch_note}
              </p>
            )}
            {selected ? (
              <div className="space-y-3">
                {selected.thumbnail_base64 && (
                  <img
                    src={`data:image/jpeg;base64,${selected.thumbnail_base64}`}
                    alt="patch"
                    className="w-full rounded-lg"
                  />
                )}
                <p className="font-display text-2xl capitalize text-sand">{selected.class}</p>
                <ConfidenceBar value={selected.confidence} label="Confidence" color="#c9a66b" />
                {(selected.top3 || []).map((t) => (
                  <ConfidenceBar key={t.class} label={t.class} value={t.confidence} />
                ))}
                <p className="font-mono text-xs text-[var(--muted)]">
                  bbox [{selected.bbox.join(', ')}]
                </p>
              </div>
            ) : (
              <p className="text-sm text-[var(--muted)]">
                {result?.patch_results?.length
                  ? 'Click a bounding box on the image.'
                  : 'No object boxes to click — scene labels above are the land-cover result.'}
              </p>
            )}
            {result && (
              <p className="mt-2 font-mono text-xs text-[var(--muted)]">
                {result.patch_results?.length || 0} confident object detections
                {result.image_size
                  ? ` · image ${result.image_size.width}×${result.image_size.height}`
                  : ''}
              </p>
            )}
          </div>
        </aside>
      </div>
    </div>
  )
}
