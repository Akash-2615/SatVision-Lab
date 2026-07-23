import { useCallback, useRef, useState } from 'react'

type Props = {
  onFile: (file: File) => void
  accept?: string
  label?: string
}

export default function ImageUploader({ onFile, accept = 'image/*', label }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [name, setName] = useState('')
  const [dragging, setDragging] = useState(false)

  const handle = useCallback(
    (file: File | undefined) => {
      if (!file) return
      setName(file.name)
      const url = URL.createObjectURL(file)
      setPreview(url)
      onFile(file)
    },
    [onFile],
  )

  return (
    <div
      className={`relative overflow-hidden rounded-xl border border-dashed transition-colors ${
        dragging ? 'border-sky bg-sky/10' : 'border-[var(--line)] bg-black/20'
      }`}
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        handle(e.dataTransfer.files?.[0])
      }}
    >
      <button
        type="button"
        className="flex w-full flex-col items-center justify-center gap-3 px-6 py-10 text-left"
        onClick={() => inputRef.current?.click()}
      >
        {preview ? (
          <img src={preview} alt="preview" className="max-h-64 w-full object-contain" />
        ) : (
          <>
            <div className="font-display text-lg text-sand">{label || 'Drop image or click to upload'}</div>
            <p className="text-sm text-[var(--muted)]">JPG, PNG, WEBP — local only</p>
          </>
        )}
        {name && <p className="font-mono text-xs text-[var(--muted)]">{name}</p>}
      </button>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => handle(e.target.files?.[0])}
      />
    </div>
  )
}
