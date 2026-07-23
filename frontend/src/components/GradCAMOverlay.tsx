type Props = {
  src?: string | null
  overlayBase64?: string | null
  showOverlay?: boolean
  alt?: string
  className?: string
}

export default function GradCAMOverlay({
  src,
  overlayBase64,
  showOverlay = true,
  alt = 'result',
  className = '',
}: Props) {
  const overlay = overlayBase64 ? `data:image/png;base64,${overlayBase64}` : null
  return (
    <div className={`relative overflow-hidden rounded-xl bg-black/30 ${className}`}>
      {src && <img src={src} alt={alt} className="block w-full object-contain" />}
      {showOverlay && overlay && (
        <img
          src={overlay}
          alt="gradcam"
          className="pointer-events-none absolute inset-0 h-full w-full object-contain opacity-70 mix-blend-screen"
        />
      )}
    </div>
  )
}
