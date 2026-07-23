import { NavLink } from 'react-router-dom'

const links = [
  { to: '/', label: 'Dashboard' },
  { to: '/classify', label: 'Object Classification' },
  { to: '/annotate', label: 'Scene Annotation' },
  { to: '/pipeline', label: 'Pipeline' },
  { to: '/dataset', label: 'Dataset' },
  { to: '/training', label: 'Metrics' },
  { to: '/logs', label: 'Logs' },
]

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen">
      <header className="border-b border-[var(--line)]/60 bg-black/20 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-5 py-4">
          <div>
            <div className="font-display text-2xl font-extrabold tracking-tight text-sand">
              SatVision Lab
            </div>
            <p className="text-xs text-[var(--muted)]">
              Small-object CNN + satellite multi-label annotation
            </p>
          </div>
          <nav className="flex flex-wrap gap-1">
            {links.map((l) => (
              <NavLink
                key={l.to}
                to={l.to}
                end={l.to === '/'}
                className={({ isActive }) =>
                  `rounded-md px-3 py-1.5 text-sm transition ${
                    isActive
                      ? 'bg-sky/25 text-white'
                      : 'text-[var(--muted)] hover:bg-white/5 hover:text-white'
                  }`
                }
              >
                {l.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-5 py-8">{children}</main>
    </div>
  )
}
