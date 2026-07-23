import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  adminMe,
  deleteClass,
  deleteSatellite,
  getDatasetClasses,
  getSatelliteDataset,
  uploadClassImages,
  uploadSatellite,
} from '../api/client'
import LabelChip from '../components/LabelChip'

export default function DatasetManager() {
  const [tab, setTab] = useState<'objects' | 'satellite'>('objects')
  const [classes, setClasses] = useState<any[]>([])
  const [sats, setSats] = useState<any[]>([])
  const [labels, setLabels] = useState<string[]>([])
  const [newClass, setNewClass] = useState('')
  const [selectedLabels, setSelectedLabels] = useState<string[]>([])
  const [filterLabel, setFilterLabel] = useState<string | null>(null)
  const [msg, setMsg] = useState('')
  const [busy, setBusy] = useState(false)
  const [isAdmin, setIsAdmin] = useState(false)

  const refresh = async () => {
    const [c, s, me] = await Promise.all([
      getDatasetClasses(),
      getSatelliteDataset(),
      adminMe().catch(() => ({ admin: false })),
    ])
    setClasses(c.classes || [])
    setSats(s.images || [])
    setLabels(s.available_labels || [])
    setIsAdmin(!!me.admin)
  }

  useEffect(() => {
    refresh().catch(console.error)
  }, [])

  const labelCounts = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const l of labels) counts[l] = 0
    for (const s of sats) {
      for (const l of s.labels || []) {
        counts[l] = (counts[l] || 0) + 1
      }
    }
    return counts
  }, [sats, labels])

  const filteredSats = useMemo(() => {
    if (!filterLabel) return sats
    return sats.filter((s) => (s.labels || []).includes(filterLabel))
  }, [sats, filterLabel])

  const uploadToClass = async (className: string, files: FileList | null) => {
    if (!files?.length) return
    if (!isAdmin) {
      setMsg('Admin unlock required on Training page to modify datasets.')
      return
    }
    setBusy(true)
    setMsg('')
    try {
      await uploadClassImages(className, Array.from(files))
      setMsg(`Uploaded ${files.length} image(s) to ${className}`)
      await refresh()
    } catch (e: any) {
      setMsg(e?.response?.data?.detail || e.message)
    } finally {
      setBusy(false)
    }
  }

  const addClass = async (files: FileList | null) => {
    if (!newClass.trim() || !files?.length) return
    await uploadToClass(newClass.trim(), files)
    setNewClass('')
  }

  const removeClass = async (name: string) => {
    if (!isAdmin) {
      setMsg('Admin unlock required on Training page to modify datasets.')
      return
    }
    if (!confirm(`Delete class "${name}" and all images?`)) return
    try {
      await deleteClass(name)
      await refresh()
    } catch (e: any) {
      setMsg(e?.response?.data?.detail || e.message)
    }
  }

  const addSat = async (files: FileList | null) => {
    if (!files?.length) return
    if (!isAdmin) {
      setMsg('Admin unlock required on Training page to modify datasets.')
      return
    }
    setBusy(true)
    try {
      for (const f of Array.from(files)) {
        await uploadSatellite(f, selectedLabels)
      }
      setMsg(`Uploaded ${files.length} satellite image(s)`)
      await refresh()
    } catch (e: any) {
      setMsg(e?.response?.data?.detail || e.message)
    } finally {
      setBusy(false)
    }
  }

  const removeSat = async (filename: string) => {
    if (!isAdmin) {
      setMsg('Admin unlock required on Training page to modify datasets.')
      return
    }
    if (!confirm(`Delete ${filename}?`)) return
    try {
      await deleteSatellite(filename)
      await refresh()
    } catch (e: any) {
      setMsg(e?.response?.data?.detail || e.message)
    }
  }

  const toggleLabel = (name: string) => {
    setSelectedLabels((prev) =>
      prev.includes(name) ? prev.filter((x) => x !== name) : [...prev, name],
    )
  }

  return (
    <div className="space-y-6">
      <header>
        <h1 className="font-display text-3xl text-sand">Dataset Manager</h1>
        <p className="text-[var(--muted)]">Local folders under backend/data/datasets/</p>
      </header>

      {!isAdmin && (
        <div className="rounded-lg border border-sand/30 bg-sand/10 px-4 py-3 text-sm">
          Dataset uploads/deletes are admin-only. Unlock on the{' '}
          <Link to="/training" className="text-sand underline">
            Training
          </Link>{' '}
          page first. Viewing remains open.
        </div>
      )}

      {msg && <p className="text-sm text-sand">{msg}</p>}

      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => setTab('objects')}
          className={`rounded-md px-4 py-2 text-sm ${tab === 'objects' ? 'bg-sky/30 text-white' : 'bg-white/5 text-[var(--muted)]'}`}
        >
          Small Object Classes
        </button>
        <button
          type="button"
          onClick={() => setTab('satellite')}
          className={`rounded-md px-4 py-2 text-sm ${tab === 'satellite' ? 'bg-moss/30 text-white' : 'bg-white/5 text-[var(--muted)]'}`}
        >
          Satellite Images
        </button>
      </div>

      {tab === 'objects' ? (
        <div className="space-y-5">
          <div className="flex flex-wrap items-end gap-3 rounded-xl border border-[var(--line)] bg-black/20 p-4">
            <div>
              <label className="text-xs text-[var(--muted)]">New class name</label>
              <input
                value={newClass}
                onChange={(e) => setNewClass(e.target.value)}
                className="mt-1 block rounded-md border border-white/10 bg-black/30 px-3 py-2 text-sm"
                placeholder="e.g. bridge"
              />
            </div>
            <label className="cursor-pointer rounded-md bg-sky px-4 py-2 text-sm text-ink">
              {busy ? 'Uploading…' : 'Add class + images'}
              <input
                type="file"
                accept="image/*"
                multiple
                className="hidden"
                disabled={busy}
                onChange={(e) => addClass(e.target.files)}
              />
            </label>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {classes.map((c) => (
              <div key={c.name} className="rounded-xl border border-[var(--line)] bg-[#132533]/70 p-4">
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="font-display text-lg capitalize text-white">{c.name}</h3>
                    <p className="font-mono text-xs text-[var(--muted)]">{c.count} images</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => removeClass(c.name)}
                    className="text-xs text-ember hover:underline"
                  >
                    Delete
                  </button>
                </div>
                <div className="mt-3 grid grid-cols-4 gap-1">
                  {(c.sample_paths || []).slice(0, 4).map((p: string, i: number) => {
                    const parts = p.split(/[/\\]/)
                    const cls = parts[parts.length - 2]
                    const file = parts[parts.length - 1]
                    return (
                      <img
                        key={i}
                        src={`/files/small_objects/${cls}/${file}`}
                        alt=""
                        className="aspect-square rounded object-cover"
                      />
                    )
                  })}
                </div>
                <label className="mt-3 inline-block cursor-pointer rounded border border-white/15 px-3 py-1.5 text-xs">
                  Upload more
                  <input
                    type="file"
                    accept="image/*"
                    multiple
                    className="hidden"
                    onChange={(e) => uploadToClass(c.name, e.target.files)}
                  />
                </label>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="space-y-5">
          <div className="rounded-xl border border-[var(--line)] bg-black/20 p-4">
            <div className="mb-1 flex flex-wrap items-baseline justify-between gap-2">
              <p className="text-sm text-white">Filter images by label</p>
              <p className="font-mono text-xs text-[var(--muted)]">
                Showing {filteredSats.length} / {sats.length}
                {filterLabel ? ` · ${filterLabel.replace(/_/g, ' ')}` : ' · all'}
              </p>
            </div>
            <p className="mb-3 text-xs text-[var(--muted)]">
              Click a label to show only matching images. Click again (or All) to clear.
            </p>
            <div className="flex flex-wrap gap-2">
              <LabelChip
                name="all"
                active={filterLabel === null}
                count={sats.length}
                onClick={() => setFilterLabel(null)}
              />
              {labels.map((l) => (
                <LabelChip
                  key={l}
                  name={l}
                  active={filterLabel === l}
                  count={labelCounts[l] || 0}
                  onClick={() => setFilterLabel((prev) => (prev === l ? null : l))}
                />
              ))}
            </div>
          </div>

          <div className="rounded-xl border border-[var(--line)] bg-black/20 p-4">
            <p className="mb-2 text-sm text-[var(--muted)]">Select labels for new uploads</p>
            <div className="mb-4 flex flex-wrap gap-2">
              {labels.map((l) => (
                <LabelChip
                  key={l}
                  name={l}
                  active={selectedLabels.includes(l)}
                  onClick={() => toggleLabel(l)}
                />
              ))}
            </div>
            <label className="inline-block cursor-pointer rounded-md bg-moss px-4 py-2 text-sm text-white">
              {busy ? 'Uploading…' : 'Upload satellite image(s)'}
              <input
                type="file"
                accept="image/*"
                multiple
                className="hidden"
                disabled={busy}
                onChange={(e) => addSat(e.target.files)}
              />
            </label>
          </div>

          <ul className="divide-y divide-white/10 overflow-hidden rounded-xl border border-[var(--line)]">
            {filteredSats.length === 0 ? (
              <li className="bg-black/20 px-4 py-8 text-center text-sm text-[var(--muted)]">
                {filterLabel
                  ? `No images labeled “${filterLabel.replace(/_/g, ' ')}”.`
                  : 'No satellite images in the dataset yet.'}
              </li>
            ) : (
              filteredSats.map((s) => (
                <li key={s.filename} className="flex flex-wrap items-center gap-4 bg-black/20 px-4 py-3">
                  <img
                    src={`/files/satellite/${s.filename}`}
                    alt=""
                    className="h-16 w-16 rounded object-cover"
                  />
                  <div className="min-w-0 flex-1">
                    <p className="truncate font-mono text-sm">{s.filename}</p>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {(s.labels || []).map((l: string) => (
                        <LabelChip
                          key={l}
                          name={l}
                          active={filterLabel === l}
                          onClick={() => setFilterLabel(l)}
                        />
                      ))}
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => removeSat(s.filename)}
                    className="text-xs text-ember hover:underline"
                  >
                    Delete
                  </button>
                </li>
              ))
            )}
          </ul>
        </div>
      )}
    </div>
  )
}
