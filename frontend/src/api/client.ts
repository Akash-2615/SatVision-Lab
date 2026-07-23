import axios from 'axios'

const TOKEN_KEY = 'satvision_admin_token'

const api = axios.create({
  baseURL: '/api',
  timeout: 600000, // training can be long
})

api.interceptors.request.use((config) => {
  const token = localStorage.getItem(TOKEN_KEY)
  if (token) {
    config.headers = config.headers || {}
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

export type Health = {
  status: string
  models_loaded: { classifier: boolean; annotator: boolean }
  classes: string[]
  labels: string[]
  device: string
}

export type ModelMeta = {
  classes: string[]
  labels: string[]
  threshold: number
  classifier: Record<string, unknown>
  annotator: Record<string, unknown>
}

export function getAdminToken() {
  return localStorage.getItem(TOKEN_KEY)
}

export function clearAdminToken() {
  localStorage.removeItem(TOKEN_KEY)
}

export async function adminLogin(password: string) {
  const form = new FormData()
  form.append('password', password)
  const { data } = await api.post('/admin/login', form)
  if (data.token) localStorage.setItem(TOKEN_KEY, data.token)
  return data
}

export async function adminLogout() {
  try {
    await api.post('/admin/logout')
  } finally {
    clearAdminToken()
  }
}

export async function adminMe() {
  const { data } = await api.get('/admin/me')
  return data as { admin: boolean; password_hint: string }
}

export async function getHealth() {
  const { data } = await api.get<Health>('/health')
  return data
}

export async function getMeta() {
  const { data } = await api.get<ModelMeta>('/meta')
  return data
}

export async function classifyPredict(file: File) {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post('/classify/predict', form)
  return data
}

export async function annotatePredict(file: File, threshold = 0.5) {
  const form = new FormData()
  form.append('file', file)
  form.append('threshold', String(threshold))
  const { data } = await api.post('/annotate/predict', form)
  return data
}

export async function pipelineAnalyze(file: File, threshold = 0.5) {
  const form = new FormData()
  form.append('file', file)
  form.append('threshold', String(threshold))
  const { data } = await api.post('/pipeline/analyze', form)
  return data
}

export async function getClassifyStatus() {
  const { data } = await api.get('/classify/status')
  return data
}

export async function getAnnotateStatus() {
  const { data } = await api.get('/annotate/status')
  return data
}

export async function trainClassifier(params: {
  epochs: number
  lr: number
  batch_size: number
}) {
  const form = new FormData()
  form.append('epochs', String(params.epochs))
  form.append('lr', String(params.lr))
  form.append('batch_size', String(params.batch_size))
  const { data } = await api.post('/classify/train', form)
  return data
}

export async function trainAnnotator(params: {
  epochs: number
  lr: number
  batch_size: number
  threshold: number
}) {
  const form = new FormData()
  form.append('epochs', String(params.epochs))
  form.append('lr', String(params.lr))
  form.append('batch_size', String(params.batch_size))
  form.append('threshold', String(params.threshold))
  const { data } = await api.post('/annotate/train', form)
  return data
}

export async function getTrainingLogs(n = 80) {
  const { data } = await api.get('/training/logs', { params: { n } })
  return data
}

export async function getDatasetClasses() {
  const { data } = await api.get('/dataset/classes')
  return data
}

export async function uploadClassImages(className: string, files: File[]) {
  const form = new FormData()
  files.forEach((f) => form.append('files', f))
  const { data } = await api.post(`/dataset/classes/${encodeURIComponent(className)}`, form)
  return data
}

export async function deleteClass(className: string) {
  const { data } = await api.delete(`/dataset/classes/${encodeURIComponent(className)}`)
  return data
}

export async function getSatelliteDataset() {
  const { data } = await api.get('/dataset/satellite')
  return data
}

export async function uploadSatellite(file: File, labels: string[]) {
  const form = new FormData()
  form.append('file', file)
  form.append('labels', JSON.stringify(labels))
  const { data } = await api.post('/dataset/satellite', form)
  return data
}

export async function deleteSatellite(filename: string) {
  const { data } = await api.delete(`/dataset/satellite/${encodeURIComponent(filename)}`)
  return data
}

export async function getLogs() {
  const { data } = await api.get('/logs')
  return data
}

export async function clearLogs() {
  const { data } = await api.delete('/logs')
  return data
}

export function fileUrl(path: string) {
  if (!path) return ''
  if (path.startsWith('http') || path.startsWith('data:')) return path
  if (path.startsWith('/')) return path
  return `/files/${path}`
}

export default api
