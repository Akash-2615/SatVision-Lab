import { Component, type ReactNode } from 'react'
import { Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import ClassifyPredict from './pages/ClassifyPredict'
import AnnotatePredict from './pages/AnnotatePredict'
import PipelineAnalyze from './pages/PipelineAnalyze'
import DatasetManager from './pages/DatasetManager'
import Training from './pages/Training'
import Logs from './pages/Logs'

class PageErrorBoundary extends Component<
  { children: ReactNode },
  { error: string | null }
> {
  state = { error: null as string | null }
  static getDerivedStateFromError(err: Error) {
    return { error: err.message || 'Page crashed' }
  }
  render() {
    if (this.state.error) {
      return (
        <div className="rounded-xl border border-ember/40 bg-ember/10 p-6">
          <h2 className="font-display text-xl text-sand">Something went wrong</h2>
          <p className="mt-2 font-mono text-sm text-[var(--muted)]">{this.state.error}</p>
          <button
            type="button"
            className="mt-4 rounded-md bg-sky px-3 py-2 text-sm text-ink"
            onClick={() => this.setState({ error: null })}
          >
            Try again
          </button>
        </div>
      )
    }
    return this.props.children
  }
}

export default function App() {
  return (
    <Layout>
      <PageErrorBoundary>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/classify" element={<ClassifyPredict />} />
          <Route path="/annotate" element={<AnnotatePredict />} />
          <Route path="/pipeline" element={<PipelineAnalyze />} />
          <Route path="/dataset" element={<DatasetManager />} />
          <Route path="/training" element={<Training />} />
          <Route path="/logs" element={<Logs />} />
        </Routes>
      </PageErrorBoundary>
    </Layout>
  )
}
