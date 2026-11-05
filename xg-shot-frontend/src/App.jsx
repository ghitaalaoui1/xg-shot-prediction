import { useMemo, useState } from 'react'
import { ChevronRight, CircleHelp, Sparkles } from 'lucide-react'
import Sidebar from './components/Sidebar.jsx'
import Pitch from './components/Pitch.jsx'
import ShotControls from './components/ShotControls.jsx'
import XgResult from './components/XgResult.jsx'
import MethodNotes from './pages/MethodNotes.jsx'
import { predictShot } from './services/api.js'

const initialPoint = { x: 105, y: 40 }
const initialControls = { defenders: 1, keeperOff: 0.8, keeperOffset: 0.6, bodyPart: 'Right Foot', technique: 'Normal', playPattern: 'Regular Play', underPressure: false, firstTime: false }

function getGeometry(point) {
  const distance = Math.hypot(120 - point.x, 40 - point.y)
  const angle = Math.abs(Math.atan2(36 - point.y, 120 - point.x) - Math.atan2(44 - point.y, 120 - point.x))
  return { distance, angle }
}

export default function App() {
  const [page, setPage] = useState('studio')
  const [point, setPoint] = useState(initialPoint)
  const [controls, setControls] = useState(initialControls)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const geometry = useMemo(() => getGeometry(point), [point])

  const calculate = async () => {
    setLoading(true)
    setError('')
    try {
      const response = await predictShot({ ...controls, ...geometry })
      setResult(response.probabilite_de_but)
    } catch (err) {
      setError(`${err.message}. Vérifie que FastAPI fonctionne sur http://127.0.0.1:8000.`)
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  return <div className="app-shell">
    <Sidebar active={page} onChange={setPage}/>
    <main className="main-area">
      <header className="topbar"><div className="breadcrumb"><span>WORKSPACE</span><ChevronRight size={13}/><b>{page === 'studio' ? 'Shot studio' : 'Model notes'}</b></div><div className="top-actions"><div className="connection"><i/> API CONNECTION</div><button className="icon-button" aria-label="Help"><CircleHelp size={17}/></button><div className="avatar">GA</div></div></header>
      {page === 'studio' ? <div className="dashboard-content">
        <section className="welcome-row"><div><div className="page-kicker"><Sparkles size={13}/> EXPECTED GOALS WORKSPACE</div><h1>Read the moment.<br/><em>Measure the chance.</em></h1><p>Build a shot scenario and let your model estimate its probability of becoming a goal.</p></div><div className="welcome-meta"><span>MODEL VERSION</span><b>LOGISTIC · V1</b><small>StatsBomb feature pipeline</small></div></section>
        <div className="section-heading"><div><span className="section-index">01</span><h2>Shot configuration</h2></div><span className="section-caption">PLACE YOUR SHOT & DEFINE THE CONTEXT</span></div>
        <div className="workbench-grid">
          <div className="left-stack"><Pitch point={point} onSelect={(p) => { setPoint(p); setResult(null); setError('') }}/><div className="position-strip"><div><span>SHOT X</span><b>{point.x.toFixed(1)}</b></div><div><span>SHOT Y</span><b>{point.y.toFixed(1)}</b></div><div><span>DISTANCE TO GOAL</span><b>{geometry.distance.toFixed(1)} yd</b></div><div><span>SHOT ANGLE</span><b>{(geometry.angle * 180 / Math.PI).toFixed(1)}°</b></div></div></div>
          <ShotControls values={controls} onChange={(updater) => { setControls(updater); setResult(null) }}/>
        </div>
        <div className="section-heading result-section-heading"><div><span className="section-index">02</span><h2>Prediction result</h2></div><span className="section-caption">INFERENCE FROM YOUR BACKEND</span></div>
        <XgResult result={result} loading={loading} error={error} onPredict={calculate} geometry={geometry}/>
        <footer className="page-footer"><span>PitchIQ <b>·</b> XG SHOT PREDICTION</span><span>Built around your existing FastAPI model</span></footer>
      </div> : <MethodNotes onBack={() => setPage('studio')}/>}
    </main>
  </div>
}
