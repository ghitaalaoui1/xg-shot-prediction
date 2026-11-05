import { ChevronDown, SlidersHorizontal } from 'lucide-react'

function RangeControl({ label, value, min, max, step, suffix, onChange }) {
  return <label className="range-control"><span className="control-label">{label}<b>{Number(value).toFixed(step < 1 ? 1 : 0)}{suffix}</b></span><input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} /></label>
}

export default function ShotControls({ values, onChange }) {
  const set = (key, value) => onChange((current) => ({ ...current, [key]: value }))
  return (
    <section className="controls-panel">
      <div className="panel-heading"><div className="panel-icon"><SlidersHorizontal size={16} /></div><div><h3>Shot context</h3><p>Adjust the scenario inputs</p></div><span className="step-tag">02 / 02</span></div>
      <div className="controls-content">
        <RangeControl label="Defenders in shooting lane" value={values.defenders} min={0} max={8} step={1} suffix="" onChange={(v) => set('defenders', v)} />
        <RangeControl label="Goalkeeper off the line" value={values.keeperOff} min={0} max={8} step={0.1} suffix=" yd" onChange={(v) => set('keeperOff', v)} />
        <RangeControl label="Goalkeeper lateral offset" value={values.keeperOffset} min={0} max={6} step={0.1} suffix=" yd" onChange={(v) => set('keeperOffset', v)} />
        <div className="control-divider" />
        <label className="select-control"><span className="control-label">Body part</span><span className="select-wrap"><select value={values.bodyPart} onChange={(e) => set('bodyPart', e.target.value)}><option>Right Foot</option><option>Left Foot</option><option>Other</option></select><ChevronDown size={14} /></span></label>
        <label className="select-control"><span className="control-label">Technique</span><span className="select-wrap"><select value={values.technique} onChange={(e) => set('technique', e.target.value)}><option>Normal</option><option>Diving Header</option><option>Half Volley</option><option>Lob</option><option>Overhead Kick</option><option>Volley</option></select><ChevronDown size={14} /></span></label>
        <label className="select-control"><span className="control-label">Play pattern</span><span className="select-wrap"><select value={values.playPattern} onChange={(e) => set('playPattern', e.target.value)}><option>Regular Play</option><option>From Counter</option><option>From Free Kick</option><option>From Goal Kick</option><option>From Keeper</option><option>From Kick Off</option><option>From Throw In</option><option>Other</option></select><ChevronDown size={14} /></span></label>
        <div className="toggle-row"><div><b>Under pressure</b><small>Defender actively challenging the shot</small></div><button type="button" role="switch" aria-checked={values.underPressure} className={`switch ${values.underPressure ? 'on' : ''}`} onClick={() => set('underPressure', !values.underPressure)}><i /></button></div>
        <div className="toggle-row"><div><b>First-time shot</b><small>Shot taken without controlling the ball</small></div><button type="button" role="switch" aria-checked={values.firstTime} className={`switch ${values.firstTime ? 'on' : ''}`} onClick={() => set('firstTime', !values.firstTime)}><i /></button></div>
      </div>
    </section>
  )
}
