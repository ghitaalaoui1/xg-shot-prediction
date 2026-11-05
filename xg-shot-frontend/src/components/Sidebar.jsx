import { Activity, CircleHelp, Crosshair, LayoutDashboard, Settings2, ShieldCheck } from 'lucide-react'

export default function Sidebar({ active, onChange }) {
  const items = [
    { id: 'studio', label: 'Shot studio', icon: Crosshair },
    { id: 'method', label: 'Model notes', icon: Activity },
  ]
  return (
    <aside className="sidebar">
      <div className="brand-lockup">
        <div className="brand-mark"><Activity size={20} strokeWidth={2.5} /></div>
        <div><strong>pitch<span>iq</span></strong><small>FOOTBALL INTELLIGENCE</small></div>
      </div>
      <div className="workspace-label">WORKSPACE</div>
      <nav className="side-nav" aria-label="Main navigation">
        {items.map(({ id, label, icon: Icon }) => (
          <button key={id} className={`nav-item ${active === id ? 'active' : ''}`} onClick={() => onChange(id)}>
            <Icon size={17} /><span>{label}</span>{active === id && <i />}
          </button>
        ))}
      </nav>
      <div className="sidebar-bottom">
        <div className="api-mini"><span className="status-dot" /><div><b>Prediction API</b><small>Connected via local proxy</small></div></div>
        <div className="sidebar-foot"><ShieldCheck size={14} /> Your model. Your data.</div>
      </div>
    </aside>
  )
}
