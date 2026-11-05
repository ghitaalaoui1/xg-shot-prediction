import { useRef } from 'react'
import { Crosshair } from 'lucide-react'

const X_MIN = 60.5   // halfway line: the model is only trained on attacking-half shots
const X_MAX = 119.2
const Y_MIN = 0.5
const Y_MAX = 79.5

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v))

export default function Pitch({ point, onSelect }) {
  const svgRef = useRef(null)

  // The SVG draws the full 120 x 80 pitch, so screen fraction maps to 0..120.
  // Shots are then clamped to the attacking half.
  const pointToPitch = (event) => {
    const rect = svgRef.current.getBoundingClientRect()
    return {
      x: clamp(((event.clientX - rect.left) / rect.width) * 120, X_MIN, X_MAX),
      y: clamp(((event.clientY - rect.top) / rect.height) * 80, Y_MIN, Y_MAX),
    }
  }

  const handleDown = (event) => {
    event.currentTarget.setPointerCapture(event.pointerId)
    onSelect(pointToPitch(event))
  }

  const handleMove = (event) => {
    if (event.buttons !== 0) onSelect(pointToPitch(event))
  }

  const handleKey = (event) => {
    const step = event.shiftKey ? 4 : 1
    const moves = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }
    const move = moves[event.key]
    if (!move) return
    event.preventDefault()
    onSelect({ x: clamp(point.x + move[0], X_MIN, X_MAX), y: clamp(point.y + move[1], Y_MIN, Y_MAX) })
  }

  return (
    <div className="pitch-shell">
      <div className="pitch-topline"><span><span className="live-dot" /> LIVE SHOT POSITION</span><span>ATTACKING HALF · STATSBOMB COORDINATES</span></div>
      <svg
        ref={svgRef}
        className="pitch"
        viewBox="0 0 120 80"
        role="application"
        tabIndex={0}
        aria-label="Interactive football pitch. Click or drag in the attacking half to place the shot, or use the arrow keys."
        onPointerDown={handleDown}
        onPointerMove={handleMove}
        onKeyDown={handleKey}
      >
        <defs>
          <pattern id="grass" width="12" height="80" patternUnits="userSpaceOnUse"><rect width="6" height="80" fill="#183d2b" opacity=".19" /></pattern>
          <radialGradient id="pitchGlow" cx=".8" cy=".5" r=".7"><stop offset="0" stopColor="#1d7548" stopOpacity=".22"/><stop offset="1" stopColor="#07150e" stopOpacity="0"/></radialGradient>
        </defs>
        <rect width="120" height="80" fill="#10271b" />
        <rect width="120" height="80" fill="url(#grass)" />
        <rect width="120" height="80" fill="url(#pitchGlow)" />
        <g fill="none" stroke="#b8d5c2" strokeOpacity=".45" strokeWidth=".55">
          <rect x=".6" y=".6" width="118.8" height="78.8" />
          <path d="M 102 0.6 V 18 H 119.4 M 102 62 H 119.4 V 79.4" />
          <path d="M 102 18 V 62" />
          <path d="M 113 30 H 119.4 V 50 H 113 Z" />
          <path d="M 108 35 A 5 5 0 0 0 108 45" />
          <circle cx="102" cy="40" r=".55" fill="#b8d5c2" />
          <path d="M 119.4 36 H 122 V 44 H 119.4" strokeWidth=".8" />
          <path d="M 60 0.6 V 79.4" strokeDasharray="1.6 1.6" opacity=".5" />
          <path d="M 60 30 A 10 10 0 0 1 60 50" strokeDasharray="1.6 1.6" opacity=".35" />
        </g>

        {/* The defensive half is drawn but inert: the model is not trained there. */}
        <g pointerEvents="none">
          <rect x="0" y="0" width="60" height="80" fill="#060f0a" opacity=".55" />
          <text x="30" y="40" textAnchor="middle" fill="#6d8a79" fontSize="2.6" fontFamily="DM Mono, monospace" letterSpacing=".3">OUTSIDE MODEL RANGE</text>
          <text x="30" y="44.5" textAnchor="middle" fill="#55705f" fontSize="2.1" fontFamily="DM Mono, monospace">ATTACKING HALF ONLY</text>
        </g>

        <g opacity=".45" fill="#d9f5e2" fontSize="2.4" fontFamily="DM Mono, monospace" pointerEvents="none">
          <text x="62" y="5">Y 0</text><text x="62" y="76">Y 80</text><text x="106" y="5">GOAL</text>
        </g>

        <path d={`M ${point.x} ${point.y} L 120 36 L 120 44 Z`} fill="#9bf3bd" opacity=".10" pointerEvents="none" />
        <path d={`M ${point.x} ${point.y} L 120 36 M ${point.x} ${point.y} L 120 44`} stroke="#9bf3bd" strokeWidth=".35" strokeDasharray="1.5 1.4" opacity=".55" pointerEvents="none" />
        <g pointerEvents="none">
          <circle cx={point.x} cy={point.y} r="3.4" fill="#75f0a5" opacity=".14" />
          <circle cx={point.x} cy={point.y} r="1.55" fill="#a5ffc5" stroke="#07150e" strokeWidth=".6" />
          <circle cx={point.x} cy={point.y} r="3" fill="none" stroke="#a5ffc5" strokeWidth=".35" strokeDasharray="1 1.2" />
        </g>
        {/* Chip sits to the right of the shot, and flips to the left near the goal
            so it never covers the marker. */}
        <g pointerEvents="none" transform={`translate(${point.x + 25 > 119 ? point.x - 25 : point.x + 4} ${Math.max(point.y - 5, 5)})`}>
          <rect width="21" height="5" rx="1" fill="#07150e" stroke="#5fd98d" strokeOpacity=".7" strokeWidth=".25" />
          <text x="1.5" y="3.35" fill="#b5f6cb" fontSize="2.15" fontFamily="DM Mono, monospace">SHOT LOCATION</text>
        </g>
      </svg>
      <div className="pitch-legend"><span><i className="legend-shot" /> Shot location</span><span><i className="legend-cone" /> Goal angle</span><span className="pitch-hint"><Crosshair size={12} /> Click or drag to reposition</span></div>
    </div>
  )
}
