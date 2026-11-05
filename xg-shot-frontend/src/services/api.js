const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api'

const BODY_PARTS = ['Left Foot', 'Other', 'Right Foot']
const TECHNIQUES = ['Diving Header', 'Half Volley', 'Lob', 'Normal', 'Overhead Kick', 'Volley']
const PLAY_PATTERNS = ['From Counter', 'From Free Kick', 'From Goal Kick', 'From Keeper', 'From Kick Off', 'From Throw In', 'Other', 'Regular Play']

const toSnakeSuffix = (value) => value.replaceAll(' ', '_')

export function buildShotPayload(shot) {
  const payload = {
    distance_to_goal: Number(shot.distance.toFixed(4)),
    shot_angle: Number(shot.angle.toFixed(6)),
    defenders_in_cone: Number(shot.defenders),
    keeper_off_line: Number(shot.keeperOff),
    keeper_lateral_offset: Number(shot.keeperOffset),
    under_pressure: shot.underPressure ? 1 : 0,
    shot_first_time: shot.firstTime ? 1 : 0,
  }

  // One-hot columns are sent as 0 / 1, matching how the model was trained.
  // Sending true / false relies on the API coercing booleans to numbers.
  BODY_PARTS.forEach((part) => {
    payload[`body_part_name_${toSnakeSuffix(part)}`] = shot.bodyPart === part ? 1 : 0
  })
  TECHNIQUES.forEach((technique) => {
    payload[`technique_name_${toSnakeSuffix(technique)}`] = shot.technique === technique ? 1 : 0
  })
  PLAY_PATTERNS.forEach((pattern) => {
    payload[`play_pattern_name_${toSnakeSuffix(pattern)}`] = shot.playPattern === pattern ? 1 : 0
  })
  return payload
}

export async function predictShot(shot) {
  const response = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(buildShotPayload(shot)),
  })

  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = typeof data.detail === 'string' ? data.detail : `API error (${response.status})`
    throw new Error(detail)
  }
  if (typeof data.probabilite_de_but !== 'number') {
    throw new Error('La réponse API ne contient pas « probabilite_de_but ».')
  }
  return data
}
