import { ArrowLeft, BrainCircuit, Database, Info, Layers3, ShieldCheck } from 'lucide-react'

export default function MethodNotes({ onBack }) {
  return <div className="method-page">
    <div className="page-kicker">PROJECT DOCUMENTATION / MODEL</div>
    <h1>Behind the <em>prediction.</em></h1>
    <p className="page-intro">A transparent look at the features sent to the existing model and the dataset described in the project README.</p>
    <div className="method-grid">
      <article className="method-card"><Database/><span>DATASET</span><strong>11,861</strong><p>Shots across 477 matches and 12 competitions, according to the repository README.</p></article>
      <article className="method-card"><Layers3/><span>FEATURE ENGINEERING</span><strong>24 inputs</strong><p>Geometric shot features, body part, technique and play-pattern indicators.</p></article>
      <article className="method-card"><BrainCircuit/><span>SERVED MODEL</span><strong>Logistic regression</strong><p>The current API loads <code>models/logistic.pkl</code> at startup.</p></article>
    </div>
    <section className="method-detail"><div className="detail-title"><Info size={17}/><h2>What influences an xG estimate?</h2></div><div className="feature-list"><div><b>Shot geometry</b><span>Distance to goal and visible angle between the posts.</span></div><div><b>Defensive pressure</b><span>Defenders inside the shooting cone and whether the shot is under pressure.</span></div><div><b>Goalkeeper position</b><span>How far the goalkeeper has advanced and their lateral offset.</span></div><div><b>Shot context</b><span>Body part, technique, first-time status and play pattern.</span></div></div></section>
    <div className="method-note"><ShieldCheck size={17}/><p><b>Model integrity</b><br/>This page documents the current API. The frontend does not recreate the ML calculation: it sends the 24 input features to <code>POST /predict</code> and displays the API response.</p></div>
    <button className="back-button" onClick={onBack}><ArrowLeft size={15}/> Back to shot studio</button>
  </div>
}
