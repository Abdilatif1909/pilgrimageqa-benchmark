import React, { useState } from 'react'

function RecommendationCard({ rec }) {
  return (
    <div className="rec-card">
      <h4>{rec.name}</h4>
      <p>{rec.description}</p>
      {rec.distance_km !== undefined && <p><small>Distance: {rec.distance_km} km</small></p>}
    </div>
  )
}

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)

  const send = async () => {
    if (!input.trim()) return
    const userMsg = { role: 'user', text: input }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)

    try {
      const res = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMsg.text })
      })
      const data = await res.json()
      const botMsg = { role: 'assistant', text: data.answer_uz }
      setMessages(prev => [...prev, botMsg])
      if (data.recommendations) {
        setMessages(prev => [...prev, { role: 'recommendations', items: data.recommendations }])
      }
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', text: 'Server error. Please check backend.' }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header>
        <h2>Pilgrimage Assistant</h2>
      </header>
      <main>
        <div className="chat">
          {messages.map((m, i) => (
            m.role === 'user' ? (
              <div key={i} className="message user">{m.text}</div>
            ) : m.role === 'assistant' ? (
              <div key={i} className="message bot">{m.text}</div>
            ) : (
              <div key={i} className="recommendations">
                <h3>Recommended places</h3>
                <div className="rec-list">
                  {m.items.map((r, idx) => <RecommendationCard key={idx} rec={r} />)}
                </div>
              </div>
            )
          ))}
        </div>
      </main>
      <footer>
        <input value={input} onChange={e => setInput(e.target.value)} placeholder="Ask in Uzbek, Russian or English..." />
        <button onClick={send} disabled={loading}>{loading ? '...' : 'Send'}</button>
      </footer>
    </div>
  )
}
