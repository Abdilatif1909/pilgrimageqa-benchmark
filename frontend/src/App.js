import React, { useEffect, useRef, useState } from 'react'
import { API_BASE_URL } from './config'
import './App.css'

function DecorativeBackdrop() {
  return (
    <div className="decorative-backdrop" aria-hidden="true">
      <svg className="decorative-backdrop__kaaba" viewBox="0 0 320 320" fill="none">
        <rect x="92" y="82" width="136" height="148" rx="10" fill="currentColor" opacity="0.08" />
        <rect x="92" y="118" width="136" height="12" fill="currentColor" opacity="0.15" />
        <rect x="105" y="95" width="38" height="14" rx="4" fill="currentColor" opacity="0.12" />
        <path d="M85 236H235" stroke="currentColor" strokeWidth="10" strokeLinecap="round" opacity="0.08" />
      </svg>

      <svg className="decorative-backdrop__nabawi" viewBox="0 0 420 220" fill="none">
        <path d="M24 184H394" stroke="currentColor" strokeWidth="3" opacity="0.18" />
        <path d="M110 184V112" stroke="currentColor" strokeWidth="4" opacity="0.22" />
        <path d="M302 184V92" stroke="currentColor" strokeWidth="4" opacity="0.22" />
        <path d="M96 112H124" stroke="currentColor" strokeWidth="4" opacity="0.22" />
        <path d="M288 92H316" stroke="currentColor" strokeWidth="4" opacity="0.22" />
        <path d="M173 184V120C173 94 193 74 219 74C245 74 265 94 265 120V184" stroke="currentColor" strokeWidth="4" opacity="0.2" />
        <path d="M189 120C189 102 203 88 221 88C239 88 253 102 253 120" stroke="currentColor" strokeWidth="4" opacity="0.22" />
        <path d="M218 72V54" stroke="currentColor" strokeWidth="4" opacity="0.2" />
      </svg>

      <svg className="decorative-backdrop__crescent" viewBox="0 0 180 180" fill="none">
        <path d="M105 22C79 30 60 54 60 83C60 118 88 146 123 146C132 146 140 144 148 140C136 152 120 160 102 160C63 160 32 129 32 90C32 56 56 27 89 20C94 19 99 19 105 22Z" fill="currentColor" opacity="0.12" />
        <circle cx="128" cy="58" r="5" fill="currentColor" opacity="0.18" />
        <circle cx="144" cy="74" r="3" fill="currentColor" opacity="0.14" />
      </svg>

      <div className="decorative-backdrop__pattern decorative-backdrop__pattern--top" />
      <div className="decorative-backdrop__pattern decorative-backdrop__pattern--bottom" />
    </div>
  )
}

function RecommendationCard({ rec }) {
  return (
    <article className="recommendation-card">
      <div className="recommendation-card__header">
        <div className="recommendation-card__icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none">
            <path d="M12 21C12 21 19 16.4 19 10.5C19 6.91 16.09 4 12.5 4C10.42 4 8.57 4.97 7.4 6.48C6.23 4.97 4.38 4 2.3 4C-1.29 4 -4.2 6.91 -4.2 10.5C-4.2 16.4 2.8 21 2.8 21H12Z" transform="translate(4.2 0)" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
        <div className="recommendation-card__content">
          <div className="recommendation-card__badge">Suggested</div>
          <h4>{rec.name}</h4>
        </div>
      </div>
      <p className="recommendation-card__description">{rec.description}</p>
      {rec.distance_km !== undefined && (
        <div className="recommendation-card__meta">
          Distance: {rec.distance_km} km
        </div>
      )}
    </article>
  )
}

function SendIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M4 12.5L20 4L15 20L10.8 13.2L4 12.5Z" fill="currentColor" opacity="0.2" />
      <path d="M20 4L10.8 13.2M20 4L15 20L10.8 13.2M20 4L4 12.5L10.8 13.2" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

function MessageBubble({ role, text, recommendations }) {
  const isUser = role === 'user'

  return (
    <div className={`chat-row ${isUser ? 'chat-row--user' : 'chat-row--assistant'}`}>
      <div className="chat-row__bubble-group">
        <div className="chat-row__main">
          <div className={`avatar ${isUser ? 'avatar--user' : 'avatar--assistant'}`}>
            {isUser ? 'You' : 'AI'}
          </div>
          <div className={`message-bubble ${isUser ? 'message-bubble--user' : 'message-bubble--assistant'}`}>
            <p className="message-bubble__text">{text}</p>
          </div>
        </div>

        {!isUser && recommendations?.length ? (
          <RecommendationSection items={recommendations} compact />
        ) : null}
      </div>
    </div>
  )
}

function RecommendationSection({ items, compact = false }) {
  if (!items?.length) return null

  return (
    <section className={`recommendations-panel ${compact ? 'recommendations-panel--compact' : ''}`}>
      <div className="recommendations-panel__header">
        <span className="recommendations-panel__eyebrow">Guidance</span>
        <h3>Recommended places</h3>
      </div>
      <div className="recommendations-grid">
        {items.map((item, index) => <RecommendationCard key={index} rec={item} />)}
      </div>
    </section>
  )
}

function TypingIndicator() {
  return (
    <div className="chat-row chat-row--assistant">
      <div className="avatar avatar--assistant">AI</div>
      <div className="message-bubble message-bubble--assistant message-bubble--typing" aria-label="Assistant is typing">
        <span className="typing-dot" />
        <span className="typing-dot" />
        <span className="typing-dot" />
      </div>
    </div>
  )
}

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef(null)

  const conversationItems = messages.reduce((accumulator, message) => {
    if (message.role === 'recommendations') {
      const lastItem = accumulator[accumulator.length - 1]
      if (lastItem?.role === 'assistant') {
        lastItem.recommendations = message.items || []
      }
      return accumulator
    }

    accumulator.push({ ...message })
    return accumulator
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [conversationItems, loading])

  const send = async () => {
    if (!input.trim() || loading) return

    const trimmedInput = input.trim()
    const userMsg = { role: 'user', text: trimmedInput }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)

    try {
      const res = await fetch(`${API_BASE_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmedInput })
      })

      const data = await res.json()
      const botMsg = { role: 'assistant', text: data.answer_uz }
      setMessages(prev => [...prev, botMsg])

      if (data.recommendations?.length) {
        setMessages(prev => [...prev, { role: 'recommendations', items: data.recommendations }])
      }
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', text: 'Server error. Please check backend.' }])
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = (event) => {
    event.preventDefault()
    send()
  }

  return (
    <div className="app-shell">
      <DecorativeBackdrop />
      <div className="app-shell__glow app-shell__glow--left" />
      <div className="app-shell__glow app-shell__glow--right" />

      <div className="assistant-layout">
        <header className="assistant-header">
          <div>
            <span className="assistant-header__eyebrow">Pilgrimage Assistant</span>
            <h1>Pilgrimage Assistant</h1>
            <p>Smart Hajj & Umrah Guidance</p>
          </div>
          <div className="assistant-header__status">
            <span className="assistant-header__dot" />
            Assistant Online
          </div>
        </header>

        <main className="assistant-main">
          <section className="chat-card">
            <div className="chat-card__body">
              {conversationItems.length === 0 ? (
                <div className="empty-state">
                  <div className="empty-state__icon">✦</div>
                  <h2>Ask anything about your pilgrimage journey</h2>
                  <p>
                    Get calm, practical guidance for Hajj and Umrah, including transport,
                    accommodation, rituals, locations, and nearby support services.
                  </p>
                </div>
              ) : (
                conversationItems.map((message, index) => (
                  <MessageBubble
                    key={index}
                    role={message.role}
                    text={message.text}
                    recommendations={message.recommendations}
                  />
                ))
              )}

              {loading && <TypingIndicator />}
              <div ref={messagesEndRef} />
            </div>

            <form className="composer composer--embedded" onSubmit={handleSubmit}>
              <div className="composer__inner">
                <input
                  value={input}
                  onChange={event => setInput(event.target.value)}
                  placeholder="Ask in Uzbek, Russian or English..."
                  className="composer__input"
                />
                <button type="submit" className="composer__button" disabled={loading || !input.trim()}>
                  <span>{loading ? 'Sending...' : 'Send'}</span>
                  <SendIcon />
                </button>
              </div>
            </form>
          </section>
        </main>
      </div>
    </div>
  )
}

