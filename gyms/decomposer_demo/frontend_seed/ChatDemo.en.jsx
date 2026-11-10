import { useEffect, useRef, useState } from 'react'
import Confetti from 'react-confetti'
import { chatCompletion } from './api'
import { executeTool, tools } from './tools'

const errors = {
  "empty": "The model returned an empty response.",
  "limit": "The assistant reached its tool limit. Try a new message.",
  "timeout": "The chat request timed out. Please try again.",
  "failed": "Could not send the message. Please try again."
}

export default function ChatDemo() {
  const [messages, setMessages] = useState([])
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [burst, setBurst] = useState(0)
  const [toolsEnabled, setToolsEnabled] = useState(true)
  const [size, setSize] = useState({ width: window.innerWidth, height: window.innerHeight })
  const bottom = useRef(null)
  const sending = useRef(false)

  useEffect(() => {
    const resize = () => setSize({ width: window.innerWidth, height: window.innerHeight })
    window.addEventListener('resize', resize)
    return () => window.removeEventListener('resize', resize)
  }, [])
  useEffect(() => { if (messages.length || busy) bottom.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, busy])

  function celebrate() { setBurst(value => value + 1) }

  async function send(text = draft) {
    if (!text.trim() || sending.current) return
    sending.current = true
    setBusy(true)
    setError('')
    setDraft('')
    const next = [...messages, { id: crypto.randomUUID(), role: 'user', content: text.trim() }]
    setMessages([...next])
    try {
      for (let turn = 0; turn < 4; turn++) {
        const response = await chatCompletion([{ role: 'system', content: "You are a helpful assistant in a chat model demo. Respond in English unless the user requests another language. Use throw_fireworks when the tool is available and the user asks to celebrate or throw confetti. Keep answers short." }, ...next].map(({ id, ...message }) => message), toolsEnabled ? tools : [])
        const message = response.choices?.[0]?.message
        if (!message) throw new Error('empty')
        next.push({ id: crypto.randomUUID(), role: 'assistant', content: message.content || null, ...(message.tool_calls?.length ? { tool_calls: message.tool_calls } : {}) })
        for (const call of message.tool_calls || []) {
          let result
          try { result = executeTool(call, celebrate) }
          catch (failure) { result = { error: failure.message } }
          next.push({ id: crypto.randomUUID(), role: 'tool', tool_call_id: call.id, content: JSON.stringify(result) })
        }
        setMessages([...next])
        if (!message.tool_calls?.length) return
      }
      throw new Error('limit')
    } catch (failure) { console.error(failure); setError(Object.hasOwn(errors, failure.message) ? failure.message : 'failed') }
    finally { setBusy(false); sending.current = false }
  }

  return (
    <div className="app">
      {burst > 0 && <Confetti key={burst} width={size.width} height={size.height} numberOfPieces={220} recycle={false} style={{ position: 'fixed', pointerEvents: 'none', zIndex: 10 }} />}
      <section className="controls" aria-label={"Settings"}>
        <label className="checkbox"><input type="checkbox" checked={toolsEnabled} onChange={event => setToolsEnabled(event.target.checked)} disabled={busy} />{"Enable tool calls"}</label>
        <button type="button" onClick={celebrate}>{"Test confetti"}</button>
        <div className="example-control">
        <label htmlFor="example">{"Example input"}</label>
        <select id="example" defaultValue="" disabled={busy} onChange={event => {
          setDraft(event.target.value === 'celebrate' ? "My experiment finished successfully! Throw confetti to celebrate." : event.target.value === 'idea' ? "Suggest a small data analysis project using a public dataset." : '')
        }}>
          <option value="">{"Custom message"}</option>
          <option value="celebrate">{"Celebrate a result"}</option>
          <option value="idea">{"Suggest a project"}</option>
        </select>
        </div>
        <button type="button" disabled={busy || !messages.length} onClick={() => { setMessages([]); setError('') }}>{"Clear chat"}</button>
      </section>
      <main className="main">
        <form onSubmit={event => { event.preventDefault(); void send() }}>
          <label htmlFor="message">{"Your message"}</label>
          <textarea id="message" placeholder={"Enter a message..."} rows={3} value={draft} onChange={event => setDraft(event.target.value)} disabled={busy} />
          <button className="primary" type="submit" disabled={busy || !draft.trim()}>{"Send"}</button>
        </form>
        <h2 className="chat-title">{"Chat history"}</h2>
        <section className="messages" aria-label={"Chat history"} aria-live="polite">
          {!messages.length && <p className="empty">{"No messages yet."}</p>}
          {messages.map(message => message.role === 'tool' ? null : (
            <div className={`message ${message.role}`} key={message.id}>
              <strong>{message.role === 'user' ? "You" : "Assistant"}</strong>
              {message.content && <p>{message.content}</p>}
              {message.tool_calls?.map(call => <details className="tool-call" key={call.id}>
                <summary>{"Tool call"}: <code>{call.function.name}</code></summary>
                <pre>{call.function.arguments}</pre>
              </details>)}
            </div>
          ))}
          {busy && <p role="status">{"Generating response"}…</p>}
          {error && <p className="error" role="alert">{errors[error]}</p>}
          <div ref={bottom} />
        </section>
      </main>
    </div>
  )
}
