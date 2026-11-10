// An OpenAI Chat Completions-compatible transport. The host relays requests
// through its existing model proxy; API credentials never enter this project.
export function chatCompletion(messages, tools) {
  const id = crypto.randomUUID()
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => finish(new Error('timeout')), 90000)
    function finish(error, data) {
      clearTimeout(timeout)
      window.removeEventListener('message', onMessage)
      if (error) reject(error)
      else resolve(data)
    }
    function onMessage(event) {
      if (event.source !== window.parent || event.data?.type !== 'fe:chat-result' || event.data.id !== id) return
      finish(event.data.error ? new Error(event.data.error) : null, event.data.data)
    }
    window.addEventListener('message', onMessage)
    window.parent.postMessage({ type: 'fe:chat', id, body: { messages, tools } }, '*')
  })
}
