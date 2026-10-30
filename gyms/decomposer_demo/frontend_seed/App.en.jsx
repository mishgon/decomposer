import ChatDemo from './ChatDemo'

export default function App() {
  return (
    <div className="page">
      <header className="page-header">
        <p className="eyebrow">Interactive experiment</p>
        <h1>Chat demo</h1>
        <p className="description">A conversation, a model, and one tool that turns words into confetti. Try the demo, then make it your own.</p>
      </header>
      <ChatDemo />
      <footer className="page-footer">
        <p>Try asking the model to celebrate a successful experiment. Compare its response with tool calls enabled and disabled.</p>
      </footer>
    </div>
  )
}
