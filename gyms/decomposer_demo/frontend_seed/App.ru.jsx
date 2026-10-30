import ChatDemo from './ChatDemo'

export default function App() {
  return (
    <div className="page">
      <header className="page-header">
        <p className="eyebrow">Интерактивный эксперимент</p>
        <h1>Демо чата</h1>
        <p className="description">Диалог, модель и один инструмент, который превращает слова в конфетти. Попробуйте демо и придумайте ему свой облик.</p>
      </header>
      <ChatDemo />
      <footer className="page-footer">
        <p>Попросите модель отметить успешный эксперимент. Сравните ответ с включёнными и выключенными вызовами инструментов.</p>
      </footer>
    </div>
  )
}
