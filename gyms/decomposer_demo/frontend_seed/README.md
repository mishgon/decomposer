# Chat demo

A basic data-science chat demo built in React, with a single-column layout and compact inline controls,
standard form controls, chat history, and OpenAI Chat Completions tool calls.
It runs in Sandpack's browser React runtime (no Node server or terminal).

- App.jsx: small editable page shell with a heading, introduction and <ChatDemo />.
- ChatDemo.jsx: working chat UI, confetti and bounded tool-call loop.
- api.js: chatCompletion(messages, tools), a bridge to the demo's existing model proxy.
- tools.js: the throw_fireworks function schema and browser tool handler.
- styles.css: plain CSS, including responsive layout.
- index.js: React entry point.

The host fixes the model and keeps credentials outside the editable application.
The fireworks button works without a model connection. Asking the assistant to celebrate
uses a real model tool call, executes react-confetti, and returns a tool result.
Dependencies are pinned in package.json. Use browser-compatible npm packages.
Read files before editing; tools remember your last read/write revision automatically. Other agents share
this project. Sandpack updates automatically when files change.

Screenshot inspection is supplied by the host. No screenshot files, imports, or dependencies are needed in this project.

The host selected this single-language seed when creating the project. Keep its language. Text lives directly in App.jsx; no localization framework or language switching is needed.
