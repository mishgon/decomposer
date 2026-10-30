export const tools = [{
  type: 'function',
  function: {
    name: 'throw_fireworks',
    description: 'Throw colorful confetti to celebrate. Use when the user asks for fireworks, confetti, or a celebration.',
    parameters: {
      type: 'object',
      properties: { reason: { type: 'string', description: 'A short reason to celebrate' } },
      required: ['reason'],
      additionalProperties: false,
    },
  },
}]

export function executeTool(call, celebrate) {
  if (call.function.name !== 'throw_fireworks') return { error: 'Unknown tool' }
  const args = JSON.parse(call.function.arguments || '{}')
  celebrate()
  return { success: true, reason: args.reason || 'A little celebration', effect: 'Confetti is falling in the chat.' }
}
