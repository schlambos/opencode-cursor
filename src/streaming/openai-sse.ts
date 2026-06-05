import {
  extractText,
  extractThinking,
  inferToolName,
  isAssistantText,
  isThinking,
  isToolCall,
  type StreamJsonEvent,
  type StreamJsonToolCallEvent,
} from "./types.js";
import { MixedDeltaTracker } from "./delta-tracker.js";
import { createLogger } from "../utils/logger.js";

const log = createLogger("streaming:openai-sse");

type OpenAiToolCall = {
  index: number;
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
};

type OpenAiDelta = {
  content?: string;
  reasoning_content?: string;
  tool_calls?: OpenAiToolCall[];
};

type OpenAiChunk = {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: OpenAiDelta;
    finish_reason: string | null;
  }>;
};

const createChunk = (id: string, created: number, model: string, delta: OpenAiDelta): OpenAiChunk => ({
  id,
  object: "chat.completion.chunk",
  created,
  model,
  choices: [
    {
      index: 0,
      delta,
      finish_reason: null,
    },
  ],
});

export const formatSseChunk = (payload: object) => `data: ${JSON.stringify(payload)}\n\n`;

export const formatSseDone = () => "data: [DONE]\n\n";

export class StreamToSseConverter {
  private readonly id: string;
  private readonly created: number;
  private readonly model: string;
  private readonly tracker = new MixedDeltaTracker();

  constructor(model: string, options?: { id?: string; created?: number }) {
    this.model = model;
    this.id = options?.id ?? `cursor-acp-${Date.now()}`;
    this.created = options?.created ?? Math.floor(Date.now() / 1000);
  }

  handleEvent(event: StreamJsonEvent): string[] {
    if (isAssistantText(event)) {
      const text = extractText(event);
      if (!text) return [];
      const delta = this.tracker.nextText(text);
      return delta ? [this.chunkWith({ content: delta })] : [];
    }

    if (isThinking(event)) {
      const text = extractThinking(event);
      if (!text) return [];
      const delta = this.tracker.nextThinking(text);
      return delta ? [this.chunkWith({ reasoning_content: delta })] : [];
    }

    if (isToolCall(event)) {
      // DIAG: log every converter-path tool_call emission so we can see
      // whether the original tool_call (e.g. "edit") ever leaks through to
      // opencode in parallel with an intercept's rerouted call (e.g. "write").
      const delta = this.toolCallDelta(event);
      try {
        const tc = delta.tool_calls?.[0];
        log.debug("Converter emitted tool_call SSE chunk", {
          callId: tc?.id,
          name: tc?.function?.name,
          argsPreview: typeof tc?.function?.arguments === "string"
            ? tc.function.arguments.slice(0, 200)
            : null,
        });
      } catch { /* ignore */ }
      return [this.chunkWith(delta)];
    }

    return [];
  }

  private chunkWith(delta: OpenAiDelta): string {
    return formatSseChunk(createChunk(this.id, this.created, this.model, delta));
  }

  private toolCallDelta(event: StreamJsonToolCallEvent): OpenAiDelta {
    const id = event.call_id ?? "unknown";
    const toolName = inferToolName(event) || "tool";
    const toolKey = Object.keys(event.tool_call ?? {})[0];
    const args = toolKey ? event.tool_call[toolKey]?.args : undefined;
    const argumentsText = args ? JSON.stringify(args) : "";

    return {
      tool_calls: [
        {
          index: 0,
          id,
          type: "function",
          function: {
            name: toolName,
            arguments: argumentsText,
          },
        },
      ],
    };
  }
}
