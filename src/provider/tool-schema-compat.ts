import * as fs from "fs";
import { createLogger } from "../utils/logger.js";
import type { OpenAiToolCall } from "../proxy/tool-loop.js";

const log = createLogger("provider:tool-schema-compat");

type JsonRecord = Record<string, unknown>;

const EDIT_COMPAT_REPAIR_ENABLED = process.env.CURSOR_ACP_EDIT_COMPAT_REPAIR !== "false";

const ARG_KEY_ALIASES = new Map<string, string>([
  ["filepath", "path"],
  ["filename", "path"],
  ["file", "path"],
  ["targetpath", "path"],
  ["directorypath", "path"],
  ["dir", "path"],
  ["folder", "path"],
  ["directory", "path"],
  ["targetdirectory", "path"],
  ["targetfile", "path"],
  ["globpattern", "pattern"],
  ["filepattern", "pattern"],
  ["searchpattern", "pattern"],
  ["includepattern", "include"],
  ["workingdirectory", "cwd"],
  ["workdir", "cwd"],
  ["currentdirectory", "cwd"],
  ["cmd", "command"],
  ["script", "command"],
  ["shellcommand", "command"],
  ["terminalcommand", "command"],
  ["contents", "content"],
  ["text", "content"],
  ["body", "content"],
  ["data", "content"],
  ["payload", "content"],
  ["streamcontent", "content"],
  ["recursive", "force"],
  ["oldstring", "old_string"],
  ["newstring", "new_string"],
]);

export interface ToolSchemaValidationResult {
  hasSchema: boolean;
  ok: boolean;
  missing: string[];
  unexpected: string[];
  typeErrors: string[];
  repairHint?: string;
}

export interface ToolSchemaCompatResult {
  toolCall: OpenAiToolCall;
  normalizedArgs: JsonRecord;
  originalArgs: JsonRecord;
  originalArgKeys: string[];
  normalizedArgKeys: string[];
  collisionKeys: string[];
  validation: ToolSchemaValidationResult;
}

export function buildToolSchemaMap(tools: Array<unknown>): Map<string, unknown> {
  const schemas = new Map<string, unknown>();
  for (const rawTool of tools) {
    const tool = isRecord(rawTool) ? rawTool : null;
    if (!tool) {
      continue;
    }
    const fn = isRecord(tool.function) ? tool.function : tool;
    const name = typeof fn.name === "string" ? fn.name.trim() : "";
    if (!name) {
      continue;
    }
    if (fn.parameters !== undefined) {
      schemas.set(name, fn.parameters);
    }
  }
  return schemas;
}

export function applyToolSchemaCompat(
  toolCall: OpenAiToolCall,
  toolSchemaMap: Map<string, unknown>,
): ToolSchemaCompatResult {
  const parsedArgs = parseArguments(toolCall.function.arguments);
  const originalArgKeys = Object.keys(parsedArgs);
  const { normalizedArgs, collisionKeys } = normalizeArgumentKeys(parsedArgs);
  const toolSpecificArgs = normalizeToolSpecificArgs(toolCall.function.name, normalizedArgs);
  const schema = toolSchemaMap.get(toolCall.function.name);
  const sanitization = sanitizeArgumentsForSchema(toolSpecificArgs, schema);
  const validation = validateToolArguments(
    toolCall.function.name,
    sanitization.args,
    schema,
    sanitization.unexpected,
    {
      originalArgs: parsedArgs,
      writeSchema: toolSchemaMap.get("write"),
    },
  );

  const normalizedToolCall: OpenAiToolCall = {
    ...toolCall,
    function: {
      ...toolCall.function,
      arguments: JSON.stringify(sanitization.args),
    },
  };

  return {
    toolCall: normalizedToolCall,
    normalizedArgs: sanitization.args,
    originalArgs: parsedArgs,
    originalArgKeys,
    normalizedArgKeys: Object.keys(sanitization.args),
    collisionKeys,
    validation,
  };
}

export function isFullFileShapedEditValidationFailure(
  toolName: string,
  args: JsonRecord,
  validation: ToolSchemaValidationResult,
  originalArgs: JsonRecord,
  writeSchema?: unknown,
): boolean {
  if (toolName.toLowerCase() !== "edit" || validation.ok) {
    return false;
  }
  return buildEditFullFileHint(args, validation.missing, validation.typeErrors, {
    originalArgs,
    writeSchema,
  }) !== null;
}

function buildWriteArguments(
  filePath: string,
  content: string,
  writeSchema: unknown,
): JsonRecord {
  if (!isRecord(writeSchema)) {
    return { path: filePath, content };
  }
  const required = Array.isArray(writeSchema.required)
    ? writeSchema.required.filter((value): value is string => typeof value === "string")
    : [];
  if (required.includes("filePath")) {
    return { filePath, content };
  }
  return { path: filePath, content };
}

function buildEditArguments(
  filePath: string,
  oldString: string,
  newString: string,
  editSchema: unknown,
): JsonRecord {
  const useCamelCase = (() => {
    if (!isRecord(editSchema)) return true;
    const required = Array.isArray(editSchema.required)
      ? editSchema.required.filter((value): value is string => typeof value === "string")
      : [];
    if (required.includes("oldString") || required.includes("newString") || required.includes("filePath")) {
      return true;
    }
    if (required.includes("old_string") || required.includes("new_string") || required.includes("path")) {
      return false;
    }
    return true;
  })();
  return useCamelCase
    ? { filePath, oldString, newString, replaceAll: false }
    : { path: filePath, old_string: oldString, new_string: newString, replace_all: false };
}

/**
 * Translate a malformed edit (path + body, no old_string) into a real edit/write
 * call. Tier 1: file does not exist → write (creation). Tier 2: file exists →
 * opencode edit with anchor-derived oldString/newString. Tier 3: refuse (null).
 *
 * Replaces the old "always reroute to write" behavior, which caused destructive
 * truncations when cursor's Composer emitted a partial streamContent chunk.
 */
export function tryRerouteEditToWrite(
  toolCall: OpenAiToolCall,
  compat: ToolSchemaCompatResult,
  allowedToolNames: Set<string>,
  toolSchemaMap: Map<string, unknown>,
): OpenAiToolCall | null {
  if (toolCall.function.name.toLowerCase() !== "edit") {
    return null;
  }
  if (!allowedToolNames.has("write") || !toolSchemaMap.has("write")) {
    return null;
  }

  const writeSchema = toolSchemaMap.get("write");
  if (
    !isFullFileShapedEditValidationFailure(
      toolCall.function.name,
      compat.normalizedArgs,
      compat.validation,
      compat.originalArgs,
      writeSchema,
    )
  ) {
    return null;
  }

  const filePath = typeof compat.normalizedArgs.path === "string" && compat.normalizedArgs.path.length > 0
    ? compat.normalizedArgs.path
    : typeof compat.normalizedArgs.filePath === "string" && compat.normalizedArgs.filePath.length > 0
      ? compat.normalizedArgs.filePath
      : null;
  if (!filePath) {
    return null;
  }

  const content =
    typeof compat.normalizedArgs.new_string === "string"
      ? compat.normalizedArgs.new_string
      : typeof compat.normalizedArgs.newString === "string"
        ? compat.normalizedArgs.newString
        : typeof compat.normalizedArgs.content === "string"
          ? compat.normalizedArgs.content
          : null;
  if (content === null) {
    return null;
  }

  // Tier 2: file exists → try to translate to opencode edit using anchors.
  let existing: string | null = null;
  try {
    const stat = fs.statSync(filePath);
    if (stat.isFile()) {
      existing = fs.readFileSync(filePath, "utf8");
    }
  } catch {
    // ENOENT / permission / etc. — fall through to Tier 1.
  }

  if (existing !== null) {
    if (allowedToolNames.has("edit") && toolSchemaMap.has("edit")) {
      const editSchema = toolSchemaMap.get("edit");
      const translated = translateStreamContentToEdit(existing, content);
      if (translated) {
        log.debug("Translated streamContent edit to opencode edit", {
          path: filePath,
          oldStringLen: translated.oldString.length,
          newStringLen: translated.newString.length,
        });
        return {
          ...toolCall,
          function: {
            name: "edit",
            arguments: JSON.stringify(
              buildEditArguments(filePath, translated.oldString, translated.newString, editSchema),
            ),
          },
        };
      }
    }
    // Tier 3: file exists but can't translate. Refuse — caller emits a hint.
    log.warn("Refusing reroute: cannot anchor streamContent in existing file", {
      path: filePath,
      existingLen: existing.length,
      streamContentLen: content.length,
    });
    return null;
  }

  // Tier 1: file does not exist → write (creation).
  return {
    ...toolCall,
    function: {
      name: "write",
      arguments: JSON.stringify(buildWriteArguments(filePath, content, writeSchema)),
    },
  };
}

/**
 * Pre-process the args a model emitted for `edit` or `write`, before they reach
 * the local registry's tool handler in `plugin.ts:createEntry`. This is a sibling
 * fix to the proxy-boundary translation in `tryRerouteEditToWrite`: some edit
 * calls bypass the proxy boundary entirely (parallel tool emissions in one
 * model turn) and arrive at the hook handler with cursor-native shapes the
 * local registry doesn't understand. We normalize and, when possible, translate
 * here so the handler sees args it can execute.
 *
 * - Aliases `filePath` → `path`, `oldString` → `old_string`, `newString` → `new_string`.
 * - For edit with a body but no old_string AND an existing file: try to derive
 *   old_string + new_string from anchors in the file (same algorithm as the
 *   proxy-side translator).
 * - Otherwise leaves args untouched. The handler will throw or create-on-ENOENT
 *   as appropriate.
 */
export function preprocessEditWriteArgs(toolName: string, args: JsonRecord): JsonRecord {
  const lowered = toolName.toLowerCase();
  if (lowered !== "edit" && lowered !== "write") {
    return args;
  }
  const out: JsonRecord = { ...args };

  if (typeof out.filePath === "string" && typeof out.path !== "string") {
    out.path = out.filePath;
  }
  if (lowered === "edit") {
    if (typeof out.oldString === "string" && typeof out.old_string !== "string") {
      out.old_string = out.oldString;
    }
    if (typeof out.newString === "string" && typeof out.new_string !== "string") {
      out.new_string = out.newString;
    }
    if (typeof out.new_string !== "string") {
      const fallback = typeof out.content === "string"
        ? out.content
        : typeof out.streamContent === "string"
          ? out.streamContent
          : null;
      if (fallback !== null) {
        out.new_string = fallback;
      }
    }
    // If the model emitted a body but no old_string, and the file exists, try
    // anchor-based translation so the registry's edit handler can apply the
    // change in-place instead of overwriting (its empty-old_string branch is
    // a full file replace, which is what truncated mod.rs earlier).
    if (
      typeof out.path === "string"
      && out.path.length > 0
      && typeof out.new_string === "string"
      && typeof out.old_string !== "string"
    ) {
      let existing: string | null = null;
      let fileMissing = false;
      try {
        const stat = fs.statSync(out.path);
        if (stat.isFile()) {
          existing = fs.readFileSync(out.path, "utf8");
        }
      } catch {
        fileMissing = true;
        // File doesn't exist — leave as-is, registry will create.
      }
      if (existing !== null) {
        const translated = translateStreamContentToEdit(existing, out.new_string);
        if (translated) {
          log.debug("Preprocessed edit args: translated streamContent to old_string/new_string", {
            path: out.path,
            oldStringLen: translated.oldString.length,
            newStringLen: translated.newString.length,
          });
          out.old_string = translated.oldString;
          out.new_string = translated.newString;
        } else {
          // File exists but can't anchor — refuse to silently rewrite. Set
          // old_string to a sentinel that can't match so the registry's edit
          // returns "Could not find the text to replace" instead of treating
          // empty old_string as a full overwrite.
          log.warn("Preprocessed edit args: cannot anchor streamContent, blocking full-file overwrite", {
            path: out.path,
            newStringLen: out.new_string.length,
          });
          out.old_string = "__cursor_acp_unmappable_edit__";
        }
      } else if (fileMissing) {
        // File does not exist. Set old_string="" so the local registry's
        // edit handler passes its required-type check (typeof oldString
        // !== "string") and reaches its ENOENT branch (which creates the
        // file with new_string). Without this, the handler throws
        // "missing required argument 'old_string'" before it can create
        // the file.
        log.debug("Preprocessed edit args: creating non-existent file via edit (old_string='')", {
          path: out.path,
          newStringLen: out.new_string.length,
        });
        out.old_string = "";
      }
    }
  }

  return out;
}

/**
 * Find oldString/newString such that opencode's edit can apply the model's
 * intended localized change. Returns null if a start anchor can't be uniquely
 * placed or an end anchor isn't found after it.
 *
 * Algorithm:
 * 1. Pick a "start anchor" — the first non-blank line of streamContent that
 *    appears exactly once in existing. If single-line is ambiguous, expand to a
 *    2-line then 3-line composite (candidate + following streamContent lines)
 *    and re-check uniqueness. Give up at 3 lines.
 * 2. Pick an "end anchor" — streamContent's last non-blank line, and find the
 *    FIRST occurrence in existing starting from the start-anchor's position.
 *    Doesn't have to be unique globally — anchoring from the start position
 *    disambiguates well enough for the localized-edit case.
 * 3. oldString = existing slice from start of start-anchor's first line to end
 *    of end-anchor's line. newString = streamContent verbatim.
 */
function translateStreamContentToEdit(
  existing: string,
  streamContent: string,
): { oldString: string; newString: string } | null {
  if (existing === streamContent) {
    return null;
  }
  const existingLines = existing.split("\n");
  const newLines = streamContent.split("\n");

  const startAnchor = findStartAnchor(existingLines, newLines);
  if (!startAnchor) return null;

  const endLineInExisting = findEndAnchor(existingLines, newLines, startAnchor.existingStartLine);
  if (endLineInExisting === -1) return null;

  if (endLineInExisting < startAnchor.existingStartLine) {
    return null;
  }

  const oldString = existingLines.slice(startAnchor.existingStartLine, endLineInExisting + 1).join("\n");
  const newString = streamContent;

  if (oldString === newString) return null;
  if (!existing.includes(oldString)) return null;

  return { oldString, newString };
}

type StartAnchorResult = {
  streamStartLine: number;
  existingStartLine: number;
};

/**
 * Find a non-blank line of streamContent that appears uniquely in existing.
 * Walks forward through streamContent's candidate lines. For each candidate,
 * tries 1-line, 2-line, 3-line composites (the candidate + following lines)
 * and returns the smallest unique match. Gives up if a candidate's
 * single-line position has 0 matches (can't disambiguate) or if all candidates
 * are exhausted.
 */
function findStartAnchor(existingLines: string[], newLines: string[]): StartAnchorResult | null {
  for (let candidateIdx = 0; candidateIdx < newLines.length; candidateIdx++) {
    if (newLines[candidateIdx].trim().length === 0) continue;

    for (let size = 1; size <= 3; size++) {
      const end = candidateIdx + size - 1;
      if (end >= newLines.length) break;
      const composite = newLines.slice(candidateIdx, end + 1);
      const positions = findCompositePositions(existingLines, composite);
      if (positions.length === 1) {
        return { streamStartLine: candidateIdx, existingStartLine: positions[0] };
      }
      if (positions.length === 0) break;
    }
  }
  return null;
}

/**
 * Find the line in existing (at or after `startLine`) that matches
 * streamContent's last non-blank line. First match wins — anchoring from the
 * start anchor's known position handles disambiguation in practice.
 */
function findEndAnchor(existingLines: string[], newLines: string[], startLine: number): number {
  let endCandidate = -1;
  for (let i = newLines.length - 1; i >= 0; i--) {
    if (newLines[i].trim().length > 0) {
      endCandidate = i;
      break;
    }
  }
  if (endCandidate === -1) return -1;
  const target = newLines[endCandidate];
  for (let i = startLine; i < existingLines.length; i++) {
    if (existingLines[i] === target) return i;
  }
  return -1;
}

function findCompositePositions(existingLines: string[], composite: string[]): number[] {
  if (composite.length === 0) return [];
  const out: number[] = [];
  const limit = existingLines.length - composite.length + 1;
  for (let i = 0; i < limit; i++) {
    let match = true;
    for (let j = 0; j < composite.length; j++) {
      if (existingLines[i + j] !== composite[j]) {
        match = false;
        break;
      }
    }
    if (match) out.push(i);
  }
  return out;
}

function parseArguments(rawArguments: string): JsonRecord {
  try {
    const parsed = JSON.parse(rawArguments);
    if (isRecord(parsed)) {
      return parsed;
    }
    return { value: parsed };
  } catch {
    return { value: rawArguments };
  }
}

function normalizeArgumentKeys(args: JsonRecord): {
  normalizedArgs: JsonRecord;
  collisionKeys: string[];
} {
  const normalizedArgs: JsonRecord = { ...args };
  const collisionKeys: string[] = [];

  for (const [rawKey, rawValue] of Object.entries(args)) {
    const canonicalKey = resolveCanonicalArgKey(rawKey);
    if (!canonicalKey || canonicalKey === rawKey) {
      continue;
    }

    const canonicalInOriginal = hasOwn(args, canonicalKey);
    const canonicalInNormalized = hasOwn(normalizedArgs, canonicalKey);
    if (canonicalInOriginal || canonicalInNormalized) {
      collisionKeys.push(rawKey);
      delete normalizedArgs[rawKey];
      continue;
    }

    normalizedArgs[canonicalKey] = rawValue;
    delete normalizedArgs[rawKey];
  }

  return { normalizedArgs, collisionKeys };
}

function resolveCanonicalArgKey(rawKey: string): string | null {
  const token = rawKey.toLowerCase().replace(/[^a-z0-9]/g, "");
  return ARG_KEY_ALIASES.get(token) ?? null;
}

function normalizeToolSpecificArgs(toolName: string, args: JsonRecord): JsonRecord {
  const normalizedToolName = toolName.toLowerCase();
  if (normalizedToolName === "bash") {
    const normalized: JsonRecord = { ...args };
    const normalizedCommand = normalizeBashCommand(normalized.command);
    if (typeof normalizedCommand === "string" && normalizedCommand.trim().length > 0) {
      normalized.command = normalizedCommand;
    }
    if (
      normalized.cwd === undefined
      && typeof normalized.path === "string"
      && normalized.path.trim().length > 0
    ) {
      normalized.cwd = normalized.path;
    }
    return normalized;
  }

  if (normalizedToolName === "rm") {
    const normalized: JsonRecord = { ...args };
    if (typeof normalized.force === "string") {
      const lowered = normalized.force.trim().toLowerCase();
      if (lowered === "true" || lowered === "1" || lowered === "yes") {
        normalized.force = true;
      } else if (lowered === "false" || lowered === "0" || lowered === "no") {
        normalized.force = false;
      }
    }
    return normalized;
  }

  if (normalizedToolName === "todowrite") {
    if (!Array.isArray(args.todos)) {
      return args;
    }

    const todos = args.todos.map((entry) => {
      if (!isRecord(entry)) {
        return entry;
      }

      const todo: JsonRecord = { ...entry };
      if (typeof todo.status === "string") {
        todo.status = normalizeTodoStatus(todo.status);
      }
      if (
        todo.priority === undefined
        || todo.priority === null
        || (typeof todo.priority === "string" && todo.priority.trim().length === 0)
      ) {
        todo.priority = "medium";
      }
      return todo;
    });

    return {
      ...args,
      todos,
    };
  }

  if (normalizedToolName === "write") {
    const normalized: JsonRecord = { ...args };

    // Some model variants confuse write/edit and send edit-style payload keys.
    // Map them into canonical write arguments before schema validation/sanitization.
    if (normalized.content === undefined && normalized.new_string !== undefined) {
      const coerced = coerceToString(normalized.new_string);
      if (coerced !== null) {
        normalized.content = coerced;
      }
      delete normalized.new_string;
    }

    if (normalized.content !== undefined && typeof normalized.content !== "string") {
      const coerced = coerceToString(normalized.content);
      if (coerced !== null) {
        normalized.content = coerced;
      }
    }

    return normalized;
  }

  if (normalizedToolName !== "edit" || !EDIT_COMPAT_REPAIR_ENABLED) {
    return args;
  }

  const repaired: JsonRecord = { ...args };
  const hasStringNew = typeof repaired.new_string === "string";
  const hasStringOld = typeof repaired.old_string === "string";

  // Coerce non-string content/streamContent into a string before repair.
  // Models frequently emit array-of-chunks (streamContent) or object payloads.
  if (repaired.content !== undefined && typeof repaired.content !== "string") {
    const coerced = coerceToString(repaired.content);
    if (coerced !== null) {
      repaired.content = coerced;
    }
  }

  const content = repaired.content;

  // Guarded compatibility repair for models that send full-content edit payloads.
  if (!hasStringNew && typeof content === "string") {
    repaired.new_string = content;
  }
  if (hasStringOld && repaired.old_string === "") {
    delete repaired.old_string;
  }

  return repaired;
}

function normalizeBashCommand(value: unknown): string | null {
  if (typeof value === "string") {
    return value;
  }
  if (Array.isArray(value)) {
    const parts = value
      .map((entry) => (typeof entry === "string" ? entry : coerceToString(entry)))
      .filter((entry): entry is string => typeof entry === "string" && entry.length > 0);
    return parts.length > 0 ? parts.join(" ") : null;
  }
  if (isRecord(value)) {
    const command = typeof value.command === "string" ? value.command : null;
    const args = Array.isArray(value.args)
      ? value.args
          .map((entry) => (typeof entry === "string" ? entry : coerceToString(entry)))
          .filter((entry): entry is string => typeof entry === "string" && entry.length > 0)
      : [];
    if (command && args.length > 0) {
      return [command, ...args].join(" ");
    }
    if (command) {
      return command;
    }
  }
  return null;
}

function normalizeTodoStatus(status: string): string {
  const normalized = status.trim().toLowerCase().replace(/[\s-]+/g, "_");
  if (normalized === "todo_status_pending") {
    return "pending";
  }
  if (normalized === "todo_status_inprogress" || normalized === "todo_status_in_progress") {
    return "in_progress";
  }
  if (
    normalized === "todo_status_done"
    || normalized === "todo_status_complete"
    || normalized === "todo_status_completed"
  ) {
    return "completed";
  }
  if (normalized === "todo" || normalized === "pending") {
    return "pending";
  }
  if (normalized === "inprogress" || normalized === "in_progress") {
    return "in_progress";
  }
  if (normalized === "done" || normalized === "complete" || normalized === "completed") {
    return "completed";
  }
  return status;
}

function sanitizeArgumentsForSchema(
  args: JsonRecord,
  schema: unknown,
): { args: JsonRecord; unexpected: string[] } {
  if (!isRecord(schema)) {
    return { args, unexpected: [] };
  }

  if (schema.additionalProperties !== false) {
    return { args, unexpected: [] };
  }

  const properties = isRecord(schema.properties) ? schema.properties : {};
  const propertyNames = new Set(Object.keys(properties));
  const sanitized: JsonRecord = {};
  const unexpected: string[] = [];

  for (const [key, value] of Object.entries(args)) {
    if (propertyNames.has(key)) {
      sanitized[key] = value;
      continue;
    }
    unexpected.push(key);
  }

  return { args: sanitized, unexpected };
}

type ValidateToolArgumentsContext = {
  originalArgs?: JsonRecord;
  writeSchema?: unknown;
};

function validateToolArguments(
  toolName: string,
  args: JsonRecord,
  schema: unknown,
  unexpected: string[],
  context: ValidateToolArgumentsContext = {},
): ToolSchemaValidationResult {
  if (!isRecord(schema)) {
    return {
      hasSchema: false,
      ok: true,
      missing: [],
      unexpected: [],
      typeErrors: [],
    };
  }

  const properties = isRecord(schema.properties) ? schema.properties : {};
  const required = Array.isArray(schema.required)
    ? schema.required.filter((value): value is string => typeof value === "string")
    : [];
  const missing = required.filter((key) => !hasOwn(args, key));

  const typeErrors: string[] = [];
  for (const [key, value] of Object.entries(args)) {
    const propertySchema = properties[key];
    if (!isRecord(propertySchema)) {
      continue;
    }
    if (!matchesType(value, propertySchema.type)) {
      if (propertySchema.type !== undefined) {
        typeErrors.push(`${key}: expected ${String(propertySchema.type)}`);
      }
      continue;
    }
    if (
      Array.isArray(propertySchema.enum)
      && !propertySchema.enum.some((candidate) => Object.is(candidate, value))
    ) {
      typeErrors.push(`${key}: expected enum ${JSON.stringify(propertySchema.enum)}`);
    }
  }

  const ok = missing.length === 0 && typeErrors.length === 0;
  return {
    hasSchema: true,
    ok,
    missing,
    unexpected,
    typeErrors,
    repairHint: ok
      ? undefined
      : buildRepairHint(toolName, args, missing, unexpected, typeErrors, context),
  };
}

function hadOldStringPropertyInPayload(args: JsonRecord): boolean {
  for (const key of Object.keys(args)) {
    const token = key.toLowerCase().replace(/[^a-z0-9]/g, "");
    if (token === "oldstring") {
      return true;
    }
  }
  return false;
}

function hasEditFilePath(args: JsonRecord): boolean {
  const pathValue = args.path ?? args.filePath;
  return typeof pathValue === "string" && pathValue.trim().length > 0;
}

function hasEditBody(args: JsonRecord): boolean {
  const body = args.new_string ?? args.newString ?? args.content;
  return typeof body === "string" && body.length > 0;
}

function writeToolExample(writeSchema: unknown): string {
  if (!isRecord(writeSchema)) {
    return "write with path and content";
  }
  const required = Array.isArray(writeSchema.required)
    ? writeSchema.required.filter((value): value is string => typeof value === "string")
    : [];
  if (required.includes("filePath")) {
    return "write with filePath and content";
  }
  return "write with path and content";
}

function buildEditFullFileHint(
  args: JsonRecord,
  missing: string[],
  typeErrors: string[],
  context: ValidateToolArgumentsContext,
): string | null {
  if (typeErrors.length > 0) {
    return null;
  }

  const missingHasOldString = missing.includes("old_string") || missing.includes("oldString");
  if (!missingHasOldString) {
    return null;
  }
  // The other missing keys are accepted as long as the normalized args have a
  // path-like and body-like value — they're typically filePath/newString
  // counterparts of the compat layer's snake_case bias against opencode's
  // camelCase schema.
  const benignMissingKeys = new Set([
    "old_string",
    "oldString",
    "new_string",
    "newString",
    "path",
    "filePath",
  ]);
  if (!missing.every((key) => benignMissingKeys.has(key))) {
    return null;
  }

  const originalArgs = context.originalArgs ?? {};
  if (hadOldStringPropertyInPayload(originalArgs)) {
    return null;
  }

  if (!hasEditFilePath(args) || !hasEditBody(args)) {
    return null;
  }

  const example = writeToolExample(context.writeSchema);
  return `For a full file body, use ${example} instead of edit without old_string`;
}

function buildRepairHint(
  toolName: string,
  args: JsonRecord,
  missing: string[],
  unexpected: string[],
  typeErrors: string[],
  context: ValidateToolArgumentsContext = {},
): string {
  const fullFileHint = toolName.toLowerCase() === "edit"
    ? buildEditFullFileHint(args, missing, typeErrors, context)
    : null;
  if (fullFileHint) {
    return fullFileHint;
  }

  const hints: string[] = [];
  if (missing.length > 0) {
    hints.push(`missing required: ${missing.join(", ")}`);
  }
  if (unexpected.length > 0) {
    hints.push(`remove unsupported fields: ${unexpected.join(", ")}`);
  }
  if (typeErrors.length > 0) {
    hints.push(`fix type errors: ${typeErrors.join("; ")}`);
  }

  if (
    toolName.toLowerCase() === "edit"
    && (missing.includes("old_string") || missing.includes("oldString") || missing.includes("new_string") || missing.includes("newString"))
  ) {
    hints.push("edit requires path, old_string, and new_string");
  }

  return hints.join(" | ");
}

function matchesType(value: unknown, schemaType: unknown): boolean {
  if (schemaType === undefined) {
    return true;
  }
  if (Array.isArray(schemaType)) {
    return schemaType.some((entry) => matchesType(value, entry));
  }
  if (typeof schemaType !== "string") {
    return true;
  }
  switch (schemaType) {
    case "string":
      return typeof value === "string";
    case "number":
      return typeof value === "number";
    case "integer":
      return typeof value === "number" && Number.isInteger(value);
    case "boolean":
      return typeof value === "boolean";
    case "object":
      return isRecord(value);
    case "array":
      return Array.isArray(value);
    case "null":
      return value === null;
    default:
      return true;
  }
}

function coerceToString(value: unknown): string | null {
  if (typeof value === "string") {
    return value;
  }
  if (value === null || value === undefined) {
    return null;
  }
  if (Array.isArray(value)) {
    const parts: string[] = [];
    for (const item of value) {
      if (typeof item === "string") {
        parts.push(item);
      } else if (isRecord(item)) {
        const text = typeof item.text === "string"
          ? item.text
          : typeof item.content === "string"
            ? item.content
            : typeof item.value === "string"
              ? item.value
              : null;
        if (text !== null) {
          parts.push(text);
        } else {
          parts.push(JSON.stringify(item));
        }
      } else {
        parts.push(String(item));
      }
    }
    return parts.length > 0 ? parts.join("") : null;
  }
  if (isRecord(value)) {
    if (typeof value.text === "string") {
      return value.text;
    }
    if (typeof value.content === "string") {
      return value.content;
    }
    if (typeof value.value === "string") {
      return value.value;
    }
    return JSON.stringify(value);
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return null;
}

function hasOwn(record: JsonRecord, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(record, key);
}

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
