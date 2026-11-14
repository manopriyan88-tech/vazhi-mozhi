/**
 * TypeScript definitions for frontend_bridge.js
 */

export interface BridgeConfig {
  baseUrl: string;
  wsUrl: string;
  sessionId: string;
  timeoutMs: number;
}

export interface RegisterKeyParams {
  apiKey: string;
  provider: string;
  endpointUrl: string;
  sessionId?: string;
  authType?: 'bearer' | 'api_key_header' | 'api_key_query' | string;
  authHeaderName?: string;
  authParamName?: string;
  skipVerification?: boolean;
  geminiApiKey?: string;
}

export interface RegisterKeyResponse {
  success: boolean;
  sessionId: string;
  provider: string;
  maskedKey: string;
  endpointUrl: string;
  authType: string;
  isActive: boolean;
  verification: {
    isActive: boolean;
    status: string;
    statusCode: number;
    latencyMs: number;
    authMethod: string;
    message: string;
    sampleData?: any;
  };
  functionDeclaration: {
    name: string;
    description: string;
    parameters: {
      type: string;
      properties: Record<string, any>;
      required?: string[];
    };
  };
  generatorSource: string;
  message: string;
}

export interface KeyListItem {
  sessionId: string;
  provider: string;
  maskedKey: string;
  endpointUrl: string;
  authType: string;
  isActive: boolean;
  generatorSource: string;
  functionDeclarationName: string;
  createdAt: string;
  updatedAt: string;
}

export interface ExecutionEvent {
  step:
    | 'connected'
    | 'inspecting_key'
    | 'keys_inspected'
    | 'planning_started'
    | 'plan_generated'
    | 'wave_started'
    | 'calling_3rd_party_api'
    | 'rate_limit_backoff'
    | 'self_healing_started'
    | 'self_healing_success'
    | 'api_response_received'
    | 'wave_completed'
    | 'synthesizing'
    | 'synthesis_completed'
    | 'completed'
    | 'error'
    | string;
  timestamp: number;
  isoTime?: string;
  provider?: string;
  taskId?: string;
  endpoint?: string;
  method?: string;
  payload?: any;
  wave?: number;
  statusCode?: number;
  elapsedSeconds?: number;
  delaySeconds?: number;
  attempt?: number;
  selfHealed?: boolean;
  data?: any;
  summary?: string;
  content?: string;
  keyInsights?: string[];
  apisUsed?: string[];
  totalTimeSeconds?: number;
  message?: string;
  [key: string]: any;
}

export interface NodeState {
  id: string;
  label: string;
  status: 'idle' | 'running' | 'success' | 'warning' | 'failed' | 'healed';
  lastUpdated: number;
  message?: string;
  [key: string]: any;
}

export declare class ApiKitchenClient {
  constructor(options?: Partial<BridgeConfig>);
  baseUrl: string;
  sessionId: string;
  timeoutMs: number;

  registerKey(params: RegisterKeyParams): Promise<RegisterKeyResponse>;
  listKeys(sessionId?: string): Promise<KeyListItem[]>;
  getKey(provider: string, sessionId?: string): Promise<any>;
  getKeySchema(provider: string, sessionId?: string): Promise<any>;
  deleteKey(provider: string, sessionId?: string): Promise<{ success: boolean; message: string }>;
  clearSession(sessionId?: string): Promise<{ success: boolean; clearedCount: number; message: string }>;
  seedDemo(options?: { execute?: boolean; prompt?: string }): Promise<any>;
  checkHealth(): Promise<{ status: string; env: string; model: string }>;
}

export declare class ExecutionWebSocket {
  constructor(options?: {
    wsUrl?: string;
    sessionId?: string;
    autoReconnect?: boolean;
    reconnectIntervalMs?: number;
    heartbeatIntervalMs?: number;
  });

  isConnected: boolean;
  connect(): Promise<void>;
  disconnect(): void;
  send(payload: any): void;

  execute(prompt: string, options?: { sessionId?: string; includeDemo?: boolean }): this;
  executeAsync(
    prompt: string,
    options?: {
      onStep?: (step: string, event: ExecutionEvent) => void;
      onLog?: (logLine: string, event: ExecutionEvent) => void;
      sessionId?: string;
      includeDemo?: boolean;
    }
  ): Promise<any>;

  on(event: string, handler: (...args: any[]) => void): this;
  off(event: string, handler: (...args: any[]) => void): this;
  once(event: string, handler: (...args: any[]) => void): this;
  emit(event: string, ...args: any[]): this;
  removeAllListeners(): this;
}

export declare function configure(options?: Partial<BridgeConfig>): BridgeConfig;
export declare function getConfig(): BridgeConfig;

export declare function registerKey(params: RegisterKeyParams): Promise<RegisterKeyResponse>;
export declare function listKeys(sessionId?: string): Promise<KeyListItem[]>;
export declare function getKey(provider: string, sessionId?: string): Promise<any>;
export declare function getKeySchema(provider: string, sessionId?: string): Promise<any>;
export declare function deleteKey(provider: string, sessionId?: string): Promise<any>;
export declare function clearSession(sessionId?: string): Promise<any>;
export declare function seedDemo(options?: { execute?: boolean; prompt?: string }): Promise<any>;
export declare function checkHealth(): Promise<any>;

export declare function createExecutionWebSocket(options?: any): ExecutionWebSocket;
export declare function executePrompt(prompt: string, options?: any): Promise<any>;

export declare function streamExecutionSSE(params: {
  prompt: string;
  sessionId?: string;
  includeDemo?: boolean;
  baseUrl?: string;
  onStep?: (step: string, data: ExecutionEvent) => void;
  onLog?: (logLine: string, data: ExecutionEvent) => void;
  onComplete?: (data: any) => void;
  onError?: (err: Error) => void;
}): { close: () => void };

export declare function formatTerminalLog(event: ExecutionEvent): string;
export declare function deriveGraphState(
  currentState?: Record<string, NodeState>,
  event?: ExecutionEvent
): Record<string, NodeState>;
