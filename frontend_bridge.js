/**
 * frontend_bridge.js
 *
 * Lightweight, universal JavaScript/TypeScript client library for the
 * Adaptive API Kitchen & Multi-Agent Orchestrator backend.
 *
 * Capabilities:
 * 1. REST client for 3rd-party API key registration & vault inspection (/api/keys/register)
 * 2. Real-time WebSocket execution client (/ws/execute) with step-based event listeners
 * 3. Server-Sent Events (SSE) fallback client (/api/stream)
 * 4. Terminal log formatter for UI consoles
 * 5. Animated node execution state reducer for visual graph interfaces
 *
 * Compatible with Node.js, ES Modules, CommonJS, and standard browser <script> tags.
 */

(function (root, factory) {
  if (typeof define === 'function' && define.amd) {
    define([], factory);
  } else if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.ApiKitchenBridge = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // =========================================================================
  // Default Configuration & URL Helpers
  // =========================================================================

  const isBrowser = typeof window !== 'undefined';

  function getDefaultBaseUrl() {
    if (isBrowser && window.location) {
      const port = window.location.port ? `:${window.location.port}` : '';
      return `${window.location.protocol}//${window.location.hostname}${port}`;
    }
    return 'http://localhost:8000';
  }

  function getDefaultWsUrl(baseUrl) {
    if (baseUrl.startsWith('https://')) {
      return baseUrl.replace('https://', 'wss://');
    }
    if (baseUrl.startsWith('http://')) {
      return baseUrl.replace('http://', 'ws://');
    }
    if (isBrowser && window.location) {
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const port = window.location.port ? `:${window.location.port}` : '';
      return `${proto}//${window.location.hostname}${port}`;
    }
    return 'ws://localhost:8000';
  }

  const globalConfig = {
    baseUrl: getDefaultBaseUrl(),
    wsUrl: getDefaultWsUrl(getDefaultBaseUrl()),
    sessionId: 'default',
    timeoutMs: 30000,
  };

  /**
   * Updates global client configuration.
   * @param {Object} options
   * @param {string} [options.baseUrl] - Base HTTP URL (e.g. 'http://localhost:8000')
   * @param {string} [options.wsUrl] - WebSocket URL (e.g. 'ws://localhost:8000')
   * @param {string} [options.sessionId] - Default session identifier
   */
  function configure(options = {}) {
    if (options.baseUrl) {
      globalConfig.baseUrl = options.baseUrl.replace(/\/+$/, '');
      if (!options.wsUrl) {
        globalConfig.wsUrl = getDefaultWsUrl(globalConfig.baseUrl);
      }
    }
    if (options.wsUrl) {
      globalConfig.wsUrl = options.wsUrl.replace(/\/+$/, '');
    }
    if (options.sessionId) {
      globalConfig.sessionId = options.sessionId;
    }
    if (options.timeoutMs) {
      globalConfig.timeoutMs = options.timeoutMs;
    }
    return { ...globalConfig };
  }

  function getConfig() {
    return { ...globalConfig };
  }

  // =========================================================================
  // Event Emitter (Lightweight Pub/Sub)
  // =========================================================================

  class EventEmitter {
    constructor() {
      this._listeners = new Map();
    }

    on(event, handler) {
      if (typeof handler !== 'function') return this;
      if (!this._listeners.has(event)) {
        this._listeners.set(event, new Set());
      }
      this._listeners.get(event).add(handler);
      return this;
    }

    off(event, handler) {
      if (this._listeners.has(event)) {
        this._listeners.get(event).delete(handler);
      }
      return this;
    }

    once(event, handler) {
      const wrapper = (...args) => {
        this.off(event, wrapper);
        handler(...args);
      };
      return this.on(event, wrapper);
    }

    emit(event, ...args) {
      if (this._listeners.has(event)) {
        for (const handler of this._listeners.get(event)) {
          try {
            handler(...args);
          } catch (err) {
            console.error(`[ApiKitchenBridge] Error in event listener '${event}':`, err);
          }
        }
      }
      // Also notify wildcard '*' listeners
      if (event !== '*' && this._listeners.has('*')) {
        for (const handler of this._listeners.get('*')) {
          try {
            handler(event, ...args);
          } catch (err) {
            console.error(`[ApiKitchenBridge] Error in wildcard listener for '${event}':`, err);
          }
        }
      }
      return this;
    }

    removeAllListeners() {
      this._listeners.clear();
      return this;
    }
  }

  // =========================================================================
  // 1. REST API Client
  // =========================================================================

  class ApiKitchenClient {
    constructor(options = {}) {
      this.baseUrl = (options.baseUrl || globalConfig.baseUrl).replace(/\/+$/, '');
      this.sessionId = options.sessionId || globalConfig.sessionId;
      this.timeoutMs = options.timeoutMs || globalConfig.timeoutMs;
    }

    async _fetch(endpoint, init = {}) {
      const url = `${this.baseUrl}${endpoint}`;
      const headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        ...(init.headers || {}),
      };

      const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
      const timeoutId = controller ? setTimeout(() => controller.abort(), this.timeoutMs) : null;

      try {
        const response = await fetch(url, {
          ...init,
          headers,
          signal: controller ? controller.signal : undefined,
        });

        clearTimeout(timeoutId);

        let data = null;
        const text = await response.text();
        if (text) {
          try {
            data = JSON.parse(text);
          } catch {
            data = { text };
          }
        }

        if (!response.ok) {
          const errMsg = data?.detail || data?.message || `HTTP ${response.status}: ${response.statusText}`;
          const err = new Error(errMsg);
          err.status = response.status;
          err.data = data;
          throw err;
        }

        return data;
      } catch (err) {
        if (timeoutId) clearTimeout(timeoutId);
        throw err;
      }
    }

    /**
     * Registers a 3rd-party API key in the session vault, verifies it via httpx probe,
     * and auto-generates a clean Gemini Function Declaration schema.
     *
     * @param {Object} params
     * @param {string} params.apiKey - 3rd-party API key or credential secret
     * @param {string} params.provider - Provider name (e.g., 'WeatherAPI', 'GitHub', 'Stripe')
     * @param {string} params.endpointUrl - Sample endpoint or documentation URL
     * @param {string} [params.sessionId] - Session namespace (defaults to client session)
     * @param {string} [params.authType] - 'bearer', 'api_key_header', 'api_key_query', or null
     * @param {string} [params.authHeaderName] - Custom header name (e.g. 'x-api-key')
     * @param {string} [params.authParamName] - Custom query param name (e.g. 'key')
     * @param {boolean} [params.skipVerification=false] - Whether to skip the async probe
     * @param {string} [params.geminiApiKey] - Optional Gemini API key override
     */
    async registerKey(params) {
      if (!params || !params.apiKey || !params.provider || !params.endpointUrl) {
        throw new Error("registerKey requires 'apiKey', 'provider', and 'endpointUrl'.");
      }

      const payload = {
        api_key: params.apiKey,
        provider: params.provider,
        endpoint_url: params.endpointUrl,
        session_id: params.sessionId || this.sessionId,
        auth_type: params.authType || null,
        auth_header_name: params.authHeaderName || null,
        auth_param_name: params.authParamName || null,
        skip_verification: Boolean(params.skipVerification),
        gemini_api_key: params.geminiApiKey || null,
      };

      return await this._fetch('/api/keys/register', {
        method: 'POST',
        body: JSON.stringify(payload),
      });
    }

    /**
     * Lists all registered keys in a session (with keys safely masked).
     * @param {string} [sessionId]
     */
    async listKeys(sessionId) {
      const sess = encodeURIComponent(sessionId || this.sessionId);
      return await this._fetch(`/api/keys?session_id=${sess}`);
    }

    /**
     * Retrieves key details and schema for a specific provider.
     * @param {string} provider
     * @param {string} [sessionId]
     */
    async getKey(provider, sessionId) {
      const prov = encodeURIComponent(provider);
      const sess = encodeURIComponent(sessionId || this.sessionId);
      return await this._fetch(`/api/keys/${prov}?session_id=${sess}`);
    }

    /**
     * Retrieves the Gemini Function Declaration JSON schema for a provider.
     * @param {string} provider
     * @param {string} [sessionId]
     */
    async getKeySchema(provider, sessionId) {
      const prov = encodeURIComponent(provider);
      const sess = encodeURIComponent(sessionId || this.sessionId);
      return await this._fetch(`/api/keys/${prov}/schema?session_id=${sess}`);
    }

    /**
     * Deletes a registered key from the session vault.
     * @param {string} provider
     * @param {string} [sessionId]
     */
    async deleteKey(provider, sessionId) {
      const prov = encodeURIComponent(provider);
      const sess = encodeURIComponent(sessionId || this.sessionId);
      return await this._fetch(`/api/keys/${prov}?session_id=${sess}`, {
        method: 'DELETE',
      });
    }

    /**
     * Clears all registered keys in a session.
     * @param {string} [sessionId]
     */
    async clearSession(sessionId) {
      const sess = encodeURIComponent(sessionId || this.sessionId);
      return await this._fetch(`/api/keys/clear?session_id=${sess}`, {
        method: 'POST',
      });
    }

    /**
     * Pre-loads free public demo APIs (Open-Meteo, GitHub, CoinGecko) and optionally executes a wave.
     * @param {Object} [options]
     * @param {boolean} [options.execute=true]
     * @param {string} [options.prompt]
     */
    async seedDemo(options = {}) {
      const execute = options.execute !== false;
      const promptParam = options.prompt ? `&prompt=${encodeURIComponent(options.prompt)}` : '';
      return await this._fetch(`/api/demo/seed?execute=${execute}${promptParam}`, {
        method: 'POST',
      });
    }

    /**
     * Backend health check.
     */
    async checkHealth() {
      return await this._fetch('/health', { method: 'GET' });
    }
  }

  // =========================================================================
  // 2. WebSocket Execution Client (/ws/execute)
  // =========================================================================

  class ExecutionWebSocket extends EventEmitter {
    /**
     * @param {Object} [options]
     * @param {string} [options.wsUrl] - WebSocket server URL
     * @param {string} [options.sessionId] - Session identifier
     * @param {boolean} [options.autoReconnect=true] - Auto reconnect on disconnect
     * @param {number} [options.reconnectIntervalMs=3000] - Backoff delay for reconnect
     * @param {number} [options.heartbeatIntervalMs=15000] - Ping interval
     */
    constructor(options = {}) {
      super();
      this.wsUrl = (options.wsUrl || globalConfig.wsUrl).replace(/\/+$/, '');
      this.sessionId = options.sessionId || globalConfig.sessionId;
      this.autoReconnect = options.autoReconnect !== false;
      this.reconnectIntervalMs = options.reconnectIntervalMs || 3000;
      this.heartbeatIntervalMs = options.heartbeatIntervalMs || 15000;

      this.ws = null;
      this._isConnected = false;
      this._manualClose = false;
      this._heartbeatTimer = null;
      this._reconnectTimer = null;
      this._activeExecution = null;
    }

    get isConnected() {
      return this._isConnected && this.ws && this.ws.readyState === (WebSocket.OPEN || 1);
    }

    /**
     * Connects to /ws/execute.
     * @returns {Promise<void>} Resolves when connection is open and authenticated.
     */
    connect() {
      if (this.isConnected) {
        return Promise.resolve();
      }

      this._manualClose = false;
      const endpoint = `${this.wsUrl}/ws/execute`;

      return new Promise((resolve, reject) => {
        try {
          const WSConstructor = typeof WebSocket !== 'undefined' ? WebSocket : null;
          if (!WSConstructor) {
            throw new Error('WebSocket is not supported in this runtime environment.');
          }

          this.ws = new WSConstructor(endpoint);

          const onOpenTimeout = setTimeout(() => {
            if (!this._isConnected) {
              reject(new Error(`WebSocket connection timeout to '${endpoint}'`));
            }
          }, 10000);

          this.ws.onopen = () => {
            clearTimeout(onOpenTimeout);
            this._isConnected = true;
            this._startHeartbeat();
            this.emit('open');
          };

          this.ws.onmessage = (event) => {
            let data = null;
            try {
              data = typeof event.data === 'string' ? JSON.parse(event.data) : event.data;
            } catch {
              data = { raw: event.data };
            }

            this._handleIncomingMessage(data);

            if (data?.step === 'connected') {
              resolve();
            }
          };

          this.ws.onerror = (err) => {
            clearTimeout(onOpenTimeout);
            this.emit('error', err);
            if (!this._isConnected) {
              reject(err);
            }
          };

          this.ws.onclose = (event) => {
            clearTimeout(onOpenTimeout);
            this._isConnected = false;
            this._stopHeartbeat();
            this.emit('close', event);

            if (!this._manualClose && this.autoReconnect) {
              this._scheduleReconnect();
            }
          };
        } catch (err) {
          reject(err);
        }
      });
    }

    _handleIncomingMessage(event) {
      if (!event) return;

      const step = event.step || event.type || 'message';

      // 1. Emit specific step listener (e.g. 'inspecting_key', 'calling_3rd_party_api', 'completed')
      this.emit(step, event);

      // 2. Emit generic 'step' listener with (step, event)
      this.emit('step', step, event);

      // 3. Emit 'log' event formatted for terminal consoles
      const formattedLog = formatTerminalLog(event);
      if (formattedLog) {
        this.emit('log', formattedLog, event);
      }

      // 4. Resolve/reject active async executePromise if listening
      if (this._activeExecution) {
        if (step === 'completed') {
          this._activeExecution.resolve(event.data || event);
          this._activeExecution = null;
        } else if (step === 'error') {
          this._activeExecution.reject(new Error(event.message || 'Execution error received'));
          this._activeExecution = null;
        }
      }
    }

    _startHeartbeat() {
      this._stopHeartbeat();
      this._heartbeatTimer = setInterval(() => {
        if (this.isConnected) {
          this.send({ action: 'ping' });
        }
      }, this.heartbeatIntervalMs);
    }

    _stopHeartbeat() {
      if (this._heartbeatTimer) {
        clearInterval(this._heartbeatTimer);
        this._heartbeatTimer = null;
      }
    }

    _scheduleReconnect() {
      if (this._reconnectTimer) return;
      this._reconnectTimer = setTimeout(() => {
        this._reconnectTimer = null;
        if (!this.isConnected && !this._manualClose) {
          this.connect().catch((err) => {
            this.emit('reconnect_failed', err);
          });
        }
      }, this.reconnectIntervalMs);
    }

    /**
     * Sends a raw JSON object over the WebSocket.
     */
    send(payload) {
      if (!this.isConnected) {
        throw new Error('WebSocket is not connected. Call connect() first.');
      }
      this.ws.send(JSON.stringify(payload));
    }

    /**
     * Initiates a multi-agent orchestration execution.
     * Events are streamed back through the registered event listeners.
     *
     * @param {string} prompt - User instruction or query
     * @param {Object} [options]
     * @param {string} [options.sessionId] - Session namespace
     * @param {boolean} [options.includeDemo=true] - Auto-load public connectors if empty
     */
    execute(prompt, options = {}) {
      if (!prompt) {
        throw new Error("execute requires a 'prompt' string.");
      }

      const payload = {
        prompt: prompt.trim(),
        session_id: options.sessionId || this.sessionId,
        include_demo: options.includeDemo !== false,
      };

      this.send(payload);
      return this;
    }

    /**
     * Convenience async helper: connects (if needed), dispatches the prompt,
     * triggers optional onStep callback on each event, and resolves with the final completed data.
     *
     * @param {string} prompt
     * @param {Object} [options]
     * @param {function(string, Object): void} [options.onStep] - Callback fired on every event step
     * @param {function(string, Object): void} [options.onLog] - Callback fired with formatted terminal lines
     * @param {string} [options.sessionId]
     * @param {boolean} [options.includeDemo=true]
     * @returns {Promise<Object>} Resolves with the SynthesizedResponse data
     */
    async executeAsync(prompt, options = {}) {
      if (!this.isConnected) {
        await this.connect();
      }

      if (options.onStep) {
        this.on('step', options.onStep);
      }
      if (options.onLog) {
        this.on('log', options.onLog);
      }

      return new Promise((resolve, reject) => {
        this._activeExecution = {
          resolve: (data) => {
            if (options.onStep) this.off('step', options.onStep);
            if (options.onLog) this.off('log', options.onLog);
            resolve(data);
          },
          reject: (err) => {
            if (options.onStep) this.off('step', options.onStep);
            if (options.onLog) this.off('log', options.onLog);
            reject(err);
          },
        };

        try {
          this.execute(prompt, options);
        } catch (err) {
          if (options.onStep) this.off('step', options.onStep);
          if (options.onLog) this.off('log', options.onLog);
          reject(err);
        }
      });
    }

    /**
     * Closes the WebSocket connection.
     */
    disconnect() {
      this._manualClose = true;
      this._stopHeartbeat();
      if (this._reconnectTimer) {
        clearTimeout(this._reconnectTimer);
        this._reconnectTimer = null;
      }
      if (this.ws) {
        this.ws.close();
        this.ws = null;
      }
      this._isConnected = false;
      this.emit('disconnected');
    }
  }

  // =========================================================================
  // 3. Server-Sent Events (SSE) Client (/api/stream)
  // =========================================================================

  /**
   * Subscribes to live execution events via Server-Sent Events (/api/stream).
   *
   * @param {Object} params
   * @param {string} params.prompt - Prompt to execute
   * @param {string} [params.sessionId] - Session namespace
   * @param {boolean} [params.includeDemo=true]
   * @param {string} [params.baseUrl]
   * @param {function(string, Object): void} [params.onStep] - Invoked on every step
   * @param {function(string, Object): void} [params.onLog] - Invoked with formatted terminal line
   * @param {function(Object): void} [params.onComplete] - Invoked upon final completed event
   * @param {function(Error): void} [params.onError] - Invoked upon error
   * @returns {{ close: function(): void }} Subscription handle to cancel the stream
   */
  function streamExecutionSSE(params) {
    if (!params || !params.prompt) {
      throw new Error("streamExecutionSSE requires a 'prompt' string.");
    }

    const base = (params.baseUrl || globalConfig.baseUrl).replace(/\/+$/, '');
    const query = new URLSearchParams({
      prompt: params.prompt,
      session_id: params.sessionId || globalConfig.sessionId,
      include_demo: String(params.includeDemo !== false),
    });

    const url = `${base}/api/stream?${query.toString()}`;

    if (typeof EventSource !== 'undefined') {
      const es = new EventSource(url);

      es.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          const step = data.step || 'message';

          if (params.onStep) {
            params.onStep(step, data);
          }

          if (params.onLog) {
            const logLine = formatTerminalLog(data);
            if (logLine) params.onLog(logLine, data);
          }

          if (step === 'completed') {
            if (params.onComplete) params.onComplete(data.data || data);
            es.close();
          } else if (step === 'error') {
            if (params.onError) params.onError(new Error(data.message || 'SSE execution error'));
            es.close();
          }
        } catch (err) {
          console.error('[ApiKitchenBridge] Error parsing SSE message:', err);
        }
      };

      es.onerror = (err) => {
        if (params.onError) {
          params.onError(err);
        }
        es.close();
      };

      return {
        close: () => es.close(),
      };
    } else {
      // Fallback for non-browser/Node environments without EventSource
      let aborted = false;
      const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;

      (async () => {
        try {
          const resp = await fetch(url, {
            headers: { 'Accept': 'text/event-stream' },
            signal: controller ? controller.signal : undefined,
          });

          if (!resp.ok) {
            throw new Error(`HTTP ${resp.status} on SSE: ${resp.statusText}`);
          }

          const reader = resp.body.getReader();
          const decoder = new TextDecoder();
          let buffer = '';

          while (!aborted) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n\n');
            buffer = lines.pop(); // keep partial

            for (const chunk of lines) {
              const dataLine = chunk.split('\n').find((l) => l.startsWith('data: '));
              if (dataLine) {
                const jsonText = dataLine.slice(6).trim();
                if (jsonText) {
                  const data = JSON.parse(jsonText);
                  const step = data.step || 'message';

                  if (params.onStep) params.onStep(step, data);
                  if (params.onLog) {
                    const logLine = formatTerminalLog(data);
                    if (logLine) params.onLog(logLine, data);
                  }

                  if (step === 'completed') {
                    if (params.onComplete) params.onComplete(data.data || data);
                    aborted = true;
                    break;
                  }
                }
              }
            }
          }
        } catch (err) {
          if (!aborted && params.onError) {
            params.onError(err);
          }
        }
      })();

      return {
        close: () => {
          aborted = true;
          if (controller) controller.abort();
        },
      };
    }
  }

  // =========================================================================
  // 4. Live Terminal Log Formatter
  // =========================================================================

  /**
   * Formats an execution event into a clean, human-readable terminal log line.
   * Suitable for terminal UI widgets and scrolling log views.
   *
   * @param {Object} event - Raw JSON event emitted by the backend
   * @returns {string} Formatted terminal log line
   */
  function formatTerminalLog(event) {
    if (!event || !event.step) return '';

    const timeStr = event.iso_time ? event.iso_time.split('T')[1].split('.')[0] : new Date().toLocaleTimeString();
    const prefix = `[${timeStr}]`;

    switch (event.step) {
      case 'connected':
        return `${prefix} [CONN] ⚡ ${event.message || 'Connected to backend execution stream.'}`;

      case 'inspecting_key':
        return `${prefix} [VAULT] 🔑 Inspecting credentials in in-memory session vault...`;

      case 'keys_inspected':
        return `${prefix} [VAULT] ✅ Discovered ${event.active_connectors_count || 0} active connector(s): [${(event.connectors || []).join(', ')}]`;

      case 'planning_started':
        return `${prefix} [PLANNER] 🧠 Decomposing prompt: "${event.prompt}"`;

      case 'plan_generated':
        return `${prefix} [PLANNER] 📋 Plan created: ${event.tasks_count || 0} tasks planned across execution waves.`;

      case 'wave_started':
        return `${prefix} [WAVE ${event.wave || 1}] 🌊 Launching parallel wave with ${event.tasks_count || 0} task(s)...`;

      case 'calling_3rd_party_api': {
        const prov = event.provider || 'API';
        const method = event.method || 'GET';
        const payloadStr = event.payload && Object.keys(event.payload).length > 0
          ? ` (payload: ${JSON.stringify(event.payload)})`
          : '';
        return `${prefix} [DISPATCH] 🚀 ${method} -> ${prov}${payloadStr}`;
      }

      case 'rate_limit_backoff':
        return `${prefix} [WARN] ⏳ HTTP 429 Rate Limit encountered. Backing off for ${event.delay_seconds}s (attempt ${event.attempt})...`;

      case 'self_healing_started':
        return `${prefix} [HEAL] 🩹 Parameter validation error detected (HTTP ${event.status_code}). Gemini auto-repairing payload...`;

      case 'self_healing_success':
        return `${prefix} [HEAL] ✨ Payload successfully healed and validated! Retrying request...`;

      case 'api_response_received': {
        const prov = event.provider || 'API';
        const code = event.status_code || 200;
        const elapsed = event.elapsed_seconds ? `${event.elapsed_seconds}s` : '';
        const icon = event.success ? '✅' : '❌';
        return `${prefix} [RESPONSE] ${icon} ${prov} responded with HTTP ${code} in ${elapsed}.`;
      }

      case 'wave_completed':
        return `${prefix} [WAVE ${event.wave || 1}] 🏁 Wave completed. (${event.completed_tasks || 0} task outcomes received).`;

      case 'synthesizing':
        return `${prefix} [SYNTHESIS] 🧬 Merging live 3rd-party data into unified structured response...`;

      case 'synthesis_completed':
        return `${prefix} [SYNTHESIS] 🎯 Synthesis complete. apis_used: [${(event.apis_used || []).join(', ')}]`;

      case 'completed': {
        const total = event.total_time_seconds ? ` in ${event.total_time_seconds}s` : '';
        return `${prefix} [DONE] 🎉 Pipeline finished successfully${total}.`;
      }

      case 'error':
        return `${prefix} [ERROR] 💥 ${event.message || 'Execution error.'}`;

      default:
        return event.message ? `${prefix} [INFO] ${event.message}` : '';
    }
  }

  // =========================================================================
  // 5. Animated Node Execution Graph State Reducer
  // =========================================================================

  /**
   * Updates an animated node execution graph state based on incoming stream events.
   * Useful for React Flow, D3, Canvas, or CSS node graph visualizations.
   *
   * @param {Object} currentState - Current graph state map: { [nodeId]: { status, message, data } }
   * @param {Object} event - Event emitted from WebSocket or SSE
   * @returns {Object} Updated node state map
   */
  function deriveGraphState(currentState = {}, event = {}) {
    const nextState = { ...currentState };
    const step = event.step;

    function setNode(nodeId, status, label, extra = {}) {
      nextState[nodeId] = {
        id: nodeId,
        label: label || nodeId,
        status, // 'idle' | 'running' | 'success' | 'warning' | 'failed' | 'healed'
        lastUpdated: Date.now(),
        ...extra,
      };
    }

    switch (step) {
      case 'inspecting_key':
        setNode('vault_inspector', 'running', 'Vault Inspector', { message: 'Checking keys...' });
        break;

      case 'keys_inspected':
        setNode('vault_inspector', 'success', 'Vault Inspector', {
          message: `${event.active_connectors_count || 0} keys active`,
          connectors: event.connectors,
        });
        break;

      case 'planning_started':
        setNode('planner', 'running', 'Planner Agent', { message: 'Formulating wave plan...' });
        break;

      case 'plan_generated':
        setNode('planner', 'success', 'Planner Agent', {
          message: `${event.tasks_count || 0} tasks planned`,
          plan: event.plan,
        });
        break;

      case 'wave_started':
        setNode(`wave_${event.wave || 1}`, 'running', `Wave ${event.wave || 1}`, {
          tasksCount: event.tasks_count,
        });
        break;

      case 'calling_3rd_party_api': {
        const provNode = `node_${event.provider || 'api'}`;
        setNode(provNode, 'running', event.provider || 'API Call', {
          method: event.method,
          endpoint: event.endpoint,
          payload: event.payload,
        });
        break;
      }

      case 'rate_limit_backoff':
        setNode('resilience_engine', 'warning', 'Rate-Limit Engine', {
          message: `Backoff ${event.delay_seconds}s (429)`,
        });
        break;

      case 'self_healing_started':
        setNode('self_healing', 'running', 'AI Self-Healing', {
          message: 'Repairing invalid parameters...',
        });
        break;

      case 'self_healing_success':
        setNode('self_healing', 'healed', 'AI Self-Healing', {
          message: 'Parameters healed & retried!',
        });
        break;

      case 'api_response_received': {
        const provNode = `node_${event.provider || 'api'}`;
        const nodeStatus = event.success ? 'success' : 'failed';
        setNode(provNode, nodeStatus, event.provider || 'API Call', {
          statusCode: event.status_code,
          elapsedSeconds: event.elapsed_seconds,
          selfHealed: event.self_healed,
        });
        break;
      }

      case 'wave_completed':
        setNode(`wave_${event.wave || 1}`, 'success', `Wave ${event.wave || 1}`, {
          message: 'Wave completed',
        });
        break;

      case 'synthesizing':
        setNode('synthesizer', 'running', 'Synthesizer Agent', {
          message: 'Synthesizing live responses...',
        });
        break;

      case 'synthesis_completed':
        setNode('synthesizer', 'success', 'Synthesizer Agent', {
          message: 'Synthesis complete',
          apisUsed: event.apis_used,
        });
        break;

      case 'completed':
        setNode('pipeline', 'success', 'Complete Pipeline', {
          data: event.data,
          totalTime: event.total_time_seconds,
        });
        break;

      case 'error':
        setNode('pipeline', 'failed', 'Pipeline Error', {
          message: event.message,
        });
        break;
    }

    return nextState;
  }

  // =========================================================================
  // Singleton Helpers & Exports
  // =========================================================================

  const defaultClient = new ApiKitchenClient();
  let defaultWs = null;

  function getDefaultWebSocket() {
    if (!defaultWs) {
      defaultWs = new ExecutionWebSocket();
    }
    return defaultWs;
  }

  return {
    // Configuration
    configure,
    getConfig,

    // REST API Helpers
    ApiKitchenClient,
    registerKey: (params) => defaultClient.registerKey(params),
    listKeys: (sessionId) => defaultClient.listKeys(sessionId),
    getKey: (provider, sessionId) => defaultClient.getKey(provider, sessionId),
    getKeySchema: (provider, sessionId) => defaultClient.getKeySchema(provider, sessionId),
    deleteKey: (provider, sessionId) => defaultClient.deleteKey(provider, sessionId),
    clearSession: (sessionId) => defaultClient.clearSession(sessionId),
    seedDemo: (options) => defaultClient.seedDemo(options),
    checkHealth: () => defaultClient.checkHealth(),

    // WebSocket Helpers
    ExecutionWebSocket,
    createExecutionWebSocket: (options) => new ExecutionWebSocket(options),
    executePrompt: (prompt, options) => getDefaultWebSocket().executeAsync(prompt, options),

    // SSE Helper
    streamExecutionSSE,

    // UI Helpers
    formatTerminalLog,
    deriveGraphState,
    EventEmitter,
  };
});
