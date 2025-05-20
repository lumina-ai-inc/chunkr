const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
type ApiOptions = {
  method?: string
  body?: object
  headers?: Record<string, string>
  timeout?: number
  signal?: AbortSignal
}

export class ApiError extends Error {
  status: number
  statusText: string

  constructor(status: number, statusText: string, message: string) {
    super(message)
    this.status = status
    this.statusText = statusText
  }
}

export async function apiCall<T>(route: string, options: ApiOptions = {}): Promise<T> {
  const { method = 'GET', body, headers = {}, timeout, signal } = options

  const apiKey = localStorage.getItem('apiKey')
  if (!apiKey && !headers['Authorization']) {
    throw new Error('No API key found')
  }

  // Create an abort controller for timeouts if needed
  let controller: AbortController | undefined
  let timeoutId: number | undefined

  if (timeout && !signal) {
    controller = new AbortController()
    timeoutId = window.setTimeout(() => controller.abort(), timeout)
  }

  try {
    const response = await fetch(`${API_URL}${route}`, {
      method,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
        ...headers
      },
      ...(body && { body: JSON.stringify(body) }),
      signal: signal || controller?.signal
    })

    if (!response.ok) {
      throw new ApiError(
        response.status,
        response.statusText,
        `API call failed: ${response.statusText}`
      )
    }

    return response.json()
  } catch (error: any) {
    if (error.name === 'AbortError') {
      throw new Error('Request timed out')
    }
    throw error
  } finally {
    if (timeoutId !== undefined) {
      window.clearTimeout(timeoutId)
    }
  }
}

const validateApiKey = async () => {
  if (!apiKey) return
  try {
    // Use an endpoint that actually exists
    await apiCall<number>('/lifetime-pages', {
      headers: { 'Authorization': `Bearer ${apiKey}` }
    })
    localStorage.setItem('apiKey', apiKey)
    setIsValidApiKey(true)
  } catch (err) {
    console.error('Invalid API Key:', err)
    localStorage.removeItem('apiKey')
    setIsValidApiKey(false)
    alert("Invalid API Key provided.")
  }
}