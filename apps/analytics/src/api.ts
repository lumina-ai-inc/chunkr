const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
type ApiOptions = {
    method?: string
    body?: object
    headers?: Record<string, string>
  }
export async function apiCall<T>(route: string, options: ApiOptions = {}): Promise<T> {
  const { method = 'GET', body, headers = {} } = options
  
  const apiKey = localStorage.getItem('apiKey')
  if (!apiKey && !headers['Authorization']) {
    throw new Error('No API key found')
  }

  const response = await fetch(`${API_URL}${route}`, {
    method,
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
      ...headers
    },
    ...(body && { body: JSON.stringify(body) })
  })

  if (!response.ok) {
    throw new Error(`API call failed: ${response.statusText}`)
  }

  return response.json()
}