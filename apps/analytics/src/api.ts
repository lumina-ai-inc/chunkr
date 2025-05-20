const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
import { useQuery } from '@tanstack/react-query'

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
    timeoutId = window.setTimeout(() => controller?.abort(), timeout)
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

// React Query hooks
export const useLifetimePages = (apiKey: string, enabled = true) => {
  return useQuery({
    queryKey: ['lifetimePages'],
    queryFn: () => apiCall<number>('/lifetime-pages', {
      headers: { 'Authorization': `Bearer ${apiKey}` }
    }),
    enabled: !!apiKey && enabled
  })
}

export const usePagesPerDay = (start: string, end: string, email: string | undefined, apiKey: string, enabled = true) => {
  const emailParam = email ? `&email=${encodeURIComponent(email)}` : ''
  return useQuery({
    queryKey: ['pagesPerDay', start, end, email],
    queryFn: () => apiCall<any[]>(`/pages-per-day?start=${start}&end=${end}${emailParam}`, {
      headers: { 'Authorization': `Bearer ${apiKey}` }
    }),
    enabled: !!apiKey && enabled
  })
}

export const useStatusBreakdown = (start: string, end: string, email: string | undefined, apiKey: string, enabled = true) => {
  const emailParam = email ? `&email=${encodeURIComponent(email)}` : ''
  return useQuery({
    queryKey: ['statusBreakdown', start, end, email],
    queryFn: () => apiCall<any[]>(`/status-breakdown?start=${start}&end=${end}${emailParam}`, {
      headers: { 'Authorization': `Bearer ${apiKey}` }
    }),
    enabled: !!apiKey && enabled
  })
}

export const useTopUsers = (start: string, end: string, apiKey: string, enabled = true) => {
  return useQuery({
    queryKey: ['topUsers', start, end],
    queryFn: () => apiCall<any[]>('/top-users', {
      method: 'POST',
      body: { start, end, limit: 5 },
      headers: { 'Authorization': `Bearer ${apiKey}` }
    }),
    enabled: !!apiKey && enabled
  })
}

export const useTaskDetails = (start: string, end: string, email: string | undefined, apiKey: string, enabled = true) => {
  const emailParam = email ? `&email=${encodeURIComponent(email)}` : ''
  return useQuery({
    queryKey: ['taskDetails', start, end, email],
    queryFn: () => apiCall<any[]>(`/task-details?start=${start}&end=${end}${emailParam}`, {
      headers: { 'Authorization': `Bearer ${apiKey}` }
    }),
    enabled: !!apiKey && enabled
  })
}

export const useValidateApiKey = (apiKey: string | null) => {
  return useQuery({
    queryKey: ['validateApiKey', apiKey],
    queryFn: async () => {
      if (!apiKey) throw new Error('No API key provided')
      try {
        await apiCall<number>('/lifetime-pages', {
          headers: { 'Authorization': `Bearer ${apiKey}` }
        })
        return true
      } catch (err) {
        console.error('Invalid API Key:', err)
        throw new Error('Invalid API Key')
      }
    },
    enabled: !!apiKey,
    retry: false
  })
}