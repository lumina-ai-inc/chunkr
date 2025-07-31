interface Env {
	MY_RATE_LIMITER: any;
}

export default {
	async fetch(request: Request, env: Env): Promise<Response> {
		const url = new URL(request.url)
		const { pathname } = url

		// Rate limiting for POST requests to task endpoint across all API versions
		if (pathname.match(/^\/api\/v\d+\/task$/) && request.method === 'POST') {
			const authHeader = request.headers.get('Authorization')
			if (!authHeader) {
				return new Response('Authorization header required', { status: 401 })
			}

			const { success } = await env.MY_RATE_LIMITER.limit({ key: `task:${authHeader}` })
			if (!success) {
				return new Response(`429 Failure – rate limit exceeded for this authorization token`, { status: 429 })
			}
		}

		// Handle API versioning by stripping version prefix from path
		// Only reroute if the origin server doesn't already handle versioned paths
		const versionMatch = pathname.match(/^\/api\/(v\d+)\/(.*)$/)
		if (versionMatch) {
			const [, version, restPath] = versionMatch

			// Check if we should handle version stripping at worker level
			// Skip rerouting for v1 if backend already handles it
			const shouldReroute = version !== 'v1' // Kube handles v1, we handle v2+

			if (shouldReroute) {
				// Strip version prefix from path
				url.pathname = `/${restPath}`

				const modifiedRequest = new Request(url.toString(), {
					method: request.method,
					headers: request.headers,
					body: request.body,
				})

				// Forward request to origin server
				return fetch(modifiedRequest)
			}
		}

		// Pass through non-versioned paths unchanged
		return fetch(request)
	}
} satisfies ExportedHandler<Env>;