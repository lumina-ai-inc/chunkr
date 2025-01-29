interface Env {
	MY_RATE_LIMITER: any;
}

export default {
	async fetch(request, env): Promise<Response> {
		const { pathname } = new URL(request.url)

		// Only apply rate limiting for POST requests to /api/v1/task
		if (pathname === '/api/v1/task' && request.method === 'POST') {
			const authHeader = request.headers.get('Authorization')
			if (!authHeader) {
				return new Response('Authorization header required', { status: 401 })
			}

			const { success } = await env.MY_RATE_LIMITER.limit({ key: `${pathname}:${authHeader}` })
			if (!success) {
				return new Response(`429 Failure – rate limit exceeded for this authorization token`, { status: 429 })
			}
		}

		return new Response(`Success!`)
	}
} satisfies ExportedHandler<Env>;