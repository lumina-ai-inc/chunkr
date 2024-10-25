interface GithubRepoStats {
  stars: number;
  forks: number;
}

export async function getRepoStats(
  owner: string,
  repo: string
): Promise<GithubRepoStats> {
  const response = await fetch(
    `https://api.github.com/repos/${owner}/${repo}`,
    {
      headers: {
        Accept: "application/vnd.github.v3+json",
      },
    }
  );

  if (!response.ok) {
    throw new Error(`GitHub API request failed: ${response.statusText}`);
  }

  const data = await response.json();

  return {
    stars: data.stargazers_count,
    forks: data.forks_count,
  };
}
