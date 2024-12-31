import axiosInstance from "./axios.config";

interface GithubRepoStats {
  stars: number;
  forks: number;
}

export async function getRepoStats(): Promise<GithubRepoStats> {
  const data = await axiosInstance.get("/github").then((res) => ({
    stars: res.data.stargazers_count,
    forks: res.data.forks_count,
  }));
  return data;
}
