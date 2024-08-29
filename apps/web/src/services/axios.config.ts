import axios from "axios";

const axiosInstance = axios.create({
  baseURL: import.meta.env.VITE_API_URL,
  headers: {
    "x-api-key": import.meta.env.VITE_API_KEY,
  },
});

export default axiosInstance;
