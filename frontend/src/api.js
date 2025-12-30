import axios from "axios";

//const API_URL = "http://127.0.0.1:8000"; // development
const API_URL = "/webmir/api";          // production

const API = axios.create({
  baseURL: API_URL,
});

export const register = (username, password) =>
  axios.post(`${API_URL}/register`, { username, password });

export const login = (username, password) =>
  axios.post(`${API_URL}/login`, { username, password });

export const getAdminData = (token) =>
  axios.get(`${API_URL}/admin`, { headers: { Authorization: `Bearer ${token}` } });

export const getManuscripts = () =>
  API.get("/manuscripts");

export const getChants = (ms_id) =>
  API.post("/search", { query: ms_id });

export const enqueue = (url, token) =>
  API.post("/enqueue", { url }, { headers: { Authorization: `Bearer ${token}` } });
