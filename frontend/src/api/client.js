import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || '';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120_000, // 2 min — ML inference can be slow
});

/** POST /api/analyze — analyze a single X-ray */
export async function analyzeXray(file, sample) {
  const form = new FormData();
  if (file) form.append('file', file);
  if (sample) form.append('sample', sample);

  const { data } = await api.post('/api/analyze', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

/** POST /api/compare — compare two X-rays */
export async function compareXrays(opts = {}) {
  const form = new FormData();
  if (opts.file1) form.append('file1', opts.file1);
  if (opts.file2) form.append('file2', opts.file2);
  if (opts.use_demo1) form.append('use_demo1', 'true');
  if (opts.use_demo2) form.append('use_demo2', 'true');

  const { data } = await api.post('/api/compare', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

/** GET /api/health */
export async function fetchHealth() {
  const { data } = await api.get('/api/health');
  return data;
}

/** GET /api/info */
export async function fetchModelInfo() {
  const { data } = await api.get('/api/info');
  return data;
}

/** GET /api/export/:fmt — returns a Blob for download */
export async function exportReport(fmt) {
  const { data } = await api.get(`/api/export/${fmt}`, { responseType: 'blob' });
  return data;
}
