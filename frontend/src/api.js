const API_BASE = import.meta.env.VITE_API_URL ?? "";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    const detail = payload?.detail || `Request failed (${response.status})`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload;
}

export function getHealth() {
  return request("/api/health");
}

export function getMetrics() {
  return request("/api/metrics");
}

export function getPlatforms() {
  return request("/api/platforms");
}

export function getSamples(label, limit = 8) {
  const query = new URLSearchParams();
  if (label) query.set("label", label);
  query.set("limit", String(limit));
  return request(`/api/samples?${query.toString()}`);
}

export function predictProfile(body) {
  return request("/api/predict", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export { API_BASE };
