import { useEffect, useMemo, useState } from "react";
import { getSamples, predictProfile } from "../api.js";
import { deriveIdentity, formatStat, platformLabel } from "../features.js";

function initialFrom(name) {
  return (name || "?").slice(0, 1).toUpperCase();
}

export default function Gallery({ onInspect }) {
  const [rows, setRows] = useState([]);
  const [filter, setFilter] = useState("all");
  const [platform, setPlatform] = useState("all");
  const [scores, setScores] = useState({});
  const [status, setStatus] = useState("loading");
  const [error, setError] = useState("");
  const [pending, setPending] = useState("");

  useEffect(() => {
    let cancelled = false;
    setStatus("loading");
    Promise.all([getSamples("real", 8), getSamples("fake", 8)])
      .then(([real, fake]) => {
        if (cancelled) return;
        setRows([...(real.samples || []), ...(fake.samples || [])]);
        setStatus("ready");
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err.message || "Could not load the sample gallery.");
        setStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const visible = useMemo(() => {
    return rows.filter((row) => {
      if (filter !== "all" && row.label !== filter) return false;
      if (platform !== "all" && row.platform !== platform) return false;
      return true;
    });
  }, [rows, filter, platform]);

  async function scoreRow(row) {
    const key = row.username;
    setPending(key);
    try {
      const result = await predictProfile(row);
      setScores((current) => ({ ...current, [key]: result }));
    } catch (err) {
      setScores((current) => ({
        ...current,
        [key]: { error: err.message || "Score failed" },
      }));
    } finally {
      setPending("");
    }
  }

  return (
    <section>
      <div className="gallery-head">
        <div>
          <p className="eyebrow">Reference set</p>
          <h2>Sample profiles</h2>
          <p className="page-lead" style={{ marginBottom: 0 }}>
            Synthetic accounts that mimic public Facebook, Instagram, and Twitter
            signals. Click a card to score the inbound request.
          </p>
        </div>
        <div className="filters">
          {[
            ["all", "All"],
            ["real", "Authentic"],
            ["fake", "Flagged"],
          ].map(([value, label]) => (
            <button
              key={value}
              type="button"
              className={filter === value ? "active" : ""}
              onClick={() => setFilter(value)}
            >
              {label}
            </button>
          ))}
          {["all", "facebook", "instagram", "twitter"].map((value) => (
            <button
              key={value}
              type="button"
              className={platform === value ? "active" : ""}
              onClick={() => setPlatform(value)}
            >
              {value === "all" ? "Any network" : platformLabel(value)}
            </button>
          ))}
        </div>
      </div>

      {status === "loading" && <div className="banner loading">Loading seeded profiles…</div>}
      {status === "error" && <div className="banner error">{error}</div>}
      {status === "ready" && visible.length === 0 && (
        <p className="empty">No profiles match this filter.</p>
      )}

      <div className="grid">
        {visible.map((row) => {
          const scored = scores[row.username];
          return (
            <article className="card" key={row.username}>
              <div className="card-top">
                <div className="avatar">{initialFrom(row.username)}</div>
                <div>
                  <h3>{row.username}</h3>
                  <div className="plat">
                    {platformLabel(row.platform)} · {row.label === "fake" ? "held out as fake" : "held out as authentic"}
                  </div>
                </div>
              </div>
              <div className="stats">
                <div>
                  Followers <strong>{formatStat(row.follower_count)}</strong>
                </div>
                <div>
                  Following <strong>{formatStat(row.following_count)}</strong>
                </div>
                <div>
                  Posts <strong>{formatStat(row.posts_count)}</strong>
                </div>
                <div>
                  Age <strong>{formatStat(row.account_age_days)}d</strong>
                </div>
              </div>
              {scored?.score != null && (
                <div className="card-score">
                  <span className="score-num">{Math.round(scored.score * 100)}</span>
                  <span className="meta">{scored.verdict}</span>
                </div>
              )}
              {scored?.error && <p className="meta">{scored.error}</p>}
              <div className="actions">
                <button
                  type="button"
                  className="btn primary"
                  disabled={pending === row.username}
                  onClick={() => scoreRow(row)}
                >
                  {pending === row.username ? "Scoring…" : "Score request"}
                </button>
                <button
                  type="button"
                  className="btn ghost"
                  onClick={() => onInspect(deriveIdentity(row))}
                >
                  Inspect signals
                </button>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
