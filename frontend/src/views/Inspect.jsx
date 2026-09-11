import { useEffect, useMemo, useRef, useState } from "react";
import Gauge from "../components/Gauge.jsx";
import { predictProfile } from "../api.js";
import { DEFAULT_PROFILE, FEATURE_FIELDS, deriveIdentity, formatStat } from "../features.js";

const SUSPICIOUS = deriveIdentity({
  username: "promo_deal84721",
  platform: "instagram",
  follower_count: 38,
  following_count: 4200,
  friends_count: 12,
  posts_count: 3,
  account_age_days: 11,
  has_profile_photo: 0,
  has_cover_photo: 0,
  bio_length: 8,
  friend_request_mutuals: 0,
  avg_likes_per_post: 1,
  posts_per_week: 7.5,
});

function verdictClass(verdict) {
  if (verdict === "Likely authentic") return "authentic";
  if (verdict === "Likely fake") return "fake";
  return "suspicious";
}

export default function Inspect({ draft, setDraft }) {
  const profile = useMemo(() => deriveIdentity(draft), [draft]);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("loading");
  const [error, setError] = useState("");
  const timer = useRef(null);

  useEffect(() => {
    setStatus("loading");
    setError("");
    clearTimeout(timer.current);
    timer.current = setTimeout(async () => {
      try {
        const payload = await predictProfile(profile);
        setResult(payload);
        setStatus("ready");
      } catch (err) {
        setResult(null);
        setStatus("error");
        setError(err.message || "Signal service unreachable.");
      }
    }, 220);
    return () => clearTimeout(timer.current);
  }, [profile]);

  function patch(partial) {
    setDraft((current) => deriveIdentity({ ...current, ...partial }));
  }

  return (
    <section>
      <p className="page-lead">
        Score an inbound request before you accept. The model reads public graph and
        profile-completeness signals — not private messages, not scraped identities.
      </p>
      {status === "error" && <div className="banner error">{error}</div>}
      {status === "loading" && <div className="banner loading">Scoring this request…</div>}

      <div className="inspect">
        <div className="panel">
          <p className="eyebrow">Inbound request</p>
          <h2>Profile signals</h2>
          <p className="meta">Adjust the public footprint. Completeness updates as you edit.</p>

          <div className="field" style={{ marginTop: 18 }}>
            <label htmlFor="username">
              Username
              <span className="value">
                {Math.round(profile.username_digit_ratio * 100)}% digits
              </span>
            </label>
            <input
              id="username"
              type="text"
              value={profile.username}
              onChange={(event) => patch({ username: event.target.value })}
            />
          </div>

          <div className="field">
            <label htmlFor="platform">Platform</label>
            <select
              id="platform"
              value={profile.platform}
              onChange={(event) => patch({ platform: event.target.value })}
            >
              <option value="facebook">Facebook</option>
              <option value="instagram">Instagram</option>
              <option value="twitter">Twitter</option>
            </select>
          </div>

          <div className="toggles">
            <button
              type="button"
              className={`toggle ${profile.has_profile_photo ? "on" : ""}`}
              onClick={() => patch({ has_profile_photo: profile.has_profile_photo ? 0 : 1 })}
            >
              <span>
                <strong>Profile photo</strong>
                Present on the account
              </span>
              <span>{profile.has_profile_photo ? "On" : "Off"}</span>
            </button>
            <button
              type="button"
              className={`toggle ${profile.has_cover_photo ? "on" : ""}`}
              onClick={() => patch({ has_cover_photo: profile.has_cover_photo ? 0 : 1 })}
            >
              <span>
                <strong>Cover photo</strong>
                Banner or header image
              </span>
              <span>{profile.has_cover_photo ? "On" : "Off"}</span>
            </button>
          </div>

          {["identity", "graph", "activity"].map((group) => (
            <div className="group" key={group}>
              {FEATURE_FIELDS.filter((field) => field.group === group).map((field) => (
                <div className="field" key={field.key}>
                  <label htmlFor={field.key}>
                    {field.label}
                    <span className="value">{formatStat(profile[field.key], field.key)}</span>
                  </label>
                  <input
                    id={field.key}
                    type="range"
                    min={field.min}
                    max={field.max}
                    step={field.step}
                    value={profile[field.key]}
                    onChange={(event) => patch({ [field.key]: Number(event.target.value) })}
                  />
                </div>
              ))}
            </div>
          ))}

          <div className="field">
            <label>
              Profile completeness
              <span className="value">{formatStat(profile.profile_completeness, "profile_completeness")}</span>
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={profile.profile_completeness}
              readOnly
              aria-readonly="true"
            />
          </div>

          <div className="actions">
            <button type="button" className="btn primary" onClick={() => setDraft(DEFAULT_PROFILE)}>
              Typical authentic
            </button>
            <button type="button" className="btn ghost" onClick={() => setDraft(SUSPICIOUS)}>
              Suspicious pattern
            </button>
          </div>
        </div>

        <div className="panel">
          <p className="eyebrow">Trust decision</p>
          <h2>Authenticity</h2>
          <div className="gauge-wrap">
            <Gauge score={result?.score ?? 0} status={status} />
            <p className={`verdict ${verdictClass(result?.verdict)}`}>
              {result?.verdict || (status === "error" ? "Unavailable" : "Reading signals")}
            </p>
            <p className="meta">
              {result
                ? `Confidence ${Math.round(result.confidence * 100)}% · P(fake) ${result.p_fake.toFixed(2)}`
                : "Live score from the sigmoid network."}
            </p>
          </div>
          <div className="chips">
            {result?.explanation?.length ? (
              result.explanation.map((item) => (
                <span key={item.feature} className={`chip ${item.direction}`}>
                  {item.label} · {item.direction === "fake" ? "toward fake" : "supports authentic"}
                </span>
              ))
            ) : (
              <span className="chip">Explanation appears after the first score.</span>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
