import { useEffect, useState } from "react";
import Sparkline from "../components/Sparkline.jsx";
import { getMetrics, getPlatforms } from "../api.js";

function Layer({ width, name, output }) {
  const dots = Math.min(width, 8);
  return (
    <div className="layer">
      <div className="stack">
        {Array.from({ length: dots }, (_, index) => (
          <span key={index} className={`dot ${output ? "out" : ""}`} />
        ))}
      </div>
      <strong>{width}</strong>
      <small>{name}</small>
    </div>
  );
}

export default function ModelLab() {
  const [metrics, setMetrics] = useState(null);
  const [platforms, setPlatforms] = useState(null);
  const [status, setStatus] = useState("loading");
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    setStatus("loading");
    Promise.all([getMetrics(), getPlatforms()])
      .then(([m, p]) => {
        if (cancelled) return;
        setMetrics(m);
        setPlatforms(p);
        setStatus("ready");
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err.message || "Could not load training artifacts.");
        setStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const layers = metrics?.architecture || [14, 16, 8, 1];
  const names = ["Input", "Hidden", "Hidden", "Sigmoid"];
  const cm = metrics?.confusion_matrix;

  return (
    <section className="lab">
      <div>
        <p className="eyebrow">From scratch</p>
        <h2>Model lab</h2>
        <p className="page-lead">
          A NumPy artificial neural network — sigmoid activations, binary
          cross-entropy, and backpropagation with mini-batch gradient descent.
          Weights were refined on synthetic Facebook, Instagram, and Twitter-style
          public signals.
        </p>
      </div>

      {status === "loading" && <div className="banner loading">Reading metrics.json…</div>}
      {status === "error" && <div className="banner error">{error}</div>}

      <div className="panel">
        <p className="eyebrow">Architecture</p>
        <h2>{layers.join(" · ")}</h2>
        <div className="layers">
          {layers.map((width, index) => (
            <div key={`${width}-${index}`} style={{ display: "contents" }}>
              {index > 0 && <div className="connector" />}
              <Layer width={width} name={names[index] || "Layer"} output={index === layers.length - 1} />
            </div>
          ))}
        </div>
        <p className="meta" style={{ marginTop: 12 }}>
          {metrics
            ? `${metrics.optimizer} · lr ${metrics.learning_rate} · ${metrics.epochs} epochs · batch ${metrics.batch_size}`
            : "Architecture loads with the trained artifact."}
        </p>
      </div>

      <div className="metrics-row">
        {[
          ["Accuracy", metrics?.accuracy],
          ["Precision", metrics?.precision],
          ["Recall", metrics?.recall],
          ["F1", metrics?.f1],
        ].map(([label, value]) => (
          <div className="metric" key={label}>
            <span>{label}</span>
            <b>{value == null ? "—" : value.toFixed(3)}</b>
          </div>
        ))}
      </div>

      <div className="panel">
        <p className="eyebrow">Held-out test</p>
        <h2>Confusion matrix</h2>
        <p className="meta">Positive class is fake — the inbound request we want to catch.</p>
        {cm ? (
          <div className="matrix" style={{ marginTop: 16 }}>
            <div />
            <div className="axis">Predicted authentic</div>
            <div className="axis">Predicted fake</div>
            <div className="axis">Actual authentic</div>
            <div className="cell">
              <b>{cm.tn}</b>
              <span className="meta">TN</span>
            </div>
            <div className="cell">
              <b>{cm.fp}</b>
              <span className="meta">FP</span>
            </div>
            <div className="axis">Actual fake</div>
            <div className="cell">
              <b>{cm.fn}</b>
              <span className="meta">FN</span>
            </div>
            <div className="cell">
              <b>{cm.tp}</b>
              <span className="meta">TP</span>
            </div>
          </div>
        ) : (
          <p className="empty">Confusion matrix unavailable.</p>
        )}
      </div>

      <div className="panel">
        <p className="eyebrow">Training</p>
        <h2>Binary cross-entropy</h2>
        <Sparkline values={metrics?.loss_history || []} />
        {platforms && (
          <p className="meta">
            Corpus {platforms.total.toLocaleString()} profiles ·{" "}
            {Object.entries(platforms.by_platform)
              .map(([name, count]) => `${name} ${count}`)
              .join(" · ")}
          </p>
        )}
      </div>
    </section>
  );
}
