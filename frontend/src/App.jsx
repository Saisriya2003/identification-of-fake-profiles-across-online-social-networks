import { useState } from "react";
import Inspect from "./views/Inspect.jsx";
import Gallery from "./views/Gallery.jsx";
import ModelLab from "./views/ModelLab.jsx";
import { DEFAULT_PROFILE } from "./features.js";

const VIEWS = [
  ["inspect", "Inspect"],
  ["gallery", "Gallery"],
  ["lab", "Model lab"],
];

export default function App() {
  const [view, setView] = useState("inspect");
  const [draft, setDraft] = useState(DEFAULT_PROFILE);

  function openInspect(profile) {
    setDraft(profile);
    setView("inspect");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <h1>Identification of Fake Profiles Across Online Social Networks</h1>
          <p>Artificial neural network scoring of friend-request authenticity</p>
        </div>
        <nav className="nav" aria-label="Primary">
          {VIEWS.map(([id, label]) => (
            <button
              key={id}
              type="button"
              className={view === id ? "active" : ""}
              onClick={() => setView(id)}
            >
              {label}
            </button>
          ))}
        </nav>
      </header>

      {view === "inspect" && <Inspect draft={draft} setDraft={setDraft} />}
      {view === "gallery" && <Gallery onInspect={openInspect} />}
      {view === "lab" && <ModelLab />}

      <footer className="foot">
        This demo scores synthetic public-profile statistics. No personal data
        was scraped. Demo runs without API keys.
      </footer>
    </div>
  );
}
