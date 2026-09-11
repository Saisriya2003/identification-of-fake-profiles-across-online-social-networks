export default function Gauge({ score = 0, status = "loading" }) {
  const pct = Math.round(Math.max(0, Math.min(1, score)) * 100);
  const radius = 86;
  const circ = 2 * Math.PI * radius;
  const offset = circ * (1 - pct / 100);
  const tone =
    pct >= 65 ? "var(--sage)" : pct <= 35 ? "var(--danger)" : "var(--copper)";

  return (
    <svg className="gauge" viewBox="0 0 200 200" role="img" aria-label={`Authenticity ${pct}`}>
      <circle cx="100" cy="100" r={radius} fill="none" stroke="var(--line)" strokeWidth="10" />
      <circle
        cx="100"
        cy="100"
        r={radius}
        fill="none"
        stroke={tone}
        strokeWidth="10"
        strokeLinecap="round"
        strokeDasharray={circ}
        strokeDashoffset={status === "error" ? circ : offset}
        transform="rotate(-90 100 100)"
      />
      <text
        x="100"
        y="96"
        textAnchor="middle"
        fill="var(--ink)"
        fontFamily="Fraunces, Georgia, serif"
        fontSize="42"
        fontWeight="650"
      >
        {status === "loading" ? "—" : pct}
      </text>
      <text
        x="100"
        y="122"
        textAnchor="middle"
        fill="var(--muted)"
        fontFamily="Outfit, sans-serif"
        fontSize="11"
        letterSpacing="2"
      >
        AUTHENTICITY
      </text>
    </svg>
  );
}
