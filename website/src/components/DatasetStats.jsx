import React from 'react';

const STATS = [
  {
    symbol: 'N',
    value: '43 019',
    label: 'Spectres retenus',
    sub: 'LAMOST DR5 filtrés',
    color: '#F59E0B',
  },
  {
    symbol: 'p',
    value: '183',
    label: 'Descripteurs',
    sub: 'spectroscopiques purs',
    color: '#38BDF8',
  },
  {
    symbol: 'λ',
    value: '3690–9100',
    label: 'Couverture spectrale',
    sub: 'Ångströms',
    color: '#A78BFA',
  },
  {
    symbol: 'R',
    value: '≈ 1800',
    label: 'Résolution',
    sub: 'basse résolution',
    color: '#34D399',
  },
  {
    symbol: '⊕',
    value: '1″',
    label: 'Croisement Gaia',
    sub: 'rayon angulaire',
    color: '#FB7185',
  },
];

function StatCard({ symbol, value, label, sub, color }) {
  const [hovered, setHovered] = React.useState(false);

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        position: 'relative',
        background: 'var(--ifm-card-background-color)',
        border: `1px solid ${hovered ? color + '55' : color + '22'}`,
        borderTop: `3px solid ${color}`,
        borderRadius: '10px',
        padding: '18px 16px 14px',
        overflow: 'hidden',
        transition: 'transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease',
        transform: hovered ? 'translateY(-3px)' : 'translateY(0)',
        boxShadow: hovered ? `0 10px 28px ${color}20` : '0 2px 8px rgba(0,0,0,0.08)',
        cursor: 'default',
      }}
    >
      {/* Symbol watermark */}
      <div
        style={{
          position: 'absolute',
          top: '10px',
          right: '14px',
          fontFamily: 'Georgia, "Times New Roman", serif',
          fontStyle: 'italic',
          fontSize: '20px',
          color: color,
          opacity: hovered ? 0.55 : 0.3,
          transition: 'opacity 0.18s ease',
          userSelect: 'none',
          lineHeight: 1,
        }}
      >
        {symbol}
      </div>

      {/* Value */}
      <div
        style={{
          fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Courier New', monospace",
          fontSize: '26px',
          fontWeight: '800',
          color: color,
          lineHeight: 1.05,
          marginBottom: '8px',
          letterSpacing: '-0.5px',
        }}
      >
        {value}
      </div>

      {/* Label */}
      <div
        style={{
          fontSize: '12px',
          fontWeight: '700',
          color: 'var(--ifm-font-color-base)',
          marginBottom: '3px',
          textTransform: 'uppercase',
          letterSpacing: '0.06em',
        }}
      >
        {label}
      </div>

      {/* Sub */}
      <div
        style={{
          fontSize: '11px',
          color: 'var(--ifm-font-color-base)',
          opacity: 0.5,
        }}
      >
        {sub}
      </div>

      {/* Bottom accent line */}
      <div
        style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          height: '2px',
          background: `linear-gradient(90deg, ${color}00, ${color}44, ${color}00)`,
          opacity: hovered ? 1 : 0,
          transition: 'opacity 0.18s ease',
        }}
      />
    </div>
  );
}

export default function DatasetStats() {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(155px, 1fr))',
        gap: '12px',
        margin: '28px 0',
      }}
    >
      {STATS.map((s) => (
        <StatCard key={s.label} {...s} />
      ))}
    </div>
  );
}
