import React from 'react';

const METRICS = [
  {
    value: '0.489',
    label: 'MSE de reconstruction',
    sub: 'autoencodeur z = 2 · N = 43 019',
    color: '#34D399',
    detail: 'vs PCA(z=2) = 0.696 · −29,8 %',
    symbol: 'MSE',
    highlight: true,
  },
  {
    value: '≈ PCA@10',
    label: 'Équivalence en compression',
    sub: 'MSE(AE, z=2) ≈ MSE(PCA, k=10)',
    color: '#38BDF8',
    detail: '5× moins de dimensions que PCA',
    symbol: 'z=2',
  },
  {
    value: '145',
    label: 'Époques d\'entraînement',
    sub: 'Best val loss = 0,548 · 143,7 s',
    color: '#F59E0B',
    detail: 'Early stopping · Ryzen 9 5950X CPU',
    symbol: 'ep.',
  },
  {
    value: '107×',
    label: 'Erreur QSO vs étoiles',
    sub: 'MSE(QSO) / MSE(STAR) médiane',
    color: '#C084FC',
    detail: 'Détection anomalie non supervisée',
    symbol: '⚡',
  },
];

export default function AEKeyMetrics() {
  return (
    <div style={{ display: 'flex', gap: '12px', margin: '28px 0', flexWrap: 'wrap' }}>
      {METRICS.map((m, i) => {
        const [hov, setHov] = React.useState(false);
        return (
          <div
            key={i}
            onMouseEnter={() => setHov(true)}
            onMouseLeave={() => setHov(false)}
            style={{
              flex: 1,
              background: hov ? `${m.color}10` : m.highlight ? `${m.color}06` : 'var(--ifm-card-background-color)',
              border: `1px solid ${hov ? m.color + '55' : m.highlight ? m.color + '35' : m.color + '22'}`,
              borderTop: `4px solid ${m.color}`,
              borderRadius: '12px',
              padding: '20px',
              textAlign: 'center',
              transition: 'all 0.15s',
              boxShadow: hov ? `0 8px 24px ${m.color}18` : m.highlight ? `0 2px 12px ${m.color}12` : 'none',
              cursor: 'default',
              position: 'relative',
              overflow: 'hidden',
              minWidth: '160px',
            }}
          >
            {/* Watermark */}
            <div style={{
              position: 'absolute', top: '8px', right: '10px',
              fontFamily: 'monospace', fontSize: '11px', fontStyle: 'italic',
              color: m.color, opacity: hov ? 0.4 : 0.15, userSelect: 'none',
            }}>{m.symbol}</div>

            {/* Glow */}
            <div style={{
              position: 'absolute', inset: 0,
              background: `radial-gradient(ellipse at 50% 0%, ${m.color}12, transparent 70%)`,
              opacity: hov ? 1 : 0, transition: 'opacity 0.2s', pointerEvents: 'none',
            }} />

            <div style={{
              fontFamily: "'JetBrains Mono', monospace", fontSize: '36px', fontWeight: '900',
              color: m.color, lineHeight: 1, marginBottom: '8px', letterSpacing: '-1px',
              position: 'relative',
            }}>{m.value}</div>

            <div style={{ fontWeight: '700', fontSize: '13px', color: 'var(--ifm-font-color-base)', marginBottom: '4px', position: 'relative' }}>
              {m.label}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--ifm-font-color-base)', opacity: 0.5, marginBottom: '10px', position: 'relative' }}>
              {m.sub}
            </div>
            <div style={{
              display: 'inline-block',
              background: `${m.color}14`, border: `1px solid ${m.color}30`,
              borderRadius: '12px', padding: '3px 10px',
              fontSize: '11px', color: m.color, opacity: 0.85, position: 'relative',
            }}>{m.detail}</div>
          </div>
        );
      })}
    </div>
  );
}
