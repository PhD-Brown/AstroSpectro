import React from 'react';

const METHODS = [
  { name: 'PCA',   sub: 'Linéaire',       color: '#38BDF8' },
  { name: 'UMAP',  sub: 'Topologique',    color: '#F59E0B' },
  { name: 't-SNE', sub: 'Probabiliste',   color: '#A78BFA' },
];

// type: 'numeric' | 'bool' | 'tristate' | 'bar'
const ROWS = [
  {
    criterion: 'MSE reconstruction (K=2)',
    note: 'PCA uniquement — Eckart-Young',
    values: ['0,696', '—', '—'],
    best: null,
  },
  {
    criterion: 'ρ(axe 1, Teff)',
    note: 'Corrélation de Spearman avec T_eff Gaia',
    type: 'bar',
    barValues: [0.831, 0.464, 0.623],
    rawLabels: ['+0,831', '+0,464', '+0,623'],
    best: 0,
    highlight: true,
  },
  {
    criterion: 'Stabilité dP (Procrustes)',
    note: 't-SNE ~60× plus reproductible qu\'UMAP',
    values: ['0 (exact)', '3,0 × 10⁻²', '5,0 × 10⁻⁴'],
    best: 0,
  },
  {
    criterion: 'Temps CPU',
    note: 'Ryzen 9 5950X · 32 fils',
    type: 'bar',
    barValues: [0.5, 40.1, 80.2],
    rawLabels: ['< 1 s', '40,1 s', '80,2 s'],
    best: 0,
    inverted: true, // lower is better
  },
  {
    criterion: 'Paramétrique',
    note: 'Peut généraliser hors échantillon',
    type: 'tristate',
    values: ['Oui', 'Partiel', 'Non'],
    best: 0,
  },
  {
    criterion: 'Non-linéaire',
    note: 'Capture les structures non-linéaires',
    type: 'bool',
    values: [false, true, true],
    best: null,
  },
  {
    criterion: 'Interprétable',
    note: 'Axes avec signification physique directe',
    type: 'bool',
    values: [true, false, false],
    best: 0,
  },
];

function BoolIcon({ value }) {
  if (value === true) return (
    <span style={{ color: '#34D399', fontWeight: '800', fontSize: '16px' }}>✓</span>
  );
  if (value === false) return (
    <span style={{ color: '#F87171', fontWeight: '800', fontSize: '16px' }}>✗</span>
  );
  return <span style={{ color: '#94A3B8', fontSize: '13px' }}>—</span>;
}

function TristateChip({ value, color }) {
  const colors = {
    'Oui':     { bg: '#34D39918', border: '#34D39944', text: '#34D399' },
    'Partiel': { bg: '#F59E0B18', border: '#F59E0B44', text: '#F59E0B' },
    'Non':     { bg: '#F8717118', border: '#F8717144', text: '#F87171' },
  };
  const c = colors[value] || { bg: '#ffffff10', border: '#ffffff30', text: '#94A3B8' };
  return (
    <span style={{
      background: c.bg,
      border: `1px solid ${c.border}`,
      color: c.text,
      borderRadius: '4px',
      padding: '2px 8px',
      fontSize: '12px',
      fontWeight: '700',
    }}>{value}</span>
  );
}

function MicroBar({ value, max, color, inverted }) {
  const pct = Math.min((value / max) * 100, 100);
  // for inverted (lower=better), flip the visual
  const displayPct = inverted ? Math.max(4, (1 - value / max) * 100 + 4) : Math.max(4, pct);
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
      <div style={{
        flex: 1,
        height: '6px',
        background: 'rgba(255,255,255,0.08)',
        borderRadius: '3px',
        overflow: 'hidden',
      }}>
        <div style={{
          width: `${displayPct}%`,
          height: '100%',
          background: color,
          borderRadius: '3px',
          opacity: 0.75,
        }} />
      </div>
    </div>
  );
}

export default function MethodComparison() {
  const [hoveredRow, setHoveredRow] = React.useState(null);

  return (
    <div style={{ margin: '28px 0' }}>
      {/* Column headers */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '200px repeat(3, 1fr)',
        gap: '0',
        marginBottom: '4px',
      }}>
        <div />
        {METHODS.map((m) => (
          <div key={m.name} style={{
            background: `${m.color}18`,
            border: `1px solid ${m.color}44`,
            borderBottom: `3px solid ${m.color}`,
            borderRadius: '8px 8px 0 0',
            padding: '14px 12px 10px',
            textAlign: 'center',
            margin: '0 3px',
          }}>
            <div style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontWeight: '800',
              fontSize: '18px',
              color: m.color,
              lineHeight: 1,
              marginBottom: '4px',
            }}>{m.name}</div>
            <div style={{
              fontSize: '11px',
              color: m.color,
              opacity: 0.65,
              textTransform: 'uppercase',
              letterSpacing: '0.08em',
            }}>{m.sub}</div>
          </div>
        ))}
      </div>

      {/* Rows */}
      <div style={{
        border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: '0 0 10px 10px',
        overflow: 'hidden',
      }}>
        {ROWS.map((row, ri) => {
          const isHighlight = row.highlight;
          const isHovered = hoveredRow === ri;
          return (
            <div
              key={row.criterion}
              onMouseEnter={() => setHoveredRow(ri)}
              onMouseLeave={() => setHoveredRow(null)}
              style={{
                display: 'grid',
                gridTemplateColumns: '200px repeat(3, 1fr)',
                borderBottom: ri < ROWS.length - 1
                  ? '1px solid rgba(255,255,255,0.05)' : 'none',
                background: isHovered
                  ? 'rgba(255,255,255,0.04)'
                  : isHighlight
                  ? 'rgba(56,189,248,0.04)'
                  : ri % 2 === 0
                  ? 'rgba(255,255,255,0.015)'
                  : 'transparent',
                transition: 'background 0.12s',
              }}
            >
              {/* Criterion label */}
              <div style={{
                padding: '14px 16px',
                borderRight: '1px solid rgba(255,255,255,0.06)',
              }}>
                <div style={{
                  fontSize: '13px',
                  fontWeight: '600',
                  color: 'var(--ifm-font-color-base)',
                  marginBottom: '3px',
                  lineHeight: 1.3,
                }}>{row.criterion}</div>
                {row.note && (
                  <div style={{
                    fontSize: '10.5px',
                    color: 'var(--ifm-font-color-base)',
                    opacity: 0.42,
                    lineHeight: 1.4,
                  }}>{row.note}</div>
                )}
              </div>

              {/* Values */}
              {METHODS.map((m, mi) => {
                const isBest = row.best === mi;
                return (
                  <div key={m.name} style={{
                    padding: '12px 14px',
                    textAlign: 'center',
                    borderRight: mi < 2 ? '1px solid rgba(255,255,255,0.04)' : 'none',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '6px',
                    position: 'relative',
                  }}>
                    {isBest && (
                      <div style={{
                        position: 'absolute',
                        top: '4px',
                        right: '6px',
                        width: '6px',
                        height: '6px',
                        borderRadius: '50%',
                        background: m.color,
                        boxShadow: `0 0 0 2px ${m.color}33`,
                      }} />
                    )}

                    {row.type === 'bool' ? (
                      <BoolIcon value={row.values[mi]} />
                    ) : row.type === 'tristate' ? (
                      <TristateChip value={row.values[mi]} color={m.color} />
                    ) : row.type === 'bar' ? (
                      <>
                        <div style={{
                          fontFamily: "'JetBrains Mono', monospace",
                          fontSize: '14px',
                          fontWeight: '700',
                          color: isBest ? m.color : 'var(--ifm-font-color-base)',
                          opacity: isBest ? 1 : 0.75,
                        }}>
                          {row.rawLabels[mi]}
                        </div>
                        <MicroBar
                          value={row.barValues[mi]}
                          max={Math.max(...row.barValues)}
                          color={m.color}
                          inverted={row.inverted}
                        />
                      </>
                    ) : (
                      <div style={{
                        fontFamily: "'JetBrains Mono', monospace",
                        fontSize: '13px',
                        fontWeight: isBest ? '700' : '400',
                        color: isBest ? m.color : 'var(--ifm-font-color-base)',
                        opacity: isBest ? 1 : 0.72,
                      }}>
                        {row.values[mi]}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div style={{
        marginTop: '10px',
        fontSize: '11px',
        color: 'var(--ifm-font-color-base)',
        opacity: 0.4,
        textAlign: 'right',
      }}>
        ● = meilleure valeur sur la ligne
      </div>
    </div>
  );
}
