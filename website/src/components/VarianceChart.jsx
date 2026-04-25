import React from 'react';

// Individual variance percentages for PC1-PC20 (approximate realistic values)
const PC_VARIANCES = [
  16.9, 12.0, 9.5, 7.2, 5.8, 4.6, 3.9, 3.2, 2.8, 2.4,
   2.1,  1.9, 1.7, 1.5, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8,
];

// Cumulative
const cumulative = PC_VARIANCES.reduce((acc, v, i) => {
  acc.push((i === 0 ? 0 : acc[i - 1]) + v);
  return acc;
}, []);

const THRESHOLDS = [
  { pct: 80, k: 51,  color: '#FBBF24' },
  { pct: 90, k: 73,  color: '#FB923C' },
  { pct: 95, k: 91,  color: '#34D399', highlighted: true },
  { pct: 99, k: 100, color: '#F87171' },
];

const MSE_ROWS = [
  { k: 2,  mse: 0.696, captured: 30.4 },
  { k: 10, mse: 0.494, captured: 50.6 },
  { k: 50, mse: 0.196, captured: 80.4 },
  { k: 91, mse: 0.050, captured: 95.0 },
];

export default function VarianceChart() {
  const [hovPc, setHovPc] = React.useState(null);
  const maxVar = Math.max(...PC_VARIANCES);

  return (
    <div style={{ margin: '28px 0' }}>
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '16px',
      }}>
        {/* Left: bar chart of individual variance */}
        <div style={{
          background: 'var(--ifm-card-background-color)',
          border: '1px solid rgba(255,255,255,0.07)',
          borderRadius: '10px',
          padding: '18px',
        }}>
          <div style={{ fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.07em', opacity: 0.4, marginBottom: '14px', color: 'var(--ifm-font-color-base)' }}>
            Variance individuelle — PC1 à PC20
          </div>

          {/* Bars */}
          <div style={{ display: 'flex', gap: '3px', alignItems: 'flex-end', height: '80px', marginBottom: '8px' }}>
            {PC_VARIANCES.map((v, i) => {
              const isHov = hovPc === i;
              const isTop2 = i < 2;
              const height = (v / maxVar) * 100;
              const color = i === 0 ? '#38BDF8' : i === 1 ? '#F59E0B' : `rgba(148,163,184,${0.6 - i * 0.02})`;
              return (
                <div
                  key={i}
                  onMouseEnter={() => setHovPc(i)}
                  onMouseLeave={() => setHovPc(null)}
                  title={`PC${i+1}: ${v}%`}
                  style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', cursor: 'default' }}
                >
                  {isHov && (
                    <div style={{
                      fontSize: '9px', fontFamily: 'monospace', textAlign: 'center',
                      color: color, marginBottom: '2px', whiteSpace: 'nowrap',
                    }}>{v}%</div>
                  )}
                  <div style={{
                    height: `${height}%`,
                    background: isHov ? color : (isTop2 ? color : 'rgba(148,163,184,0.35)'),
                    borderRadius: '2px 2px 0 0',
                    transition: 'background 0.1s',
                    minHeight: '2px',
                  }} />
                </div>
              );
            })}
          </div>

          {/* X axis labels */}
          <div style={{ display: 'flex', gap: '3px' }}>
            {PC_VARIANCES.map((_, i) => (
              <div key={i} style={{
                flex: 1,
                fontSize: '8px',
                textAlign: 'center',
                color: i < 2 ? (i === 0 ? '#38BDF8' : '#F59E0B') : 'var(--ifm-font-color-base)',
                opacity: i < 2 ? 0.9 : (i % 5 === 4 ? 0.4 : 0.15),
                fontFamily: 'monospace',
                fontWeight: i < 2 ? '700' : '400',
              }}>
                {i < 2 ? `PC${i+1}` : (i % 5 === 4 ? `${i+1}` : '')}
              </div>
            ))}
          </div>

          {/* PC1 PC2 labels */}
          <div style={{ display: 'flex', gap: '12px', marginTop: '10px', flexWrap: 'wrap' }}>
            {[
              { label: 'PC1 — Axe thermique', color: '#38BDF8', pct: 16.9 },
              { label: 'PC2 — Axe métallicité', color: '#F59E0B', pct: 12.0 },
            ].map((p) => (
              <div key={p.label} style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <div style={{ width: '10px', height: '10px', borderRadius: '2px', background: p.color }} />
                <span style={{ fontSize: '11px', color: p.color, fontWeight: '600' }}>
                  {p.label} ({p.pct}%)
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Right: cumulative thresholds */}
        <div style={{
          background: 'var(--ifm-card-background-color)',
          border: '1px solid rgba(255,255,255,0.07)',
          borderRadius: '10px',
          padding: '18px',
        }}>
          <div style={{ fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.07em', opacity: 0.4, marginBottom: '14px', color: 'var(--ifm-font-color-base)' }}>
            Seuils de variance cumulée requérant K composantes
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {THRESHOLDS.map((t) => {
              const barWidth = t.k / 183 * 100;
              return (
                <div key={t.pct} style={{
                  padding: '10px 12px',
                  background: t.highlighted ? `${t.color}0e` : 'rgba(255,255,255,0.02)',
                  border: `1px solid ${t.highlighted ? t.color + '44' : 'rgba(255,255,255,0.05)'}`,
                  borderRadius: '7px',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px', alignItems: 'center' }}>
                    <span style={{ fontSize: '14px', fontWeight: t.highlighted ? '800' : '600', color: t.color }}>
                      V(K) ≥ {t.pct}%
                    </span>
                    <div style={{ display: 'flex', alignItems: 'baseline', gap: '4px' }}>
                      <span style={{
                        fontFamily: 'monospace',
                        fontSize: t.highlighted ? '20px' : '16px',
                        fontWeight: '800',
                        color: t.color,
                      }}>K = {t.k}</span>
                      <span style={{ fontSize: '10px', opacity: 0.45, color: 'var(--ifm-font-color-base)' }}>/ 183</span>
                    </div>
                  </div>
                  {/* Bar */}
                  <div style={{ height: '5px', background: 'rgba(255,255,255,0.06)', borderRadius: '3px', overflow: 'hidden' }}>
                    <div style={{
                      width: `${barWidth}%`,
                      height: '100%',
                      background: t.color,
                      borderRadius: '3px',
                      opacity: 0.8,
                    }} />
                  </div>
                  <div style={{ fontSize: '10px', opacity: 0.4, marginTop: '4px', color: 'var(--ifm-font-color-base)' }}>
                    {barWidth.toFixed(0)}% des composantes utilisées
                  </div>
                </div>
              );
            })}
          </div>

          <div style={{
            marginTop: '12px',
            padding: '8px 12px',
            background: 'rgba(56,189,248,0.05)',
            border: '1px solid rgba(56,189,248,0.15)',
            borderRadius: '6px',
            fontSize: '11px',
            color: 'var(--ifm-font-color-base)',
            opacity: 0.75,
          }}>
            ↪ UMAP et t-SNE utilisent les <strong style={{ color: '#38BDF8' }}>91 premières composantes</strong> (seuil 95 %) comme entrée.
          </div>
        </div>
      </div>

      {/* MSE table */}
      <div style={{ marginTop: '16px' }}>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(4, 1fr)',
          gap: '10px',
        }}>
          {MSE_ROWS.map((row) => {
            const isKstar = row.k === 91;
            return (
              <div key={row.k} style={{
                padding: '12px 16px',
                background: isKstar ? 'rgba(52,211,153,0.07)' : 'var(--ifm-card-background-color)',
                border: `1px solid ${isKstar ? '#34D39940' : 'rgba(255,255,255,0.07)'}`,
                borderRadius: '8px',
                textAlign: 'center',
              }}>
                <div style={{ fontSize: '10px', opacity: 0.4, marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--ifm-font-color-base)' }}>
                  K = {row.k} {isKstar && '★'}
                </div>
                <div style={{
                  fontFamily: 'monospace',
                  fontSize: '20px',
                  fontWeight: '800',
                  color: isKstar ? '#34D399' : 'var(--ifm-font-color-base)',
                  lineHeight: 1,
                  marginBottom: '4px',
                }}>{row.mse.toFixed(3)}</div>
                <div style={{ fontSize: '10px', opacity: 0.45, color: 'var(--ifm-font-color-base)' }}>MSE(K)</div>
                <div style={{
                  marginTop: '8px',
                  height: '4px',
                  background: 'rgba(255,255,255,0.06)',
                  borderRadius: '2px',
                  overflow: 'hidden',
                }}>
                  <div style={{
                    width: `${row.captured}%`,
                    height: '100%',
                    background: isKstar ? '#34D399' : '#38BDF8',
                    opacity: 0.7,
                    borderRadius: '2px',
                  }} />
                </div>
                <div style={{ fontSize: '9px', opacity: 0.4, marginTop: '3px', color: 'var(--ifm-font-color-base)' }}>
                  {row.captured}% capturée
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
