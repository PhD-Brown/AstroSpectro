import React from 'react';

// Real data from phy3500_autoencoder_run_latest.json
const PCA_POINTS = [
  { k: 1,  mse: 0.813 },
  { k: 2,  mse: 0.696 },
  { k: 3,  mse: 0.650 },
  { k: 5,  mse: 0.583 },
  { k: 8,  mse: 0.524 },
  { k: 10, mse: 0.494 },
  { k: 15, mse: 0.432 },
  { k: 20, mse: 0.382 },
  { k: 30, mse: 0.309 },
  { k: 50, mse: 0.196 },
];

const AE_MSE = 0.489;  // AE(z=2) — equivalent to PCA(k=10)

const W = 480;
const H = 200;
const PAD = { left: 50, right: 20, top: 22, bottom: 36 };

const K_MIN = 0;
const K_MAX = 52;
const MSE_MIN = 0.1;
const MSE_MAX = 0.85;

function px(k)   { return PAD.left + ((k - K_MIN) / (K_MAX - K_MIN)) * (W - PAD.left - PAD.right); }
function py(mse) { return H - PAD.bottom - ((mse - MSE_MIN) / (MSE_MAX - MSE_MIN)) * (H - PAD.top - PAD.bottom); }

function pathPCA() {
  return PCA_POINTS
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${px(p.k).toFixed(1)},${py(p.mse).toFixed(1)}`)
    .join(' ');
}

const AE_PY = py(AE_MSE);
const AE_PX = px(2); // visual x at k=2 (its dimensionality)

// Find equivalent PCA k (where PCA MSE ≈ AE MSE)
const EQUIV_K = 10; // PCA(k=10) = 0.494 ≈ AE(z=2) = 0.489

export default function AEReconBenchmark() {
  const [hovPt, setHovPt] = React.useState(null);

  return (
    <div style={{ margin: '28px 0' }}>
      <div style={{
        background: 'var(--ifm-card-background-color)',
        border: '1px solid rgba(255,255,255,0.07)',
        borderRadius: '10px',
        padding: '20px',
      }}>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          marginBottom: '16px',
          flexWrap: 'wrap',
          gap: '10px',
        }}>
          <div>
            <div style={{ fontSize: '13px', fontWeight: '700', color: 'var(--ifm-font-color-base)' }}>
              MSE de reconstruction : Autoencodeur (z=2) vs PCA (k=1…50)
            </div>
            <div style={{ fontSize: '10.5px', opacity: 0.45, color: 'var(--ifm-font-color-base)', marginTop: '2px' }}>
              N = 43 019 spectres · 183 features · run 20260405T220839Z
            </div>
          </div>
          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            {[
              { color: '#38BDF8', label: 'PCA (k composantes)' },
              { color: '#34D399', label: 'AE (z = 2 fixe)', dashed: false, dot: true },
            ].map((l) => (
              <div key={l.label} style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', color: l.color, opacity: 0.8 }}>
                {l.dot ? (
                  <div style={{ width: '12px', height: '3px', background: l.color, borderRadius: '1.5px' }} />
                ) : (
                  <div style={{ width: '12px', height: '2px', background: l.color }} />
                )}
                {l.label}
              </div>
            ))}
          </div>
        </div>

        {/* SVG Chart */}
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
          {/* Grid */}
          {[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8].map((m) => (
            <g key={m}>
              <line x1={PAD.left} y1={py(m)} x2={W - PAD.right} y2={py(m)}
                stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
              <text x={PAD.left - 4} y={py(m) + 3.5} fill="rgba(255,255,255,0.25)"
                fontSize="8" fontFamily="monospace" textAnchor="end">{m.toFixed(1)}</text>
            </g>
          ))}

          {/* AE horizontal line (dashed) */}
          <line
            x1={PAD.left} y1={AE_PY}
            x2={W - PAD.right} y2={AE_PY}
            stroke="#34D399" strokeWidth="1.5" strokeDasharray="5,3" opacity="0.7"
          />

          {/* AE label */}
          <text x={W - PAD.right - 2} y={AE_PY - 4} fill="#34D399" fontSize="9"
            fontFamily="monospace" textAnchor="end" fontWeight="bold">
            AE(z=2) = 0.489
          </text>

          {/* Equivalence bracket */}
          <line x1={px(EQUIV_K)} y1={py(0.494)} x2={px(EQUIV_K)} y2={py(AE_MSE)}
            stroke="#FBBF24" strokeWidth="1" strokeDasharray="2,2" opacity="0.7" />
          <circle cx={px(EQUIV_K)} cy={py(0.494)} r="4" fill="#FBBF24" opacity="0.8" />
          <text x={px(EQUIV_K) + 5} y={py(0.52)} fill="#FBBF24" fontSize="8.5"
            fontFamily="monospace">PCA(k=10)≈AE</text>

          {/* PCA curve */}
          <path d={pathPCA()} fill="none" stroke="#38BDF8" strokeWidth="2" opacity="0.85" />

          {/* PCA dots + hover */}
          {PCA_POINTS.map((p, i) => (
            <circle
              key={i}
              cx={px(p.k)} cy={py(p.mse)} r={hovPt === i ? 5 : 3.5}
              fill="#38BDF8" opacity={hovPt === i ? 1 : 0.8}
              style={{ cursor: 'pointer', transition: 'r 0.1s' }}
              onMouseEnter={() => setHovPt(i)}
              onMouseLeave={() => setHovPt(null)}
            />
          ))}

          {/* Tooltip */}
          {hovPt !== null && (() => {
            const p = PCA_POINTS[hovPt];
            const tx = px(p.k);
            const ty = py(p.mse) - 26;
            return (
              <g>
                <rect x={tx - 28} y={ty - 2} width="56" height="20" rx="4"
                  fill="#0d1117" stroke="#38BDF8" strokeWidth="0.5" opacity="0.95" />
                <text x={tx} y={ty + 12} fill="#38BDF8" fontSize="10" fontFamily="monospace"
                  textAnchor="middle">k={p.k} · {p.mse.toFixed(3)}</text>
              </g>
            );
          })()}

          {/* Axes */}
          <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom}
            stroke="rgba(255,255,255,0.15)" strokeWidth="1" />
          <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom}
            stroke="rgba(255,255,255,0.15)" strokeWidth="1" />

          {/* X ticks */}
          {[1, 2, 5, 10, 20, 30, 50].map((k) => (
            <g key={k}>
              <line x1={px(k)} y1={H - PAD.bottom} x2={px(k)} y2={H - PAD.bottom + 4}
                stroke="rgba(255,255,255,0.2)" strokeWidth="1" />
              <text x={px(k)} y={H - PAD.bottom + 13} fill="rgba(255,255,255,0.3)"
                fontSize="8" fontFamily="monospace" textAnchor="middle">{k}</text>
            </g>
          ))}

          {/* Axis labels */}
          <text x={PAD.left + (W - PAD.left - PAD.right) / 2} y={H - 2}
            fill="rgba(255,255,255,0.25)" fontSize="9" fontFamily="monospace" textAnchor="middle">
            K composantes PCA →
          </text>
          <text x={10} y={H / 2} fill="rgba(255,255,255,0.25)" fontSize="9" fontFamily="monospace"
            transform={`rotate(-90, 10, ${H / 2})`} textAnchor="middle">MSE</text>
        </svg>

        {/* Key finding box */}
        <div style={{
          marginTop: '14px',
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '10px',
        }}>
          <div style={{
            padding: '10px 14px',
            background: 'rgba(52,211,153,0.07)',
            border: '1px solid rgba(52,211,153,0.22)',
            borderLeft: '3px solid #34D399',
            borderRadius: '7px',
            fontSize: '12px',
            color: 'var(--ifm-font-color-base)',
            opacity: 0.85,
            lineHeight: 1.6,
          }}>
            <strong style={{ color: '#34D399' }}>AE(z=2) ≈ PCA(k=10)</strong> — avec seulement 2 dimensions latentes, l'autoencodeur atteint la même MSE que la PCA avec 10 composantes. La non-linéarité compense la dimensionnalité.
          </div>
          <div style={{
            padding: '10px 14px',
            background: 'rgba(56,189,248,0.05)',
            border: '1px solid rgba(56,189,248,0.15)',
            borderLeft: '3px solid #38BDF8',
            borderRadius: '7px',
            fontSize: '12px',
            color: 'var(--ifm-font-color-base)',
            opacity: 0.85,
            lineHeight: 1.6,
          }}>
            <strong style={{ color: '#38BDF8' }}>Limite :</strong> PCA(k=50) atteint MSE = 0.196, bien en dessous de l'AE(z=2). La PCA profite de ses 50 degrés de liberté — l'autoencodeur 2D est contraint par sa faible dimensionnalité latente.
          </div>
        </div>
      </div>
    </div>
  );
}
