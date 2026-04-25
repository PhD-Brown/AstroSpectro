import React from 'react';

const UMAP_SEEDS = [
  { seed: 1, dP: 0.00701 },
  { seed: 2, dP: 0.03802 },
  { seed: 3, dP: 0.04971 },
  { seed: 4, dP: 0.02461 },
];
const UMAP_MEAN = 0.030;

const TSNE_SEEDS = [
  { seed: 1, dP: 0.000490 },
  { seed: 2, dP: 0.000101 },
  { seed: 3, dP: 0.000285 },
  { seed: 4, dP: 0.001016 },
];
const TSNE_MEAN = 5.0e-4;

const UMAP_COLOR  = '#F59E0B';
const TSNE_COLOR  = '#A78BFA';
const MAX_DP      = 0.06; // common y-axis max

function SeedBar({ seed, dP, color, maxDP, label }) {
  const heightPct = (dP / maxDP) * 100;
  const [hov, setHov] = React.useState(false);

  return (
    <div
      style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px', flex: 1 }}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
    >
      {/* Tooltip */}
      {hov && (
        <div style={{
          background: 'var(--ifm-card-background-color)',
          border: `1px solid ${color}55`,
          borderRadius: '5px',
          padding: '4px 8px',
          fontSize: '11px',
          fontFamily: 'monospace',
          color: color,
          whiteSpace: 'nowrap',
          position: 'absolute',
          transform: 'translateY(-28px)',
          boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
          zIndex: 10,
        }}>
          dP = {dP.toExponential(2)}
        </div>
      )}

      {/* Bar column */}
      <div style={{
        width: '100%',
        height: '120px',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'flex-end',
        position: 'relative',
      }}>
        <div style={{
          width: '100%',
          height: `${Math.max(heightPct, 1)}%`,
          background: hov
            ? color
            : `linear-gradient(180deg, ${color}, ${color}99)`,
          borderRadius: '4px 4px 0 0',
          transition: 'background 0.15s',
          minHeight: '3px',
        }} />
      </div>

      {/* Seed label */}
      <div style={{
        fontSize: '11px',
        color: 'var(--ifm-font-color-base)',
        opacity: 0.5,
        fontFamily: 'monospace',
      }}>g{seed}</div>
    </div>
  );
}

function Panel({ title, color, seeds, mean, maxDP, subtitle }) {
  const meanPct = (mean / maxDP) * 100;

  return (
    <div style={{
      flex: 1,
      background: 'var(--ifm-card-background-color)',
      border: `1px solid ${color}33`,
      borderTop: `3px solid ${color}`,
      borderRadius: '10px',
      padding: '20px',
    }}>
      {/* Header */}
      <div style={{ marginBottom: '16px' }}>
        <div style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontWeight: '800',
          fontSize: '16px',
          color: color,
          marginBottom: '3px',
        }}>{title}</div>
        <div style={{
          fontSize: '11px',
          color: 'var(--ifm-font-color-base)',
          opacity: 0.5,
        }}>{subtitle}</div>
      </div>

      {/* Chart area */}
      <div style={{ position: 'relative', marginBottom: '12px' }}>
        {/* Y-axis labels */}
        <div style={{
          position: 'absolute',
          left: 0,
          top: 0,
          bottom: '20px',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          width: '48px',
        }}>
          {[maxDP, maxDP * 0.5, 0].map((v) => (
            <div key={v} style={{
              fontSize: '9px',
              color: 'var(--ifm-font-color-base)',
              opacity: 0.35,
              fontFamily: 'monospace',
              textAlign: 'right',
              paddingRight: '6px',
            }}>
              {v === 0 ? '0' : v.toExponential(0)}
            </div>
          ))}
        </div>

        {/* Bars */}
        <div style={{
          marginLeft: '54px',
          position: 'relative',
        }}>
          {/* Grid lines */}
          {[0.25, 0.5, 0.75, 1].map((f) => (
            <div key={f} style={{
              position: 'absolute',
              left: 0,
              right: 0,
              bottom: `calc(20px + ${f * 120}px)`,
              height: '1px',
              background: 'rgba(255,255,255,0.05)',
            }} />
          ))}

          {/* Mean line */}
          <div style={{
            position: 'absolute',
            left: 0,
            right: 0,
            bottom: `calc(20px + ${meanPct * 1.2}px)`,
            height: '2px',
            background: `${color}99`,
            zIndex: 5,
          }}>
            <div style={{
              position: 'absolute',
              right: 0,
              top: '-9px',
              fontSize: '9px',
              color: color,
              fontFamily: 'monospace',
              whiteSpace: 'nowrap',
              opacity: 0.8,
            }}>μ = {mean.toExponential(1)}</div>
          </div>

          {/* Bars row */}
          <div style={{
            display: 'flex',
            gap: '8px',
            alignItems: 'flex-end',
            position: 'relative',
          }}>
            {seeds.map((s) => (
              <SeedBar
                key={s.seed}
                seed={s.seed}
                dP={s.dP}
                color={color}
                maxDP={maxDP}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Mean stat */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '8px 10px',
        background: `${color}0c`,
        border: `1px solid ${color}25`,
        borderRadius: '6px',
        marginTop: '8px',
      }}>
        <span style={{ fontSize: '12px', opacity: 0.65, color: 'var(--ifm-font-color-base)' }}>Moyenne dP</span>
        <span style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontWeight: '800',
          fontSize: '14px',
          color: color,
        }}>
          {mean < 0.001 ? mean.toExponential(1) : mean.toFixed(3)}
        </span>
      </div>
    </div>
  );
}

export default function ProcrustesStability() {
  return (
    <div style={{ margin: '28px 0' }}>
      {/* 60× badge */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '12px',
        marginBottom: '20px',
        padding: '14px 20px',
        background: 'rgba(167,139,250,0.06)',
        border: '1px solid rgba(167,139,250,0.2)',
        borderRadius: '10px',
      }}>
        <span style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontWeight: '900',
          fontSize: '28px',
          color: '#A78BFA',
          letterSpacing: '-1px',
        }}>~60×</span>
        <div>
          <div style={{ fontWeight: '700', fontSize: '14px', color: 'var(--ifm-font-color-base)' }}>
            t-SNE est plus reproductible qu'UMAP
          </div>
          <div style={{ fontSize: '12px', opacity: 0.55, color: 'var(--ifm-font-color-base)' }}>
            Résultat contre-intuitif — les deux méthodes utilisent la SGD
          </div>
        </div>
      </div>

      {/* Side-by-side panels */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '14px',
      }}>
        <Panel
          title="UMAP"
          subtitle="Initialisation spectrale — SGD avec éch. négatif"
          color={UMAP_COLOR}
          seeds={UMAP_SEEDS}
          mean={UMAP_MEAN}
          maxDP={MAX_DP}
        />
        <Panel
          title="t-SNE"
          subtitle="init='pca' — pénalité KL concentrée localement"
          color={TSNE_COLOR}
          seeds={TSNE_SEEDS}
          mean={TSNE_MEAN}
          maxDP={MAX_DP}
        />
      </div>

      {/* Explanation chips */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '10px',
        marginTop: '14px',
      }}>
        <div style={{
          padding: '10px 14px',
          background: `${UMAP_COLOR}0a`,
          border: `1px solid ${UMAP_COLOR}25`,
          borderRadius: '7px',
          fontSize: '12px',
          color: 'var(--ifm-font-color-base)',
          opacity: 0.8,
        }}>
          <strong style={{ color: UMAP_COLOR }}>UMAP instable :</strong> L'initialisation spectrale (Laplacien) varie d'une graine à l'autre. La SGD avec échantillonnage négatif introduit une stochasticité forte dans les forces répulsives.
        </div>
        <div style={{
          padding: '10px 14px',
          background: `${TSNE_COLOR}0a`,
          border: `1px solid ${TSNE_COLOR}25`,
          borderRadius: '7px',
          fontSize: '12px',
          color: 'var(--ifm-font-color-base)',
          opacity: 0.8,
        }}>
          <strong style={{ color: TSNE_COLOR }}>t-SNE stable :</strong> <code>init='pca'</code> fixe une position de départ identique. La pénalité KL pénalise uniquement les voisins proches — peu sensible à la graine pour les grandes structures.
        </div>
      </div>
    </div>
  );
}
