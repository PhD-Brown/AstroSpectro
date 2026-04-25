import React from 'react';

const ARROW = () => (
  <div style={{
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    height: '28px',
    color: 'rgba(255,255,255,0.2)',
    fontSize: '18px',
    userSelect: 'none',
    flexShrink: 0,
  }}>↓</div>
);

const ARROW_H = () => (
  <div style={{
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    width: '28px',
    color: 'rgba(255,255,255,0.2)',
    fontSize: '16px',
    userSelect: 'none',
    flexShrink: 0,
  }}>→</div>
);

function Node({ label, sub, meta, color, dim, isInput, isFork }) {
  const [hov, setHov] = React.useState(false);

  return (
    <div
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        background: hov ? `${color}16` : `${color}0a`,
        border: `1px solid ${hov ? color + '66' : color + '33'}`,
        borderRadius: '10px',
        padding: '14px 18px',
        transition: 'all 0.15s',
        boxShadow: hov ? `0 4px 18px ${color}18` : 'none',
        cursor: 'default',
        minWidth: '140px',
        textAlign: 'center',
        flex: isFork ? 1 : undefined,
      }}
    >
      {/* Dimension badge */}
      {dim && (
        <div style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: '18px',
          fontWeight: '900',
          color: color,
          lineHeight: 1,
          marginBottom: '4px',
        }}>{dim}</div>
      )}

      {/* Label */}
      <div style={{
        fontWeight: '700',
        fontSize: '13px',
        color: 'var(--ifm-font-color-base)',
        marginBottom: dim ? '2px' : '4px',
      }}>{label}</div>

      {/* Sub */}
      {sub && (
        <div style={{
          fontSize: '11px',
          color: color,
          opacity: 0.75,
          fontFamily: 'monospace',
          marginBottom: meta ? '6px' : 0,
        }}>{sub}</div>
      )}

      {/* Meta */}
      {meta && (
        <div style={{
          fontSize: '10px',
          opacity: 0.45,
          color: 'var(--ifm-font-color-base)',
          lineHeight: 1.5,
        }}>{meta}</div>
      )}
    </div>
  );
}

export default function DimRedPipeline() {
  return (
    <div style={{
      margin: '28px 0',
      background: 'var(--ifm-card-background-color)',
      border: '1px solid rgba(255,255,255,0.07)',
      borderRadius: '12px',
      padding: '24px',
    }}>
      <div style={{
        fontSize: '11px',
        fontWeight: '700',
        textTransform: 'uppercase',
        letterSpacing: '0.08em',
        opacity: 0.4,
        marginBottom: '20px',
        color: 'var(--ifm-font-color-base)',
      }}>
        Pipeline de réduction de dimension — flux des données
      </div>

      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '0',
      }}>
        {/* Step 1 - Raw input */}
        <Node
          dim="43 019 × 183"
          label="Matrice brute X"
          sub="183 descripteurs spectroscopiques"
          meta="dtype float64 · ~60 MB"
          color="#94A3B8"
        />
        <ARROW />

        {/* Step 2 - StandardScaler */}
        <Node
          dim="43 019 × 183"
          label="StandardScaler"
          sub="x̃ᵢⱼ = (xᵢⱼ − x̄ⱼ) / sⱼ"
          meta="Variance = 1 par descripteur · Moy = 0"
          color="#60A5FA"
        />
        <ARROW />

        {/* Step 3 - PCA */}
        <Node
          dim="43 019 × 91"
          label="PCA (K = 91)"
          sub="95 % de variance · Lanczos SVD"
          meta="t < 1 s · déterministe · Eckart-Young optimal"
          color="#38BDF8"
        />

        {/* Fork arrow */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0',
          width: '100%',
          maxWidth: '520px',
          margin: '0',
        }}>
          {/* Left branch */}
          <div style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
          }}>
            <div style={{ height: '28px', width: '1px', background: 'rgba(245,158,11,0.3)' }} />
            <div style={{ width: '50%', height: '1px', background: 'rgba(245,158,11,0.3)', alignSelf: 'flex-end' }} />
          </div>
          {/* Right branch */}
          <div style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
          }}>
            <div style={{ height: '28px', width: '1px', background: 'rgba(167,139,250,0.3)' }} />
            <div style={{ width: '50%', height: '1px', background: 'rgba(167,139,250,0.3)', alignSelf: 'flex-start' }} />
          </div>
        </div>

        {/* Fork nodes */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '14px',
          width: '100%',
          maxWidth: '520px',
        }}>
          {/* UMAP branch */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0' }}>
            <ARROW />
            <Node
              dim="43 019 × 2"
              label="UMAP"
              sub="Topologique · Manifold"
              meta={'n_neighbors=15 · min_dist=0.1\nt = 40,1 s · non déterministe'}
              color="#F59E0B"
              isFork
            />
          </div>

          {/* t-SNE branch */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0' }}>
            <ARROW />
            <Node
              dim="43 019 × 2"
              label="t-SNE"
              sub="Probabiliste · KL divergence"
              meta={'perplexité=30 · init=\'pca\'\nt = 80,2 s · stable (×60)'}
              color="#A78BFA"
              isFork
            />
          </div>
        </div>

        {/* Output */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '14px',
          width: '100%',
          maxWidth: '520px',
          marginTop: '0',
        }}>
          {[
            { label: 'Continent continu', sub: 'Séquence Harvard visible\nHDBSCAN → 20 clusters', color: '#F59E0B' },
            { label: 'Archipel compact', sub: 'Sous-populations séparées\nt-SNE ~60× plus stable', color: '#A78BFA' },
          ].map((o) => (
            <div key={o.label} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <ARROW />
              <div style={{
                width: '100%',
                padding: '10px 14px',
                background: `${o.color}0c`,
                border: `1px dashed ${o.color}40`,
                borderRadius: '8px',
                textAlign: 'center',
              }}>
                <div style={{ fontSize: '12px', fontWeight: '700', color: o.color, marginBottom: '3px' }}>
                  {o.label}
                </div>
                <div style={{ fontSize: '11px', opacity: 0.55, color: 'var(--ifm-font-color-base)', whiteSpace: 'pre-line' }}>
                  {o.sub}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* HDBSCAN (only under UMAP) */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '14px',
          width: '100%',
          maxWidth: '520px',
          marginTop: '0',
        }}>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <ARROW />
            <div style={{
              width: '100%',
              padding: '10px 14px',
              background: 'rgba(52,211,153,0.06)',
              border: '1px solid rgba(52,211,153,0.25)',
              borderRadius: '8px',
              textAlign: 'center',
            }}>
              <div style={{ fontSize: '12px', fontWeight: '700', color: '#34D399', marginBottom: '2px' }}>
                HDBSCAN
              </div>
              <div style={{ fontSize: '11px', opacity: 0.55, color: 'var(--ifm-font-color-base)' }}>
                20 clusters · 6,14 % bruit
              </div>
            </div>
          </div>
          <div /> {/* empty right column */}
        </div>
      </div>

      {/* Legend */}
      <div style={{
        marginTop: '20px',
        display: 'flex',
        gap: '16px',
        flexWrap: 'wrap',
        justifyContent: 'center',
        fontSize: '11px',
        opacity: 0.45,
        color: 'var(--ifm-font-color-base)',
      }}>
        <span>Entrée : X ∈ ℝ^(43019×183)</span>
        <span>·</span>
        <span>Intermédiaire PCA : Z_pca ∈ ℝ^(43019×91)</span>
        <span>·</span>
        <span>Sorties : Z_umap, Z_tsne ∈ ℝ^(43019×2)</span>
      </div>
    </div>
  );
}
