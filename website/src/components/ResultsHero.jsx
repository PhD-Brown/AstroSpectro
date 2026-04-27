import React from 'react';

const RESULTS = [
  {
    axis: 'PCA',
    headline: 'PC1 = Axe thermique',
    metric: '+0,831',
    metricLabel: 'ρ(PC1, Teff)',
    details: ['16,9 % de variance · λ₁ = 30,19', '91 composantes → 95 % de variance', 'Balmer 31,5 % + Ca II 22,4 %'],
    color: '#38BDF8',
    badge: 'Linéaire',
  },
  {
    axis: 'UMAP',
    headline: 'Séquence Harvard émergente',
    metric: '20',
    metricLabel: 'clusters HDBSCAN',
    details: ['Structure en continent continu', 'Naines / géantes séparées sans étiquette', 'ρ(ax.1, Teff) = +0,464'],
    color: '#F59E0B',
    badge: 'Topologique',
  },
  {
    axis: 't-SNE',
    headline: '60× plus reproductible',
    metric: '5×10⁻⁴',
    metricLabel: 'dP Procrustes moyen',
    details: ['init=\'pca\' → stabilité maximale', 'Archipel de clusters compacts', 'Perplexité 30 — optimal [15,50]'],
    color: '#A78BFA',
    badge: 'Probabiliste',
  },
  {
    axis: 'XGBoost',
    headline: '87 % balanced accuracy',
    metric: '~0,964',
    metricLabel: 'ROC-AUC macro',
    details: ['spectro_only=True — physique pure', 'Confiance médiane 96,3 %', 'Cohérence géographique dans UMAP'],
    color: '#34D399',
    badge: 'Supervisé',
  },
  {
    axis: 'SHAP',
    headline: 'Ca II > Balmer',
    metric: '97,9 %',
    metricLabel: 'features physiques top-30',
    details: ['Métallicité plus discriminante que T_eff', 'Validation scientifique centrale', 'Ca II K rangs 1–3'],
    color: '#C084FC',
    badge: 'Interprétabilité',
  },
];

function ResultCard({ axis, headline, metric, metricLabel, details, color, badge }) {
  const [hov, setHov] = React.useState(false);

  return (
    <div
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        background: hov ? `${color}0e` : 'var(--ifm-card-background-color)',
        border: `1px solid ${hov ? color + '55' : color + '25'}`,
        borderTop: `4px solid ${color}`,
        borderRadius: '10px',
        padding: '18px',
        transition: 'all 0.15s',
        boxShadow: hov ? `0 6px 24px ${color}15` : 'none',
        cursor: 'default',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Glow */}
      <div style={{
        position: 'absolute',
        top: 0, left: 0, right: 0,
        height: '60px',
        background: `radial-gradient(ellipse at 50% -10%, ${color}18, transparent)`,
        opacity: hov ? 1 : 0,
        transition: 'opacity 0.2s',
        pointerEvents: 'none',
      }} />

      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontWeight: '900',
          fontSize: '16px',
          color: color,
          lineHeight: 1,
        }}>{axis}</div>
        <div style={{
          background: `${color}18`,
          border: `1px solid ${color}35`,
          borderRadius: '4px',
          padding: '2px 7px',
          fontSize: '10px',
          fontWeight: '700',
          color: color,
          fontFamily: 'monospace',
        }}>{badge}</div>
      </div>

      {/* Headline */}
      <div style={{
        fontWeight: '700',
        fontSize: '14px',
        color: 'var(--ifm-font-color-base)',
        lineHeight: 1.3,
      }}>{headline}</div>

      {/* Big metric */}
      <div style={{ textAlign: 'center', padding: '10px 0 8px' }}>
        <div style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: '34px',
          fontWeight: '900',
          color: color,
          lineHeight: 1,
          letterSpacing: '-1px',
        }}>{metric}</div>
        <div style={{ fontSize: '10px', opacity: 0.5, marginTop: '4px', color: 'var(--ifm-font-color-base)' }}>
          {metricLabel}
        </div>
      </div>

      {/* Details */}
      <div style={{
        borderTop: `1px solid ${color}18`,
        paddingTop: '10px',
        display: 'flex',
        flexDirection: 'column',
        gap: '4px',
      }}>
        {details.map((d, i) => (
          <div key={i} style={{
            display: 'flex',
            gap: '6px',
            fontSize: '11px',
            color: 'var(--ifm-font-color-base)',
            opacity: 0.6,
            lineHeight: 1.4,
          }}>
            <span style={{ color: color, flexShrink: 0, opacity: 0.7 }}>›</span>
            {d}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ResultsHero() {
  return (
    <div style={{ margin: '28px 0' }}>
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
        gap: '12px',
      }}>
        {RESULTS.map((r) => <ResultCard key={r.axis} {...r} />)}
      </div>
    </div>
  );
}
