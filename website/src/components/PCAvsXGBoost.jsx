import React from 'react';

const SIDES = [
  {
    method: 'PCA',
    goal: 'Direction de variance maximale',
    color: '#38BDF8',
    topFeature: 'Raies de Balmer',
    topFeatureColor: '#60A5FA',
    topFeatureSub: 'Hα, Hβ, Hγ… PC1 = 31.5%',
    result: 'Température T_eff',
    resultMetric: 'ρ(PC1, Teff) = +0.831',
    interpretation: 'Les spectres varient surtout en Teff entre les types O et M. Le Balmer est le plus variable dans l\'espace de variance.',
    nature: 'Non supervisé',
    icon: '∑',
    bullets: [
      'Maximise la variance totale',
      'Balmer domine → PC1 thermique',
      'Température = source de variance #1',
      'Aucune étiquette utilisée',
    ],
  },
  {
    method: 'XGBoost',
    goal: 'Frontière de décision optimale',
    color: '#F59E0B',
    topFeature: 'Raies Ca II H&K',
    topFeatureColor: '#C084FC',
    topFeatureSub: 'Ca K, Ca H (rangs 1-5 SHAP)',
    result: 'Métallicité + gravité',
    resultMetric: 'Ca II domine top-5 SHAP',
    interpretation: 'Pour distinguer les classes, Ca II et Mg b sont plus discriminants que Balmer — la métallicité sépare mieux les types spectraux que la température seule.',
    nature: 'Supervisé',
    icon: '⟂',
    bullets: [
      'Minimise l\'erreur de classification',
      'Ca II domine → discriminabilité',
      'Métallicité = meilleur séparateur',
      'Étiquettes de type spectral utilisées',
    ],
  },
];

function Side({ method, goal, color, topFeature, topFeatureColor, topFeatureSub, result, resultMetric, interpretation, nature, icon, bullets }) {
  const [hov, setHov] = React.useState(false);

  return (
    <div
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        flex: 1,
        background: hov ? `${color}08` : 'var(--ifm-card-background-color)',
        border: `1px solid ${hov ? color + '55' : color + '25'}`,
        borderTop: `4px solid ${color}`,
        borderRadius: '10px',
        padding: '20px',
        transition: 'all 0.15s',
        cursor: 'default',
        minWidth: 0,
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '14px' }}>
        <div>
          <div style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontWeight: '900',
            fontSize: '20px',
            color: color,
          }}>{method}</div>
          <div style={{ fontSize: '11px', opacity: 0.55, color: 'var(--ifm-font-color-base)' }}>{goal}</div>
        </div>
        <div style={{
          background: `${color}15`,
          border: `1px solid ${color}35`,
          borderRadius: '5px',
          padding: '2px 8px',
          fontSize: '10px',
          fontWeight: '700',
          color: color,
        }}>{nature}</div>
      </div>

      {/* Top feature */}
      <div style={{
        padding: '12px',
        background: `${topFeatureColor}0e`,
        border: `1px solid ${topFeatureColor}30`,
        borderRadius: '7px',
        marginBottom: '14px',
      }}>
        <div style={{ fontSize: '10px', opacity: 0.45, marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--ifm-font-color-base)' }}>
          Feature dominante
        </div>
        <div style={{ fontWeight: '700', fontSize: '14px', color: topFeatureColor, marginBottom: '2px' }}>
          {topFeature}
        </div>
        <div style={{ fontSize: '11px', fontFamily: 'monospace', color: topFeatureColor, opacity: 0.7 }}>
          {topFeatureSub}
        </div>
      </div>

      {/* Result */}
      <div style={{ marginBottom: '14px' }}>
        <div style={{ fontSize: '10px', opacity: 0.4, marginBottom: '3px', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--ifm-font-color-base)' }}>
          Résultat physique
        </div>
        <div style={{ fontWeight: '800', fontSize: '15px', color: color, marginBottom: '3px' }}>{result}</div>
        <div style={{ fontFamily: 'monospace', fontSize: '11px', color: color, opacity: 0.75 }}>{resultMetric}</div>
      </div>

      {/* Bullets */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
        {bullets.map((b, i) => (
          <div key={i} style={{ display: 'flex', gap: '7px', alignItems: 'flex-start', fontSize: '11.5px', color: 'var(--ifm-font-color-base)', opacity: 0.7 }}>
            <span style={{ color: color, flexShrink: 0, marginTop: '1px' }}>›</span>
            <span>{b}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function PCAvsXGBoost() {
  return (
    <div style={{ margin: '28px 0' }}>
      <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', marginBottom: '14px' }}>
        {SIDES.map((s) => <Side key={s.method} {...s} />)}
      </div>

      {/* Synthesis */}
      <div style={{
        padding: '14px 18px',
        background: 'rgba(255,255,255,0.03)',
        border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: '8px',
        fontSize: '13px',
        color: 'var(--ifm-font-color-base)',
        opacity: 0.85,
        lineHeight: 1.65,
      }}>
        <strong style={{ color: '#34D399' }}>✓ Ces deux résultats sont cohérents et complémentaires</strong> — ils répondent à deux questions différentes. La PCA répond à : <em>«quelle est la source de variance dominante ?»</em> → la température. XGBoost répond à : <em>«quelle information sépare le mieux les classes ?»</em> → la métallicité via Ca II. Le fait que les deux approches, supervisée et non supervisée, convergent vers la même structure physique confirme la validité des 183 descripteurs.
      </div>
    </div>
  );
}
