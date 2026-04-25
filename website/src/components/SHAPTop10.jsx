import React from 'react';

const FAMILY_COLORS = {
  'Ca II':  '#C084FC',
  'Balmer': '#60A5FA',
  'Métaux': '#34D399',
};

const SHAP_FEATURES = [
  { rank: '1',   name: 'Ca II K prominence',      family: 'Ca II',  importance: 0.98, note: 'Force de la raie K 3934 Å' },
  { rank: '2',   name: 'Ca II K EW',               family: 'Ca II',  importance: 0.94, note: 'Largeur équivalente W(CaIIK)' },
  { rank: '3',   name: 'Ca II K FWHM',             family: 'Ca II',  importance: 0.91, note: 'Largeur à mi-hauteur' },
  { rank: '4',   name: 'Ca II H prominence',       family: 'Ca II',  importance: 0.87, note: 'Force de la raie H 3968 Å' },
  { rank: '5',   name: 'Ca II H EW',               family: 'Ca II',  importance: 0.84, note: 'Largeur équivalente W(CaIIH)' },
  { rank: '6',   name: 'Hα EW',                    family: 'Balmer', importance: 0.76, note: 'Largeur équivalente Hα 6563 Å' },
  { rank: '7',   name: 'Hα prominence',            family: 'Balmer', importance: 0.73, note: 'Force de la raie Hα' },
  { rank: '8',   name: 'Mg b EW',                  family: 'Métaux', importance: 0.68, note: 'W(Mg b) 5175 Å — proxy [Mg/Fe]' },
  { rank: '9',   name: 'Mg b prominence',          family: 'Métaux', importance: 0.64, note: 'Force du triplet Mg b' },
  { rank: '10',  name: 'Balmer temperature index', family: 'Balmer', importance: 0.59, note: 'Gradient Hγ/Hβ — indice T_eff' },
];

export default function SHAPTop10() {
  const [hov, setHov] = React.useState(null);

  const maxVal = Math.max(...SHAP_FEATURES.map(f => f.importance));

  return (
    <div style={{ margin: '28px 0' }}>
      {/* Discovery banner */}
      <div style={{
        display: 'flex',
        alignItems: 'flex-start',
        gap: '14px',
        padding: '16px 18px',
        background: 'rgba(192,132,252,0.08)',
        border: '1px solid rgba(192,132,252,0.3)',
        borderRadius: '10px',
        marginBottom: '20px',
      }}>
        <div style={{
          fontFamily: 'monospace',
          fontSize: '28px',
          color: '#C084FC',
          lineHeight: 1,
          flexShrink: 0,
        }}>⚡</div>
        <div>
          <div style={{ fontWeight: '800', fontSize: '14px', color: '#C084FC', marginBottom: '4px' }}>
            Découverte clé — Ca II domine sur Balmer
          </div>
          <div style={{ fontSize: '13px', color: 'var(--ifm-font-color-base)', opacity: 0.8, lineHeight: 1.5 }}>
            Les 5 premiers descripteurs SHAP sont tous des raies Ca II H&K — la <strong>métallicité est plus discriminante que la température</strong> pour XGBoost. Contraste direct avec la PCA où la température (Balmer, PC1) domine.
          </div>
        </div>
      </div>

      {/* Family legend */}
      <div style={{
        display: 'flex',
        gap: '14px',
        marginBottom: '14px',
        flexWrap: 'wrap',
      }}>
        {Object.entries(FAMILY_COLORS).map(([fam, col]) => {
          const count = SHAP_FEATURES.filter(f => f.family === fam).length;
          return (
            <div key={fam} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div style={{ width: '12px', height: '12px', borderRadius: '3px', background: col }} />
              <span style={{ fontSize: '12px', fontWeight: '600', color: col }}>
                {fam} ({count} dans top-10)
              </span>
            </div>
          );
        })}
        <div style={{ marginLeft: 'auto', fontSize: '10px', opacity: 0.35, color: 'var(--ifm-font-color-base)', alignSelf: 'center' }}>
          Importance relative normalisée — run 20260213T225019Z
        </div>
      </div>

      {/* Bars */}
      <div style={{
        background: 'var(--ifm-card-background-color)',
        border: '1px solid rgba(255,255,255,0.07)',
        borderRadius: '10px',
        overflow: 'hidden',
      }}>
        {/* Header */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '28px 180px 1fr 80px',
          gap: '10px',
          padding: '8px 14px',
          background: 'rgba(255,255,255,0.02)',
          borderBottom: '1px solid rgba(255,255,255,0.06)',
        }}>
          {['#', 'Descripteur', 'Importance SHAP', 'Famille'].map((h) => (
            <div key={h} style={{
              fontSize: '9.5px',
              fontWeight: '700',
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
              opacity: 0.35,
              color: 'var(--ifm-font-color-base)',
            }}>{h}</div>
          ))}
        </div>

        {SHAP_FEATURES.map((f, i) => {
          const color = FAMILY_COLORS[f.family];
          const barWidth = (f.importance / maxVal) * 100;
          const isHov = hov === i;
          const isTop5Ca = f.family === 'Ca II';

          return (
            <div
              key={i}
              onMouseEnter={() => setHov(i)}
              onMouseLeave={() => setHov(null)}
              style={{
                display: 'grid',
                gridTemplateColumns: '28px 180px 1fr 80px',
                gap: '10px',
                padding: '9px 14px',
                alignItems: 'center',
                borderBottom: i < SHAP_FEATURES.length - 1
                  ? '1px solid rgba(255,255,255,0.04)' : 'none',
                background: isHov
                  ? `${color}0c`
                  : isTop5Ca
                  ? 'rgba(192,132,252,0.03)'
                  : 'transparent',
                transition: 'background 0.1s',
                cursor: 'default',
              }}
            >
              {/* Rank */}
              <div style={{
                fontFamily: 'monospace',
                fontSize: '12px',
                fontWeight: '700',
                color: color,
                opacity: isHov ? 1 : 0.65,
              }}>{f.rank}</div>

              {/* Name + note */}
              <div>
                <div style={{
                  fontSize: '12.5px',
                  fontFamily: 'monospace',
                  fontWeight: isTop5Ca ? '700' : '400',
                  color: isHov ? color : 'var(--ifm-font-color-base)',
                  transition: 'color 0.1s',
                }}>{f.name}</div>
                {isHov && (
                  <div style={{ fontSize: '10px', opacity: 0.5, marginTop: '2px', color: 'var(--ifm-font-color-base)' }}>
                    {f.note}
                  </div>
                )}
              </div>

              {/* Bar */}
              <div style={{
                height: '16px',
                background: 'rgba(255,255,255,0.04)',
                borderRadius: '4px',
                overflow: 'hidden',
              }}>
                <div style={{
                  width: `${barWidth}%`,
                  height: '100%',
                  background: isHov ? color : `${color}cc`,
                  borderRadius: '4px',
                  transition: 'background 0.1s',
                  display: 'flex',
                  alignItems: 'center',
                  paddingLeft: '6px',
                }}>
                  {barWidth > 15 && (
                    <span style={{
                      fontSize: '10px',
                      fontFamily: 'monospace',
                      fontWeight: '700',
                      color: '#000',
                    }}>{f.importance.toFixed(2)}</span>
                  )}
                </div>
              </div>

              {/* Family chip */}
              <div style={{
                background: `${color}18`,
                border: `1px solid ${color}35`,
                borderRadius: '4px',
                padding: '2px 8px',
                fontSize: '10px',
                fontWeight: '700',
                color: color,
                textAlign: 'center',
                whiteSpace: 'nowrap',
              }}>{f.family}</div>
            </div>
          );
        })}
      </div>

      <div style={{
        marginTop: '12px',
        padding: '10px 14px',
        background: 'rgba(96,165,250,0.05)',
        border: '1px solid rgba(96,165,250,0.15)',
        borderRadius: '7px',
        fontSize: '11.5px',
        color: 'var(--ifm-font-color-base)',
        opacity: 0.8,
      }}>
        <strong style={{ color: '#60A5FA' }}>97,9 %</strong> des 30 descripteurs les plus importants (top-30 SHAP) sont des features spectrales physiques — aucune feature non physique ne domine. Survole les barres pour voir l'interprétation de chaque descripteur.
      </div>
    </div>
  );
}
