import React from 'react';

const SPECS = [
  {
    group: 'Télescope',
    color: '#38BDF8',
    fields: [
      { label: 'Nom complet', value: 'Large Sky Area Multi-Object Fiber Spectroscopic Telescope', mono: false },
      { label: 'Localisation', value: 'Station astronomique de Xinglong, Hebei, Chine (alt. 900 m)', mono: false },
      { label: 'Miroir principal', value: '4 m × 3,7 m (segmenté, actif)', mono: false },
      { label: 'Champ de vue', value: '5° de diamètre — le plus grand au monde', mono: false },
      { label: 'Fibres optiques', value: '4 000 fibres simultanées', mono: false },
    ],
  },
  {
    group: 'Spectral',
    color: '#A78BFA',
    fields: [
      { label: 'Couverture spectrale', value: '3 690–9 100 Å (bras bleu + bras rouge)', mono: true },
      { label: 'Résolution spectrale', value: 'R ≈ 1 800 (basse résolution)', mono: true },
      { label: 'Canaux par spectre (flux)', value: '3 921 canaux', mono: true },
      { label: 'Coupure instrumentale', value: '~5 900 Å (séparation bras bleu/rouge)', mono: true },
    ],
  },
  {
    group: 'Data Release 5',
    color: '#34D399',
    fields: [
      { label: 'Spectres total DR5', value: '> 9 millions de spectres', mono: false },
      { label: 'Période observationnelle', value: '2011–2017', mono: false },
      { label: 'Format de distribution', value: '.fits.gz (FITS compressé gzip)', mono: true },
      { label: 'Accès public', value: 'http://dr5.lamost.org/', mono: true },
    ],
  },
  {
    group: 'AstroSpectro — sous-ensemble',
    color: '#F59E0B',
    fields: [
      { label: 'Spectres téléchargés', value: '~250 000 fichiers FITS', mono: false },
      { label: 'Après filtrage qualité', value: 'N = 43 019 (SNR > 10 · RUWE < 1,4)', mono: false },
      { label: 'Descripteurs extraits', value: 'p = 183 (mode spectro_only=True)', mono: false },
      { label: 'Croisement Gaia DR3', value: 'Rayon 1″ · ~100 % des étoiles appariées', mono: false },
    ],
  },
];

export default function LAMOSTSpecs() {
  const [activeGroup, setActiveGroup] = React.useState(null);

  return (
    <div style={{ margin: '28px 0' }}>
      {/* Tab headers */}
      <div style={{
        display: 'flex',
        gap: '4px',
        flexWrap: 'wrap',
        marginBottom: '0',
        borderBottom: '1px solid rgba(255,255,255,0.07)',
      }}>
        {SPECS.map((g) => {
          const isActive = activeGroup === g.group || (activeGroup === null && g.group === SPECS[0].group);
          return (
            <button
              key={g.group}
              onClick={() => setActiveGroup(g.group)}
              style={{
                background: isActive ? `${g.color}14` : 'transparent',
                border: 'none',
                borderBottom: `3px solid ${isActive ? g.color : 'transparent'}`,
                padding: '9px 16px',
                cursor: 'pointer',
                fontSize: '12px',
                fontWeight: isActive ? '700' : '400',
                color: isActive ? g.color : 'var(--ifm-font-color-base)',
                opacity: isActive ? 1 : 0.5,
                borderRadius: '6px 6px 0 0',
                marginBottom: '-1px',
                transition: 'all 0.13s',
              }}
            >{g.group}</button>
          );
        })}
      </div>

      {SPECS.map((g) => {
        const isActive = activeGroup === g.group || (activeGroup === null && g.group === SPECS[0].group);
        if (!isActive) return null;

        return (
          <div key={g.group} style={{
            background: 'var(--ifm-card-background-color)',
            border: `1px solid ${g.color}25`,
            borderTop: 'none',
            borderRadius: '0 0 10px 10px',
          }}>
            {g.fields.map((f, i) => (
              <div key={i} style={{
                display: 'grid',
                gridTemplateColumns: '220px 1fr',
                alignItems: 'center',
                gap: '16px',
                padding: '12px 18px',
                borderBottom: i < g.fields.length - 1 ? '1px solid rgba(255,255,255,0.04)' : 'none',
                background: i % 2 === 0 ? 'rgba(255,255,255,0.01)' : 'transparent',
              }}>
                <div style={{
                  fontSize: '12px',
                  color: 'var(--ifm-font-color-base)',
                  opacity: 0.55,
                  fontWeight: '500',
                }}>{f.label}</div>
                <div style={{
                  fontFamily: f.mono ? "'JetBrains Mono', monospace" : 'inherit',
                  fontSize: f.mono ? '13px' : '14px',
                  fontWeight: '600',
                  color: g.color,
                }}>{f.value}</div>
              </div>
            ))}
          </div>
        );
      })}
    </div>
  );
}
