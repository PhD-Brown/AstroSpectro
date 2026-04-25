import React from 'react';

const WL_MIN = 3690;
const WL_MAX = 9100;
function wlPct(lambda) {
  return ((lambda - WL_MIN) / (WL_MAX - WL_MIN)) * 100;
}

const SPECIES_COLORS = {
  'TiO': '#F87171',
  'VO':  '#FB923C',
  'CN':  '#FBBF24',
  'CH':  '#34D399',
  'CaH': '#60A5FA',
  'MgH': '#C084FC',
};

const BANDS = [
  { name: 'TiO 6180', center: 6205, hw: 50,  species: 'TiO' },
  { name: 'TiO 7050', center: 7100, hw: 100, species: 'TiO' },
  { name: 'TiO 7600', center: 7650, hw: 100, species: 'TiO' },
  { name: 'TiO 8200', center: 8250, hw: 100, species: 'TiO' },
  { name: 'TiO 8400', center: 8450, hw: 100, species: 'TiO' },
  { name: 'TiO 8860', center: 8885, hw: 50,  species: 'TiO' },
  { name: 'VO 7400',  center: 7475, hw: 150, species: 'VO'  },
  { name: 'VO 7900',  center: 7975, hw: 150, species: 'VO'  },
  { name: 'CN 4142',  center: 4160, hw: 35,  species: 'CN'  },
  { name: 'CN 8200',  center: 8250, hw: 100, species: 'CN'  },
  { name: 'CH 4300',  center: 4305, hw: 25,  species: 'CH'  },
  { name: 'CaH 6380', center: 6385, hw: 10,  species: 'CaH' },
  { name: 'CaH 6830', center: 6835, hw: 10,  species: 'CaH' },
  { name: 'CaH 6975', center: 6980, hw: 10,  species: 'CaH' },
  { name: 'MgH 5140', center: 5145, hw: 10,  species: 'MgH' },
];

const SPECIES_INFO = {
  'TiO': { label: 'Oxyde de titane',  note: 'Étoiles M — naines K tardives', starType: 'K–M' },
  'VO':  { label: 'Oxyde de vanadium',note: 'Étoiles M les plus froides',    starType: 'M'   },
  'CN':  { label: 'Cyanure',          note: 'Géantes — évolution stellaire',  starType: 'Géantes' },
  'CH':  { label: 'Méthylidyne',      note: 'Bande G — sous-géantes G',      starType: 'G' },
  'CaH': { label: 'Hydrure de calcium',note: 'Sous-types M — naines froides', starType: 'M' },
  'MgH': { label: 'Hydrure de magnésium',note: 'Étoiles froides / métallicité',starType: 'K–M' },
};

function BandChip({ name, center, hw, species }) {
  const [hov, setHov] = React.useState(false);
  const color = SPECIES_COLORS[species];

  return (
    <div
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        background: hov ? `${color}20` : `${color}0c`,
        border: `1px solid ${hov ? color + '66' : color + '33'}`,
        borderRadius: '6px',
        padding: '6px 10px',
        cursor: 'default',
        transition: 'all 0.12s',
        transform: hov ? 'translateY(-1px)' : 'none',
        boxShadow: hov ? `0 3px 10px ${color}20` : 'none',
      }}
    >
      <div style={{
        fontFamily: 'monospace',
        fontSize: '12px',
        fontWeight: '700',
        color: color,
        marginBottom: '3px',
      }}>{name}</div>
      <div style={{
        fontSize: '10px',
        color: 'var(--ifm-font-color-base)',
        opacity: 0.5,
        fontFamily: 'monospace',
      }}>
        {center - hw}–{center + hw} Å
      </div>
    </div>
  );
}

export default function MolecularBands() {
  const speciesGroups = {};
  for (const b of BANDS) {
    if (!speciesGroups[b.species]) speciesGroups[b.species] = [];
    speciesGroups[b.species].push(b);
  }

  return (
    <div style={{ margin: '20px 0' }}>
      {/* Spectrum position visual */}
      <div style={{
        background: 'rgba(255,255,255,0.02)',
        border: '1px solid rgba(255,255,255,0.06)',
        borderRadius: '10px',
        padding: '14px 16px',
        marginBottom: '16px',
      }}>
        <div style={{ fontSize: '10px', opacity: 0.35, marginBottom: '10px', color: 'var(--ifm-font-color-base)', fontFamily: 'monospace' }}>
          Position des bandes dans 3690–9100 Å
        </div>
        <div style={{ position: 'relative', height: '20px' }}>
          {/* Track */}
          <div style={{
            position: 'absolute',
            left: 0, right: 0, top: '50%',
            height: '2px',
            background: 'rgba(255,255,255,0.06)',
            transform: 'translateY(-50%)',
          }} />
          {/* Band segments */}
          {BANDS.map((b, i) => {
            const color = SPECIES_COLORS[b.species];
            const leftPct = wlPct(b.center - b.hw);
            const widthPct = Math.max(wlPct(b.center + b.hw) - leftPct, 0.4);
            return (
              <div key={i} title={`${b.name}: ${b.center - b.hw}–${b.center + b.hw} Å`} style={{
                position: 'absolute',
                left: `${leftPct}%`,
                width: `${widthPct}%`,
                top: '20%',
                height: '60%',
                background: color,
                borderRadius: '2px',
                opacity: 0.7,
              }} />
            );
          })}
          {/* WL ticks */}
          {[4000, 5000, 6000, 7000, 8000, 9000].map((wl) => (
            <div key={wl} style={{
              position: 'absolute',
              left: `${wlPct(wl)}%`,
              top: '100%',
              marginTop: '3px',
              fontSize: '8.5px',
              fontFamily: 'monospace',
              color: 'var(--ifm-font-color-base)',
              opacity: 0.25,
              transform: 'translateX(-50%)',
            }}>{wl}</div>
          ))}
        </div>
      </div>

      {/* Species groups */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        {Object.entries(speciesGroups).map(([sp, bands]) => {
          const color = SPECIES_COLORS[sp];
          const info = SPECIES_INFO[sp];
          return (
            <div key={sp} style={{
              display: 'grid',
              gridTemplateColumns: '120px 1fr',
              gap: '12px',
              alignItems: 'start',
              padding: '12px 14px',
              background: 'var(--ifm-card-background-color)',
              border: `1px solid ${color}22`,
              borderLeft: `4px solid ${color}`,
              borderRadius: '8px',
            }}>
              {/* Species info */}
              <div>
                <div style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  fontWeight: '900',
                  fontSize: '18px',
                  color: color,
                  lineHeight: 1,
                  marginBottom: '4px',
                }}>{sp}</div>
                <div style={{
                  fontSize: '11px',
                  color: 'var(--ifm-font-color-base)',
                  opacity: 0.55,
                  marginBottom: '3px',
                  lineHeight: 1.4,
                }}>{info.label}</div>
                <div style={{
                  background: `${color}18`,
                  border: `1px solid ${color}35`,
                  borderRadius: '4px',
                  padding: '1px 7px',
                  fontSize: '10px',
                  fontWeight: '600',
                  color: color,
                  display: 'inline-block',
                }}>{info.starType}</div>
                <div style={{
                  fontSize: '10px',
                  color: 'var(--ifm-font-color-base)',
                  opacity: 0.4,
                  marginTop: '4px',
                  fontStyle: 'italic',
                }}>{info.note}</div>
              </div>

              {/* Band chips */}
              <div style={{
                display: 'flex',
                flexWrap: 'wrap',
                gap: '6px',
                alignContent: 'flex-start',
              }}>
                {bands.map((b) => (
                  <BandChip key={b.name} {...b} />
                ))}
              </div>
            </div>
          );
        })}
      </div>

      <div style={{
        marginTop: '12px',
        padding: '10px 14px',
        background: 'rgba(248,113,113,0.06)',
        border: '1px solid rgba(248,113,113,0.2)',
        borderRadius: '7px',
        fontSize: '12px',
        color: 'var(--ifm-font-color-base)',
        opacity: 0.8,
      }}>
        <strong style={{ color: '#F87171' }}>Descripteur extrait par bande :</strong> rapport{' '}
        <code>flux_bande / flux_continuum</code>. Ces bandes apparaissent dans les clusters HDBSCAN correspondant aux naines K-M dans l'espace UMAP.
      </div>
    </div>
  );
}
