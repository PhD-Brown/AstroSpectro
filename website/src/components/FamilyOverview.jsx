import React from 'react';

const FAMILIES = [
  {
    id: 'balmer',
    name: 'Raies de Balmer',
    sub: 'Hα–Hθ',
    count: 35,
    color: '#60A5FA',
    pc1Pct: 31.5,
    pc2Pct: null,
    indicator: 'Teff — axe thermique PC1',
  },
  {
    id: 'caii',
    name: 'Calcium ionisé',
    sub: 'Ca II H&K + triplet IR',
    count: 20,
    color: '#C084FC',
    pc1Pct: 22.4,
    pc2Pct: null,
    indicator: 'Métallicité + chromosphère',
  },
  {
    id: 'mg',
    name: 'Magnésium & α',
    sub: 'Mg b, Mg 5184, [α/Fe]',
    count: 15,
    color: '#34D399',
    pc1Pct: null,
    pc2Pct: null,
    indicator: '[Mg/Fe], population I/II',
  },
  {
    id: 'fe',
    name: 'Fer & métaux',
    sub: 'Fe, Cr, V, Ni, Ba, Sr...',
    count: 30,
    color: '#FB923C',
    pc1Pct: null,
    pc2Pct: 43.3,
    indicator: '[Fe/H] — axe métallicité PC2',
  },
  {
    id: 'lick',
    name: 'Indices Lick/SDSS',
    sub: 'Lick IDS, CaH, TiO',
    count: 18,
    color: '#FBBF24',
    pc1Pct: null,
    pc2Pct: null,
    indicator: 'Âge + métallicité composite',
  },
  {
    id: 'continuum',
    name: 'Continuum',
    sub: 'Pentes, courbures, couleurs',
    count: 65,
    color: '#F87171',
    pc1Pct: null,
    pc2Pct: null,
    indicator: 'Teff (pente) · luminosité (sauts)',
  },
];

const TOTAL = 183;

// Build a flat array of 183 family IDs for the mosaic
function buildMosaic() {
  const cells = [];
  for (const f of FAMILIES) {
    for (let i = 0; i < f.count; i++) {
      cells.push(f.id);
    }
  }
  return cells;
}

const MOSAIC = buildMosaic();
const COLS = 15;

export default function FamilyOverview() {
  const [hoveredFamily, setHoveredFamily] = React.useState(null);

  const familyMap = {};
  for (const f of FAMILIES) familyMap[f.id] = f;

  return (
    <div style={{ margin: '28px 0' }}>
      {/* Stacked bar */}
      <div style={{
        display: 'flex',
        height: '10px',
        borderRadius: '5px',
        overflow: 'hidden',
        marginBottom: '20px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
      }}>
        {FAMILIES.map((f) => (
          <div
            key={f.id}
            title={`${f.name} : ${f.count} descripteurs (${((f.count / TOTAL) * 100).toFixed(1)}%)`}
            style={{
              flex: `0 0 ${(f.count / TOTAL) * 100}%`,
              background: f.color,
              opacity: hoveredFamily && hoveredFamily !== f.id ? 0.3 : 1,
              transition: 'opacity 0.15s',
              cursor: 'default',
            }}
          />
        ))}
      </div>

      {/* Mosaic grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: `repeat(${COLS}, 1fr)`,
        gap: '3px',
        marginBottom: '20px',
        padding: '14px',
        background: 'rgba(255,255,255,0.02)',
        border: '1px solid rgba(255,255,255,0.06)',
        borderRadius: '10px',
      }}>
        {MOSAIC.map((fid, i) => {
          const f = familyMap[fid];
          const isHov = hoveredFamily === fid;
          return (
            <div
              key={i}
              onMouseEnter={() => setHoveredFamily(fid)}
              onMouseLeave={() => setHoveredFamily(null)}
              title={f.name}
              style={{
                aspectRatio: '1',
                borderRadius: '2px',
                background: f.color,
                opacity: hoveredFamily && !isHov ? 0.2 : isHov ? 1 : 0.7,
                transition: 'opacity 0.1s, transform 0.1s',
                transform: isHov ? 'scale(1.3)' : 'scale(1)',
                cursor: 'default',
              }}
            />
          );
        })}
      </div>

      {/* Family stat cards */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
        gap: '10px',
      }}>
        {FAMILIES.map((f) => {
          const pct = ((f.count / TOTAL) * 100).toFixed(1);
          const isHov = hoveredFamily === f.id;
          return (
            <div
              key={f.id}
              onMouseEnter={() => setHoveredFamily(f.id)}
              onMouseLeave={() => setHoveredFamily(null)}
              style={{
                display: 'flex',
                gap: '12px',
                alignItems: 'flex-start',
                padding: '12px 14px',
                background: isHov ? `${f.color}10` : 'var(--ifm-card-background-color)',
                border: `1px solid ${isHov ? f.color + '55' : f.color + '22'}`,
                borderLeft: `4px solid ${f.color}`,
                borderRadius: '8px',
                transition: 'all 0.15s',
                cursor: 'default',
              }}
            >
              {/* Count badge */}
              <div style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontWeight: '900',
                fontSize: '22px',
                color: f.color,
                lineHeight: 1,
                flexShrink: 0,
                minWidth: '36px',
                textAlign: 'right',
              }}>{f.count}</div>

              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{
                  fontWeight: '700',
                  fontSize: '13px',
                  color: f.color,
                  marginBottom: '1px',
                }}>{f.name}</div>
                <div style={{
                  fontSize: '10.5px',
                  fontFamily: 'monospace',
                  color: 'var(--ifm-font-color-base)',
                  opacity: 0.5,
                  marginBottom: '5px',
                }}>{f.sub}</div>

                {/* Mini bar */}
                <div style={{
                  height: '3px',
                  background: 'rgba(255,255,255,0.08)',
                  borderRadius: '2px',
                  marginBottom: '5px',
                  overflow: 'hidden',
                }}>
                  <div style={{
                    width: `${pct}%`,
                    height: '100%',
                    background: f.color,
                    borderRadius: '2px',
                  }} />
                </div>

                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}>
                  <div style={{
                    fontSize: '10.5px',
                    color: 'var(--ifm-font-color-base)',
                    opacity: 0.5,
                    fontStyle: 'italic',
                  }}>{f.indicator}</div>

                  {/* PCA tag */}
                  {(f.pc1Pct || f.pc2Pct) && (
                    <div style={{
                      background: `${f.color}18`,
                      border: `1px solid ${f.color}40`,
                      borderRadius: '4px',
                      padding: '1px 6px',
                      fontSize: '10px',
                      fontWeight: '700',
                      color: f.color,
                      whiteSpace: 'nowrap',
                      marginLeft: '6px',
                      flexShrink: 0,
                    }}>
                      {f.pc1Pct ? `PC1 ${f.pc1Pct}%` : `PC2 ${f.pc2Pct}%`}
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div style={{
        marginTop: '10px',
        textAlign: 'center',
        fontSize: '11px',
        opacity: 0.35,
        color: 'var(--ifm-font-color-base)',
      }}>
        Chaque case = 1 descripteur spectroscopique · 183 total · survoler une famille pour la mettre en évidence
      </div>
    </div>
  );
}
