import React from 'react';

const WL_MIN = 3690;
const WL_MAX = 9100;
const SPAN = WL_MAX - WL_MIN;

function wlToX(lambda) {
  return ((lambda - WL_MIN) / SPAN) * 100;
}

// Spectral gradient segments (approximate visible + near-IR colors)
const SPECTRUM_SEGMENTS = [
  { from: 3690, to: 4000, color: '#6B21A8' },  // deep violet/UV
  { from: 4000, to: 4500, color: '#7C3AED' },  // violet
  { from: 4500, to: 5000, color: '#2563EB' },  // blue
  { from: 5000, to: 5500, color: '#0891B2' },  // cyan-blue
  { from: 5500, to: 5800, color: '#16A34A' },  // green
  { from: 5800, to: 6000, color: '#CA8A04' },  // yellow-green
  { from: 6000, to: 6500, color: '#EA580C' },  // orange
  { from: 6500, to: 7000, color: '#DC2626' },  // red
  { from: 7000, to: 9100, color: '#7F1D1D' },  // deep red / IR
];

// Key spectral lines by family
const LINES = [
  // Balmer (blue) - label top row
  { lambda: 3797.9, label: 'H10', family: 'balmer', color: '#60A5FA', row: 1 },
  { lambda: 3835.4, label: 'H9',  family: 'balmer', color: '#60A5FA', row: 1 },
  { lambda: 3889.0, label: 'H8',  family: 'balmer', color: '#60A5FA', row: 1 },
  { lambda: 3970.1, label: 'Hε',  family: 'balmer', color: '#60A5FA', row: 1 },
  { lambda: 4101.7, label: 'Hδ',  family: 'balmer', color: '#60A5FA', row: 1 },
  { lambda: 4340.5, label: 'Hγ',  family: 'balmer', color: '#60A5FA', row: 1 },
  { lambda: 4861.3, label: 'Hβ',  family: 'balmer', color: '#60A5FA', row: 1 },
  { lambda: 6562.8, label: 'Hα',  family: 'balmer', color: '#60A5FA', row: 1, important: true },

  // Ca II (violet) - row 2
  { lambda: 3933.7, label: 'Ca K', family: 'caii',   color: '#C084FC', row: 2, important: true },
  { lambda: 3968.5, label: 'Ca H', family: 'caii',   color: '#C084FC', row: 2, important: true },
  { lambda: 8498.0, label: 'IR1',  family: 'caii',   color: '#C084FC', row: 2 },
  { lambda: 8542.1, label: 'IR2',  family: 'caii',   color: '#C084FC', row: 2 },
  { lambda: 8662.1, label: 'IR3',  family: 'caii',   color: '#C084FC', row: 2 },

  // Mg (green) - row 2
  { lambda: 5167.0, label: 'Mg b', family: 'mg',     color: '#34D399', row: 2, important: true },
  { lambda: 5184.0, label: '',      family: 'mg',     color: '#34D399', row: 2 },

  // Fe/metals (orange) - row 3
  { lambda: 4383.5, label: 'Fe',   family: 'fe',     color: '#FB923C', row: 3 },
  { lambda: 5270.4, label: 'Fe',   family: 'fe',     color: '#FB923C', row: 3, important: true },
  { lambda: 5335.2, label: 'Fe',   family: 'fe',     color: '#FB923C', row: 3 },
  { lambda: 5895.9, label: 'Na D', family: 'fe',     color: '#FB923C', row: 3, important: true },

  // Molecular bands (red/IR) - row 3
  { lambda: 6205.0, label: 'TiO',  family: 'mol',    color: '#F87171', row: 3 },
  { lambda: 7100.0, label: 'TiO',  family: 'mol',    color: '#F87171', row: 3 },
  { lambda: 7650.0, label: 'TiO',  family: 'mol',    color: '#F87171', row: 3, important: true },
  { lambda: 8250.0, label: 'TiO',  family: 'mol',    color: '#F87171', row: 3 },
];

// LAMOST instrument split
const INSTRUMENT_SPLIT = 5900; // approximate blue/red spectrograph split

const FAMILY_LEGEND = [
  { id: 'balmer', label: 'Balmer (H)',        color: '#60A5FA', row: 1 },
  { id: 'caii',   label: 'Ca II H&K + IR',    color: '#C084FC', row: 2 },
  { id: 'mg',     label: 'Mg b',              color: '#34D399', row: 2 },
  { id: 'fe',     label: 'Fer & métaux',      color: '#FB923C', row: 3 },
  { id: 'mol',    label: 'Bandes moléculaires', color: '#F87171', row: 3 },
];

export default function SpectralCoverage() {
  const [hovLine, setHovLine] = React.useState(null);

  return (
    <div style={{
      margin: '28px 0',
      background: 'var(--ifm-card-background-color)',
      border: '1px solid rgba(255,255,255,0.07)',
      borderRadius: '12px',
      padding: '20px 24px',
      overflow: 'hidden',
    }}>
      <div style={{
        fontSize: '11px',
        fontWeight: '700',
        textTransform: 'uppercase',
        letterSpacing: '0.08em',
        opacity: 0.4,
        color: 'var(--ifm-font-color-base)',
        marginBottom: '16px',
      }}>
        Couverture spectrale LAMOST DR5 — 3 690–9 100 Å — positions des familles de raies
      </div>

      {/* Spectrum bar */}
      <div style={{ position: 'relative', marginBottom: '8px' }}>
        <div style={{
          display: 'flex',
          height: '28px',
          borderRadius: '6px',
          overflow: 'hidden',
          position: 'relative',
        }}>
          {SPECTRUM_SEGMENTS.map((seg, i) => (
            <div
              key={i}
              style={{
                flex: `0 0 ${((seg.to - seg.from) / SPAN) * 100}%`,
                background: seg.color,
                opacity: 0.55,
              }}
            />
          ))}
          {/* Instrument split line */}
          <div style={{
            position: 'absolute',
            left: `${wlToX(INSTRUMENT_SPLIT)}%`,
            top: 0,
            bottom: 0,
            width: '2px',
            background: 'rgba(255,255,255,0.5)',
          }} />
        </div>

        {/* Instrument label */}
        <div style={{
          position: 'absolute',
          left: `${wlToX(INSTRUMENT_SPLIT)}%`,
          top: '30px',
          transform: 'translateX(-50%)',
          fontSize: '9px',
          color: 'rgba(255,255,255,0.45)',
          whiteSpace: 'nowrap',
          fontFamily: 'monospace',
        }}>bras bleu | bras rouge</div>
      </div>

      {/* Spectral line rows */}
      {[1, 2, 3].map((row) => (
        <div key={row} style={{
          position: 'relative',
          height: '36px',
          marginTop: '4px',
        }}>
          {LINES.filter((l) => l.row === row).map((line, i) => {
            const x = wlToX(line.lambda);
            const isHov = hovLine === `${line.family}-${line.lambda}`;
            return (
              <div
                key={i}
                onMouseEnter={() => setHovLine(`${line.family}-${line.lambda}`)}
                onMouseLeave={() => setHovLine(null)}
                style={{
                  position: 'absolute',
                  left: `${x}%`,
                  top: 0,
                  transform: 'translateX(-50%)',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  cursor: 'default',
                  zIndex: isHov ? 10 : 1,
                }}
              >
                {/* Tick mark */}
                <div style={{
                  width: line.important ? '2px' : '1px',
                  height: line.important ? '14px' : '10px',
                  background: line.color,
                  opacity: isHov ? 1 : line.important ? 0.9 : 0.55,
                  transition: 'opacity 0.1s',
                }} />

                {/* Label */}
                {line.label && (
                  <div style={{
                    fontSize: isHov ? '10px' : '8.5px',
                    fontFamily: 'monospace',
                    color: line.color,
                    opacity: isHov ? 1 : line.important ? 0.85 : 0.55,
                    whiteSpace: 'nowrap',
                    marginTop: '1px',
                    fontWeight: line.important ? '700' : '400',
                    transition: 'all 0.1s',
                    background: isHov ? 'var(--ifm-card-background-color)' : 'transparent',
                    padding: isHov ? '0 3px' : '0',
                    borderRadius: '2px',
                  }}>
                    {isHov ? `${line.label} ${line.lambda}Å` : line.label}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      ))}

      {/* Wavelength axis */}
      <div style={{
        position: 'relative',
        height: '20px',
        marginTop: '8px',
        borderTop: '1px solid rgba(255,255,255,0.08)',
      }}>
        {[3690, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9100].map((wl) => (
          <div
            key={wl}
            style={{
              position: 'absolute',
              left: `${wlToX(wl)}%`,
              top: '4px',
              transform: 'translateX(-50%)',
              fontSize: '9px',
              fontFamily: 'monospace',
              color: 'var(--ifm-font-color-base)',
              opacity: 0.35,
              whiteSpace: 'nowrap',
            }}
          >
            {wl}
          </div>
        ))}
      </div>

      {/* Legend */}
      <div style={{
        display: 'flex',
        gap: '14px',
        flexWrap: 'wrap',
        marginTop: '14px',
        paddingTop: '12px',
        borderTop: '1px solid rgba(255,255,255,0.06)',
      }}>
        {FAMILY_LEGEND.map((f) => (
          <div key={f.id} style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <div style={{
              width: '10px',
              height: '10px',
              borderRadius: '2px',
              background: f.color,
              flexShrink: 0,
            }} />
            <span style={{ fontSize: '11px', color: 'var(--ifm-font-color-base)', opacity: 0.65 }}>
              {f.label}
            </span>
          </div>
        ))}
        <div style={{
          marginLeft: 'auto',
          fontSize: '10px',
          color: 'var(--ifm-font-color-base)',
          opacity: 0.3,
          fontFamily: 'monospace',
          alignSelf: 'center',
        }}>Å</div>
      </div>
    </div>
  );
}
