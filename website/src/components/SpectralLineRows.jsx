import React from 'react';

const WL_MIN = 3690;
const WL_MAX = 9100;

function wlPct(lambda) {
  return ((lambda - WL_MIN) / (WL_MAX - WL_MIN)) * 100;
}

function LambdaBar({ lambda, color }) {
  const pct = wlPct(lambda);
  return (
    <div style={{
      position: 'relative',
      width: '100%',
      height: '4px',
      background: 'rgba(255,255,255,0.05)',
      borderRadius: '2px',
    }}>
      <div style={{
        position: 'absolute',
        left: `${pct}%`,
        top: '50%',
        width: '8px',
        height: '8px',
        borderRadius: '50%',
        background: color,
        transform: 'translate(-50%, -50%)',
        boxShadow: `0 0 0 2px ${color}33`,
      }} />
    </div>
  );
}

/**
 * SpectralLineRows — generic spectral line table component
 *
 * Props:
 *   color    : hex color for the family
 *   columns  : array of column header strings e.g. ['Raie','λ (Å)','Espèce','Descripteurs extraits']
 *   rows     : array of row objects, each with:
 *                label     : string (row name)
 *                lambda    : number|null (wavelength for position bar)
 *                tag       : string|null (species chip text)
 *                desc      : string (main descriptor text)
 *                highlight : bool (bold row)
 *   footer   : string|null (small note below table)
 */
export default function SpectralLineRows({ color = '#60A5FA', columns, rows, footer }) {
  const [hovIdx, setHovIdx] = React.useState(null);

  return (
    <div style={{ margin: '14px 0' }}>
      <div style={{
        border: '1px solid rgba(255,255,255,0.07)',
        borderRadius: '9px',
        overflow: 'hidden',
      }}>
        {/* Header */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: columns.map((_, i) => {
            if (i === 0) return '90px';
            if (columns[i] === 'λ (Å)') return '80px';
            if (columns[i] === 'Espèce') return '60px';
            if (columns[i] === 'Position') return '80px';
            return '1fr';
          }).join(' '),
          background: `${color}12`,
          borderBottom: `2px solid ${color}30`,
          padding: '7px 14px',
          gap: '10px',
        }}>
          {columns.map((col) => (
            <div key={col} style={{
              fontSize: '10px',
              fontWeight: '700',
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
              color: color,
              opacity: 0.75,
            }}>{col}</div>
          ))}
        </div>

        {/* Rows */}
        {rows.map((row, i) => {
          const isHov = hovIdx === i;
          const hasLambda = row.lambda != null;
          return (
            <div
              key={i}
              onMouseEnter={() => setHovIdx(i)}
              onMouseLeave={() => setHovIdx(null)}
              style={{
                display: 'grid',
                gridTemplateColumns: columns.map((col, ci) => {
                  if (ci === 0) return '90px';
                  if (col === 'λ (Å)') return '80px';
                  if (col === 'Espèce') return '60px';
                  if (col === 'Position') return '80px';
                  return '1fr';
                }).join(' '),
                alignItems: 'center',
                gap: '10px',
                padding: '8px 14px',
                borderBottom: i < rows.length - 1 ? '1px solid rgba(255,255,255,0.04)' : 'none',
                background: isHov
                  ? `${color}09`
                  : row.highlight
                  ? `${color}05`
                  : i % 2 === 0 ? 'rgba(255,255,255,0.01)' : 'transparent',
                transition: 'background 0.1s',
                cursor: 'default',
              }}
            >
              {columns.map((col, ci) => {
                if (ci === 0) {
                  return (
                    <div key={ci} style={{
                      fontFamily: 'monospace',
                      fontSize: '12.5px',
                      fontWeight: row.highlight ? '800' : '600',
                      color: isHov ? color : 'var(--ifm-font-color-base)',
                      transition: 'color 0.1s',
                    }}>{row.label}</div>
                  );
                }
                if (col === 'λ (Å)') {
                  return (
                    <div key={ci} style={{
                      fontFamily: 'monospace',
                      fontSize: '12px',
                      fontWeight: '600',
                      color: color,
                      opacity: 0.9,
                    }}>{row.lambda != null ? row.lambda.toFixed(1) : '—'}</div>
                  );
                }
                if (col === 'Position') {
                  return (
                    <div key={ci} style={{ paddingRight: '4px' }}>
                      {hasLambda
                        ? <LambdaBar lambda={row.lambda} color={color} />
                        : <div style={{ height: '4px' }} />
                      }
                    </div>
                  );
                }
                if (col === 'Espèce') {
                  return (
                    <div key={ci}>
                      {row.tag ? (
                        <span style={{
                          background: `${color}18`,
                          border: `1px solid ${color}35`,
                          borderRadius: '4px',
                          padding: '1px 6px',
                          fontSize: '10px',
                          fontFamily: 'monospace',
                          fontWeight: '700',
                          color: color,
                        }}>{row.tag}</span>
                      ) : <span style={{ opacity: 0.3 }}>—</span>}
                    </div>
                  );
                }
                // Default: descriptor text
                return (
                  <div key={ci} style={{
                    fontSize: '12px',
                    color: 'var(--ifm-font-color-base)',
                    opacity: isHov ? 0.9 : 0.65,
                    lineHeight: 1.4,
                    transition: 'opacity 0.1s',
                  }}>{row.desc}</div>
                );
              })}
            </div>
          );
        })}
      </div>

      {footer && (
        <div style={{
          marginTop: '8px',
          fontSize: '11px',
          color: 'var(--ifm-font-color-base)',
          opacity: 0.5,
          fontStyle: 'italic',
          paddingLeft: '4px',
        }}>{footer}</div>
      )}
    </div>
  );
}
