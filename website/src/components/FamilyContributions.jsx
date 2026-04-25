import React from 'react';

const FAMILIES = [
  { name: 'Raies de Balmer', sub: 'H α–θ', pc1: 31.5, pc2:  8.2, color: '#60A5FA', pc1dom: true  },
  { name: 'Calcium ionisé',  sub: 'Ca II H&K + IR', pc1: 22.4, pc2: 12.1, color: '#C084FC', pc1dom: true  },
  { name: 'Fer & métaux',    sub: 'Fe, Cr, V, Ni…', pc1: 15.3, pc2: 43.3, color: '#FB923C', pc2dom: true  },
  { name: 'Magnésium / α',  sub: 'Mg b, [Mg/Fe]', pc1: 10.8, pc2: 18.4, color: '#34D399', pc2dom: true  },
  { name: 'Indices Lick',    sub: 'Lick IDS, SDSS', pc1:  9.7, pc2: 11.6, color: '#FBBF24', neutral: true },
  { name: 'Continuum',       sub: 'pentes, couleurs', pc1: 10.3, pc2:  6.4, color: '#F87171', pc1dom: true  },
];

const MAX_PCT = 50; // bar scale max

const PC1_COLOR = '#38BDF8';
const PC2_COLOR = '#F59E0B';

function Row({ name, sub, pc1, pc2, color, pc1dom, pc2dom, neutral }) {
  const [hov, setHov] = React.useState(false);
  const pc1Width = (pc1 / MAX_PCT) * 100;
  const pc2Width = (pc2 / MAX_PCT) * 100;

  return (
    <div
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'grid',
        gridTemplateColumns: '160px 1fr 1fr',
        alignItems: 'center',
        gap: '0',
        padding: '10px 14px',
        borderBottom: '1px solid rgba(255,255,255,0.04)',
        background: hov ? `${color}08` : 'transparent',
        transition: 'background 0.12s',
        cursor: 'default',
      }}
    >
      {/* Family name */}
      <div>
        <div style={{
          fontSize: '13px',
          fontWeight: '600',
          color: hov ? color : 'var(--ifm-font-color-base)',
          transition: 'color 0.12s',
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
        }}>
          <div style={{ width: '8px', height: '8px', borderRadius: '2px', background: color, flexShrink: 0 }} />
          {name}
        </div>
        <div style={{ fontSize: '10px', opacity: 0.4, marginLeft: '14px', color: 'var(--ifm-font-color-base)' }}>
          {sub}
        </div>
      </div>

      {/* PC1 bar */}
      <div style={{ paddingRight: '8px', paddingLeft: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{
            flex: 1,
            height: '18px',
            background: 'rgba(255,255,255,0.04)',
            borderRadius: '4px',
            overflow: 'hidden',
          }}>
            <div style={{
              width: `${pc1Width}%`,
              height: '100%',
              background: pc1dom ? PC1_COLOR : `${PC1_COLOR}60`,
              borderRadius: '4px',
              transition: 'width 0.3s ease',
              display: 'flex',
              alignItems: 'center',
              paddingLeft: '6px',
            }}>
              {pc1 > 8 && (
                <span style={{
                  fontSize: '11px',
                  fontFamily: 'monospace',
                  fontWeight: pc1dom ? '800' : '400',
                  color: pc1dom ? '#000' : 'rgba(0,0,0,0.6)',
                  whiteSpace: 'nowrap',
                }}>{pc1}%</span>
              )}
            </div>
          </div>
          {pc1 <= 8 && (
            <span style={{
              fontSize: '11px',
              fontFamily: 'monospace',
              color: PC1_COLOR,
              opacity: 0.7,
              minWidth: '32px',
            }}>{pc1}%</span>
          )}
          {pc1dom && (
            <span style={{ fontSize: '9px', color: PC1_COLOR, fontWeight: '700', whiteSpace: 'nowrap' }}>★ dom.</span>
          )}
        </div>
      </div>

      {/* PC2 bar */}
      <div style={{ paddingLeft: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{
            flex: 1,
            height: '18px',
            background: 'rgba(255,255,255,0.04)',
            borderRadius: '4px',
            overflow: 'hidden',
          }}>
            <div style={{
              width: `${pc2Width}%`,
              height: '100%',
              background: pc2dom ? PC2_COLOR : `${PC2_COLOR}60`,
              borderRadius: '4px',
              transition: 'width 0.3s ease',
              display: 'flex',
              alignItems: 'center',
              paddingLeft: '6px',
            }}>
              {pc2 > 8 && (
                <span style={{
                  fontSize: '11px',
                  fontFamily: 'monospace',
                  fontWeight: pc2dom ? '800' : '400',
                  color: pc2dom ? '#000' : 'rgba(0,0,0,0.6)',
                  whiteSpace: 'nowrap',
                }}>{pc2}%</span>
              )}
            </div>
          </div>
          {pc2 <= 8 && (
            <span style={{
              fontSize: '11px',
              fontFamily: 'monospace',
              color: PC2_COLOR,
              opacity: 0.7,
              minWidth: '32px',
            }}>{pc2}%</span>
          )}
          {pc2dom && (
            <span style={{ fontSize: '9px', color: PC2_COLOR, fontWeight: '700', whiteSpace: 'nowrap' }}>★ dom.</span>
          )}
        </div>
      </div>
    </div>
  );
}

export default function FamilyContributions() {
  return (
    <div style={{ margin: '28px 0' }}>
      <div style={{
        border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: '10px',
        overflow: 'hidden',
      }}>
        {/* Column headers */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '160px 1fr 1fr',
          background: 'rgba(255,255,255,0.03)',
          borderBottom: '1px solid rgba(255,255,255,0.08)',
          padding: '10px 14px',
          gap: '0',
        }}>
          <div style={{ fontSize: '10px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.06em', opacity: 0.4, color: 'var(--ifm-font-color-base)' }}>
            Famille
          </div>
          <div style={{
            paddingLeft: '8px',
            fontSize: '12px',
            fontWeight: '800',
            color: PC1_COLOR,
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}>
            <div style={{ width: '10px', height: '10px', borderRadius: '2px', background: PC1_COLOR }} />
            PC1 — Axe thermique
          </div>
          <div style={{
            paddingLeft: '8px',
            fontSize: '12px',
            fontWeight: '800',
            color: PC2_COLOR,
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}>
            <div style={{ width: '10px', height: '10px', borderRadius: '2px', background: PC2_COLOR }} />
            PC2 — Axe métallicité
          </div>
        </div>

        {/* Scale note */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '160px 1fr 1fr',
          padding: '4px 14px',
          background: 'rgba(255,255,255,0.01)',
          borderBottom: '1px solid rgba(255,255,255,0.04)',
        }}>
          <div />
          <div style={{ paddingLeft: '8px', fontSize: '9px', opacity: 0.3, color: 'var(--ifm-font-color-base)', fontFamily: 'monospace' }}>0%{'─'.repeat(10)}→ {MAX_PCT}%</div>
          <div style={{ paddingLeft: '8px', fontSize: '9px', opacity: 0.3, color: 'var(--ifm-font-color-base)', fontFamily: 'monospace' }}>0%{'─'.repeat(10)}→ {MAX_PCT}%</div>
        </div>

        {/* Rows */}
        {FAMILIES.map((f) => <Row key={f.name} {...f} />)}
      </div>

      {/* Interpretation */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '10px',
        marginTop: '14px',
      }}>
        <div style={{
          padding: '10px 14px',
          background: 'rgba(56,189,248,0.06)',
          border: '1px solid rgba(56,189,248,0.2)',
          borderRadius: '7px',
          fontSize: '12px',
          color: 'var(--ifm-font-color-base)',
          opacity: 0.8,
        }}>
          <strong style={{ color: '#38BDF8' }}>PC1 thermique :</strong> Balmer (31,5 %) + Ca II (22,4 %) = 53,9 % — les indicateurs de température dominent sans ambiguïté.
        </div>
        <div style={{
          padding: '10px 14px',
          background: 'rgba(245,158,11,0.06)',
          border: '1px solid rgba(245,158,11,0.2)',
          borderRadius: '7px',
          fontSize: '12px',
          color: 'var(--ifm-font-color-base)',
          opacity: 0.8,
        }}>
          <strong style={{ color: '#F59E0B' }}>PC2 composition :</strong> Fer & métaux (43,3 %) — seule famille dominant clairement PC2. Le reste est distribué entre Mg (18,4 %) et Ca II (12,1 %).
        </div>
      </div>
    </div>
  );
}
