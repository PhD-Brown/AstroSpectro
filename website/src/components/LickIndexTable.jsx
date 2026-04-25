import React from 'react';

const WL_MIN = 3690;
const WL_MAX = 9100;
function wlPct(lambda) {
  return ((lambda - WL_MIN) / (WL_MAX - WL_MIN)) * 100;
}
function parseRange(rangeStr) {
  const [a, b] = rangeStr.replace(/\s/g, '').split('–').map(Number);
  return [a, b];
}

const ORIGIN_STYLES = {
  'Lick':     { bg: '#F59E0B18', border: '#F59E0B44', color: '#F59E0B' },
  'SDSS':     { bg: '#38BDF818', border: '#38BDF844', color: '#38BDF8' },
  'standard': { bg: '#94A3B818', border: '#94A3B844', color: '#94A3B8' },
};

const INDICES = [
  { name: 'TiO 5',      range: '7126–7135', origin: 'Lick',     obs: 'Bandes TiO — étoiles M froides' },
  { name: 'Dn4000',     range: '4000–4100', origin: 'standard', obs: 'Saut 4000 Å — âge / métallicité' },
  { name: 'Dn4000 SDSS',range: '4050–4250', origin: 'SDSS',     obs: 'Version étendue du saut D4000' },
  { name: 'G4300',      range: '4280–4320', origin: 'Lick',     obs: 'Bande CH — sous-géantes G' },
  { name: 'Ca4227',     range: '4225–4235', origin: 'Lick',     obs: 'Ca I — métallicité' },
  { name: 'Hβ index',   range: '4840–4870', origin: 'Lick',     obs: 'Hβ — âge stellaire' },
  { name: 'Mgb index',  range: '5160–5190', origin: 'Lick',     obs: 'Mg b — [Mg/Fe]' },
  { name: 'Ca II triplet', range: '8480–8680', origin: 'standard', obs: 'Ca II IR — gravité / activité' },
  { name: 'CaH 2',      range: '6814–6846', origin: 'standard', obs: 'CaH — sous-types M' },
  { name: 'CaH 3',      range: '6960–6990', origin: 'standard', obs: 'CaH — sous-types M' },
  { name: 'CN 1',       range: '4142–4177', origin: 'Lick',     obs: 'CN — géantes / évolution stellaire' },
  { name: 'CN 2',       range: '4216–4251', origin: 'Lick',     obs: 'CN — géantes / évolution stellaire' },
  { name: 'Mg 1',       range: '5069–5135', origin: 'Lick',     obs: '[Mg/Fe]' },
  { name: 'Mg 2',       range: '5154–5197', origin: 'Lick',     obs: '[Mg/Fe]' },
  { name: 'Na D Lick',  range: '5876–5909', origin: 'Lick',     obs: 'Na D — gravité / âge' },
  { name: 'TiO 1 Lick', range: '5936–5994', origin: 'Lick',     obs: 'TiO — étoiles froides' },
  { name: 'TiO 2 Lick', range: '6190–6272', origin: 'Lick',     obs: 'TiO — étoiles froides' },
  { name: 'Paschen 12', range: '8730–8772', origin: 'standard', obs: 'Série de Paschen' },
];

function OriginChip({ origin }) {
  const s = ORIGIN_STYLES[origin] || ORIGIN_STYLES['standard'];
  return (
    <span style={{
      background: s.bg,
      border: `1px solid ${s.border}`,
      borderRadius: '4px',
      padding: '2px 8px',
      fontSize: '11px',
      fontWeight: '700',
      fontFamily: 'monospace',
      color: s.color,
      whiteSpace: 'nowrap',
    }}>{origin}</span>
  );
}

function RangeBar({ range, origin }) {
  const s = ORIGIN_STYLES[origin] || ORIGIN_STYLES['standard'];
  let lo, hi;
  try {
    [lo, hi] = parseRange(range);
  } catch {
    return null;
  }
  const leftPct  = wlPct(lo);
  const widthPct = wlPct(hi) - leftPct;

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
      {/* Range text */}
      <span style={{
        fontFamily: 'monospace',
        fontSize: '11.5px',
        color: s.color,
        whiteSpace: 'nowrap',
        minWidth: '90px',
        fontWeight: '600',
      }}>{range} Å</span>

      {/* Position strip */}
      <div style={{
        flex: 1,
        height: '6px',
        background: 'rgba(255,255,255,0.05)',
        borderRadius: '3px',
        position: 'relative',
        overflow: 'hidden',
        minWidth: '60px',
      }}>
        <div style={{
          position: 'absolute',
          left: `${leftPct}%`,
          width: `${Math.max(widthPct, 0.8)}%`,
          top: 0,
          bottom: 0,
          background: s.color,
          borderRadius: '2px',
          opacity: 0.75,
        }} />
      </div>
    </div>
  );
}

export default function LickIndexTable() {
  const [activeOrigin, setActiveOrigin] = React.useState(null);
  const [hovRow, setHovRow] = React.useState(null);

  const origins = ['Lick', 'SDSS', 'standard'];
  const counts = {};
  for (const o of origins) {
    counts[o] = INDICES.filter(i => i.origin === o).length;
  }

  const visible = activeOrigin
    ? INDICES.filter(i => i.origin === activeOrigin)
    : INDICES;

  return (
    <div style={{ margin: '20px 0' }}>
      {/* Filter chips */}
      <div style={{
        display: 'flex',
        gap: '8px',
        alignItems: 'center',
        marginBottom: '12px',
        flexWrap: 'wrap',
      }}>
        <span style={{ fontSize: '11px', opacity: 0.45, color: 'var(--ifm-font-color-base)' }}>
          Filtrer par origine :
        </span>
        <button
          onClick={() => setActiveOrigin(null)}
          style={{
            background: !activeOrigin ? 'rgba(255,255,255,0.1)' : 'transparent',
            border: `1px solid ${!activeOrigin ? 'rgba(255,255,255,0.3)' : 'rgba(255,255,255,0.1)'}`,
            borderRadius: '4px',
            padding: '3px 10px',
            fontSize: '11px',
            color: 'var(--ifm-font-color-base)',
            cursor: 'pointer',
            fontWeight: !activeOrigin ? '700' : '400',
          }}
        >Tous ({INDICES.length})</button>
        {origins.map((o) => {
          const s = ORIGIN_STYLES[o];
          const isActive = activeOrigin === o;
          return (
            <button
              key={o}
              onClick={() => setActiveOrigin(isActive ? null : o)}
              style={{
                background: isActive ? s.bg : 'transparent',
                border: `1px solid ${isActive ? s.border : s.border}`,
                borderRadius: '4px',
                padding: '3px 10px',
                fontSize: '11px',
                color: s.color,
                cursor: 'pointer',
                fontWeight: isActive ? '700' : '400',
                fontFamily: 'monospace',
              }}
            >{o} ({counts[o]})</button>
          );
        })}

        {/* Range bar legend */}
        <div style={{
          marginLeft: 'auto',
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          fontSize: '10px',
          opacity: 0.35,
          color: 'var(--ifm-font-color-base)',
        }}>
          <div style={{ width: '40px', height: '4px', background: 'rgba(255,255,255,0.3)', borderRadius: '2px', position: 'relative' }}>
            <div style={{ position: 'absolute', left: '30%', width: '20%', height: '100%', background: '#94A3B8', borderRadius: '2px' }} />
          </div>
          position dans 3690–9100 Å
        </div>
      </div>

      {/* Table */}
      <div style={{
        border: '1px solid rgba(255,255,255,0.07)',
        borderRadius: '10px',
        overflow: 'hidden',
      }}>
        {/* Header */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '130px 200px 90px 1fr',
          gap: '0',
          background: 'rgba(255,255,255,0.03)',
          borderBottom: '1px solid rgba(255,255,255,0.07)',
          padding: '8px 14px',
        }}>
          {['Indice', 'Fenêtre λ', 'Origine', 'Observable'].map((h) => (
            <div key={h} style={{
              fontSize: '10px',
              fontWeight: '700',
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
              color: 'var(--ifm-font-color-base)',
              opacity: 0.4,
            }}>{h}</div>
          ))}
        </div>

        {/* Rows */}
        {visible.map((row, ri) => {
          const s = ORIGIN_STYLES[row.origin];
          const isHov = hovRow === ri;
          return (
            <div
              key={row.name}
              onMouseEnter={() => setHovRow(ri)}
              onMouseLeave={() => setHovRow(null)}
              style={{
                display: 'grid',
                gridTemplateColumns: '130px 200px 90px 1fr',
                gap: '0',
                padding: '8px 14px',
                borderBottom: ri < visible.length - 1
                  ? '1px solid rgba(255,255,255,0.04)' : 'none',
                background: isHov
                  ? `${s.color}09`
                  : ri % 2 === 0 ? 'rgba(255,255,255,0.012)' : 'transparent',
                transition: 'background 0.1s',
                alignItems: 'center',
              }}
            >
              {/* Index name */}
              <div style={{
                fontFamily: 'monospace',
                fontSize: '12.5px',
                fontWeight: '600',
                color: isHov ? s.color : 'var(--ifm-font-color-base)',
                transition: 'color 0.1s',
              }}>{row.name}</div>

              {/* Range + bar */}
              <div style={{ paddingRight: '12px' }}>
                <RangeBar range={row.range} origin={row.origin} />
              </div>

              {/* Origin chip */}
              <div>
                <OriginChip origin={row.origin} />
              </div>

              {/* Observable */}
              <div style={{
                fontSize: '12px',
                color: 'var(--ifm-font-color-base)',
                opacity: isHov ? 0.85 : 0.6,
                transition: 'opacity 0.1s',
              }}>{row.obs}</div>
            </div>
          );
        })}
      </div>

      {activeOrigin && (
        <div style={{
          marginTop: '8px',
          fontSize: '11px',
          opacity: 0.4,
          color: 'var(--ifm-font-color-base)',
          textAlign: 'center',
        }}>
          {visible.length} indices · <button
            onClick={() => setActiveOrigin(null)}
            style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', textDecoration: 'underline', fontSize: 'inherit', opacity: 'inherit', padding: 0 }}
          >voir tous</button>
        </div>
      )}
    </div>
  );
}
