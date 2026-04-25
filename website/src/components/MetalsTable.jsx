import React from 'react';

const WL_MIN = 3690;
const WL_MAX = 9100;
function wlPct(lambda) {
  return ((lambda - WL_MIN) / (WL_MAX - WL_MIN)) * 100;
}

const ELEMENTS = [
  {
    symbol: 'Fe',
    name: 'Fer',
    color: '#FB923C',
    roman: 'Fe I / Fe II',
    lines: [
      { lambda: 4383.5, species: 'Fe I',  force: 'medium' },
      { lambda: 4531.1, species: 'Fe II', force: 'weak'   },
      { lambda: 5270.4, species: 'Fe I',  force: 'medium' },
      { lambda: 5335.2, species: 'Fe I',  force: 'medium' },
      { lambda: 5406.8, species: 'Fe I',  force: 'weak'   },
      { lambda: 5709.4, species: 'Fe I',  force: 'weak'   },
    ],
    pcaNote: 'PC2 loading +0,174',
  },
  {
    symbol: 'Cr',
    name: 'Chrome',
    color: '#FBBF24',
    roman: 'Cr I',
    lines: [
      { lambda: 5206.0, species: 'Cr I', force: 'weak' },
      { lambda: 5208.4, species: 'Cr I', force: 'weak' },
    ],
    pcaNote: null,
  },
  {
    symbol: 'Ni',
    name: 'Nickel',
    color: '#34D399',
    roman: 'Ni I',
    lines: [
      { lambda: 5081.1, species: 'Ni I', force: 'weak' },
    ],
    pcaNote: null,
  },
  {
    symbol: 'Na',
    name: 'Sodium',
    color: '#F87171',
    roman: 'Na I',
    lines: [
      { lambda: 5889.9, species: 'Na I', force: 'medium' },
      { lambda: 5895.9, species: 'Na I', force: 'medium' },
    ],
    pcaNote: 'Doublet D — ratio W(MgB)/W(NaD)',
  },
  {
    symbol: 'Ba',
    name: 'Baryum',
    color: '#60A5FA',
    roman: 'Ba II',
    lines: [
      { lambda: 4554.0, species: 'Ba II', force: 'medium' },
      { lambda: 6496.9, species: 'Ba II', force: 'medium' },
    ],
    pcaNote: null,
  },
  {
    symbol: 'Sr',
    name: 'Strontium',
    color: '#818CF8',
    roman: 'Sr II',
    lines: [
      { lambda: 4077.7, species: 'Sr II', force: 'medium' },
    ],
    pcaNote: null,
  },
  {
    symbol: 'Si',
    name: 'Silicium',
    color: '#C084FC',
    roman: 'Si II',
    lines: [
      { lambda: 4128.1, species: 'Si II', force: 'medium' },
      { lambda: 4130.9, species: 'Si II', force: 'medium' },
    ],
    pcaNote: null,
  },
  {
    symbol: 'Ti',
    name: 'Titane',
    color: '#E879F9',
    roman: 'Ti II',
    lines: [
      { lambda: 4758.1, species: 'Ti II', force: 'weak' },
      { lambda: 4764.9, species: 'Ti II', force: 'weak' },
    ],
    pcaNote: 'proxy α-éléments',
  },
  {
    symbol: 'Al',
    name: 'Aluminium',
    color: '#94A3B8',
    roman: 'Al I',
    lines: [
      { lambda: 3944.0, species: 'Al I', force: 'weak' },
    ],
    pcaNote: null,
  },
  {
    symbol: 'Co',
    name: 'Cobalt',
    color: '#6EE7B7',
    roman: 'Co I',
    lines: [
      { lambda: 5301.0, species: 'Co I', force: 'weak' },
    ],
    pcaNote: null,
  },
  {
    symbol: 'V',
    name: 'Vanadium',
    color: '#FCA5A5',
    roman: 'V II',
    lines: [
      { lambda: 4379.2, species: 'V II', force: 'weak' },
    ],
    pcaNote: null,
  },
];

const FORCE_DOTS = {
  strong: 3,
  medium: 2,
  weak:   1,
};

function ForceDots({ force, color }) {
  const count = FORCE_DOTS[force] ?? 1;
  return (
    <div style={{ display: 'flex', gap: '2px', alignItems: 'center' }}>
      {[1,2,3].map((i) => (
        <div key={i} style={{
          width: '6px',
          height: '6px',
          borderRadius: '50%',
          background: i <= count ? color : 'rgba(255,255,255,0.08)',
          transition: 'background 0.1s',
        }} />
      ))}
    </div>
  );
}

function LambdaDot({ lambda, color }) {
  const pct = wlPct(lambda);
  return (
    <div style={{
      position: 'relative',
      width: '100%',
      height: '4px',
      background: 'rgba(255,255,255,0.06)',
      borderRadius: '2px',
      overflow: 'visible',
    }}>
      <div style={{
        position: 'absolute',
        left: `${pct}%`,
        top: '50%',
        width: '6px',
        height: '6px',
        borderRadius: '50%',
        background: color,
        transform: 'translate(-50%, -50%)',
        boxShadow: `0 0 0 2px ${color}33`,
      }} />
    </div>
  );
}

function ElementCard({ symbol, name, color, roman, lines, pcaNote }) {
  const [hov, setHov] = React.useState(false);
  const isLarge = lines.length >= 4;

  return (
    <div
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        background: hov ? `${color}0e` : 'var(--ifm-card-background-color)',
        border: `1px solid ${hov ? color + '55' : color + '25'}`,
        borderTop: `3px solid ${color}`,
        borderRadius: '8px',
        padding: '12px 14px',
        transition: 'all 0.15s',
        gridColumn: isLarge ? 'span 2' : 'span 1',
      }}
    >
      {/* Card header */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        marginBottom: '10px',
      }}>
        <div>
          <div style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontWeight: '900',
            fontSize: '20px',
            color: color,
            lineHeight: 1,
            marginBottom: '2px',
          }}>{symbol}</div>
          <div style={{
            fontSize: '10px',
            color: 'var(--ifm-font-color-base)',
            opacity: 0.45,
          }}>{name} · {roman}</div>
        </div>
        <div style={{
          background: `${color}15`,
          border: `1px solid ${color}30`,
          borderRadius: '12px',
          padding: '2px 8px',
          fontSize: '11px',
          fontFamily: 'monospace',
          fontWeight: '700',
          color: color,
        }}>{lines.length} {lines.length === 1 ? 'raie' : 'raies'}</div>
      </div>

      {/* Lines */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
        {lines.map((l, i) => (
          <div key={i} style={{
            display: 'grid',
            gridTemplateColumns: '72px 50px 1fr 46px',
            alignItems: 'center',
            gap: '8px',
            padding: '4px 6px',
            borderRadius: '4px',
            background: 'rgba(255,255,255,0.02)',
          }}>
            {/* Lambda */}
            <div style={{
              fontFamily: 'monospace',
              fontSize: '12px',
              fontWeight: '600',
              color: color,
            }}>{l.lambda.toFixed(1)} Å</div>

            {/* Species chip */}
            <div style={{
              background: `${color}18`,
              borderRadius: '3px',
              padding: '1px 5px',
              fontSize: '10px',
              fontFamily: 'monospace',
              fontWeight: '700',
              color: color,
              textAlign: 'center',
            }}>{l.species}</div>

            {/* Lambda position strip */}
            <LambdaDot lambda={l.lambda} color={color} />

            {/* Force */}
            <ForceDots force={l.force} color={color} />
          </div>
        ))}
      </div>

      {/* PCA note */}
      {pcaNote && (
        <div style={{
          marginTop: '8px',
          fontSize: '10px',
          color: color,
          opacity: 0.7,
          fontStyle: 'italic',
          borderTop: `1px solid ${color}20`,
          paddingTop: '6px',
        }}>→ {pcaNote}</div>
      )}
    </div>
  );
}

export default function MetalsTable() {
  return (
    <div style={{ margin: '20px 0' }}>
      {/* Legend */}
      <div style={{
        display: 'flex',
        gap: '16px',
        alignItems: 'center',
        marginBottom: '14px',
        flexWrap: 'wrap',
        fontSize: '11px',
        color: 'var(--ifm-font-color-base)',
        opacity: 0.55,
      }}>
        <span>Force :</span>
        {[
          { label: 'strong', n: 3 },
          { label: 'medium', n: 2 },
          { label: 'weak',   n: 1 },
        ].map((f) => (
          <div key={f.label} style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <div style={{ display: 'flex', gap: '2px' }}>
              {[1,2,3].map((i) => (
                <div key={i} style={{
                  width: '6px', height: '6px', borderRadius: '50%',
                  background: i <= f.n ? '#94A3B8' : 'rgba(255,255,255,0.08)',
                }} />
              ))}
            </div>
            <span>{f.label}</span>
          </div>
        ))}
        <span style={{ marginLeft: '8px' }}>· La barre montre la position λ dans 3690–9100 Å</span>
      </div>

      {/* Element grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: '10px',
      }}>
        {ELEMENTS.map((el) => (
          <ElementCard key={el.symbol} {...el} />
        ))}
      </div>

      {/* Footer */}
      <div style={{
        marginTop: '12px',
        padding: '10px 14px',
        background: 'rgba(251,146,60,0.06)',
        border: '1px solid rgba(251,146,60,0.2)',
        borderRadius: '7px',
        fontSize: '12px',
        color: 'var(--ifm-font-color-base)',
        opacity: 0.8,
      }}>
        <strong style={{ color: '#FB923C' }}>Pour chaque raie :</strong> prominence, FWHM et W(raie) via{' '}
        <code>specutils.analysis.equivalent_width</code>. Composites :{' '}
        <code>feature_metal_index_combined</code>,{' '}
        <code>feature_iron_peak_index</code>,{' '}
        <code>feature_alpha_elements_index</code>.
      </div>
    </div>
  );
}
