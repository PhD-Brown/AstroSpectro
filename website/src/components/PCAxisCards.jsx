import React from 'react';

const PC1_FAMILIES = [
  { name: 'Balmer (H)',   pct: 31.5, color: '#60A5FA' },
  { name: 'Ca II',        pct: 22.4, color: '#C084FC' },
  { name: 'Fer & métaux', pct: 15.3, color: '#FB923C' },
  { name: 'Mg / α',      pct: 10.8, color: '#34D399' },
  { name: 'Lick/SDSS',   pct:  9.7, color: '#FBBF24' },
  { name: 'Continuum',   pct: 10.3, color: '#F87171' },
];

const PC2_FAMILIES = [
  { name: 'Balmer (H)',   pct:  8.2, color: '#60A5FA' },
  { name: 'Ca II',        pct: 12.1, color: '#C084FC' },
  { name: 'Fer & métaux', pct: 43.3, color: '#FB923C' },
  { name: 'Mg / α',      pct: 18.4, color: '#34D399' },
  { name: 'Lick/SDSS',   pct: 11.6, color: '#FBBF24' },
  { name: 'Continuum',   pct:  6.4, color: '#F87171' },
];

function FamilyBar({ families }) {
  return (
    <div>
      {/* Stacked bar */}
      <div style={{ display: 'flex', height: '8px', borderRadius: '4px', overflow: 'hidden', marginBottom: '6px' }}>
        {families.map((f) => (
          <div key={f.name} title={`${f.name}: ${f.pct}%`} style={{
            flex: `0 0 ${f.pct}%`,
            background: f.color,
          }} />
        ))}
      </div>
      {/* Legend */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
        {families.map((f) => (
          <div key={f.name} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <div style={{ width: '8px', height: '8px', borderRadius: '2px', background: f.color, flexShrink: 0 }} />
            <span style={{ fontSize: '10px', opacity: 0.6, color: 'var(--ifm-font-color-base)', whiteSpace: 'nowrap' }}>
              {f.name} {f.pct}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function RhoDisplay({ value, label, color, isBig }) {
  const abs = Math.abs(value);
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: isBig ? '36px' : '22px',
        fontWeight: '900',
        color: color,
        lineHeight: 1,
        letterSpacing: '-1px',
      }}>
        {value > 0 ? '+' : ''}{value.toFixed(3)}
      </div>
      <div style={{ fontSize: '10px', opacity: 0.5, marginTop: '3px', color: 'var(--ifm-font-color-base)' }}>
        ρ({label})
      </div>
      {/* Strength bar */}
      <div style={{
        height: '3px',
        background: 'rgba(255,255,255,0.08)',
        borderRadius: '2px',
        marginTop: '5px',
        overflow: 'hidden',
      }}>
        <div style={{
          width: `${abs * 100}%`,
          height: '100%',
          background: color,
          borderRadius: '2px',
        }} />
      </div>
    </div>
  );
}

function PCCard({ pcLabel, axiName, axiSub, color, variancePct, lambda, topRho, topRhoLabel, secondRho, secondRhoLabel, families, topLoading, topLoadingW, warning }) {
  const [hov, setHov] = React.useState(false);
  return (
    <div
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        flex: 1,
        background: 'var(--ifm-card-background-color)',
        border: `1px solid ${hov ? color + '66' : color + '30'}`,
        borderTop: `4px solid ${color}`,
        borderRadius: '12px',
        padding: '22px',
        transition: 'border-color 0.15s, box-shadow 0.15s',
        boxShadow: hov ? `0 8px 28px ${color}15` : '0 2px 8px rgba(0,0,0,0.08)',
        minWidth: 0,
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '18px' }}>
        <div>
          <div style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontWeight: '900',
            fontSize: '28px',
            color: color,
            lineHeight: 1,
            letterSpacing: '-1px',
          }}>{pcLabel}</div>
          <div style={{
            fontWeight: '800',
            fontSize: '15px',
            color: 'var(--ifm-font-color-base)',
            marginTop: '4px',
          }}>{axiName}</div>
          <div style={{ fontSize: '12px', opacity: 0.55, marginTop: '2px', color: 'var(--ifm-font-color-base)' }}>
            {axiSub}
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{
            fontFamily: 'monospace',
            fontSize: '22px',
            fontWeight: '800',
            color: color,
          }}>{variancePct}%</div>
          <div style={{ fontSize: '10px', opacity: 0.45, color: 'var(--ifm-font-color-base)' }}>variance · λ={lambda}</div>
        </div>
      </div>

      {/* Correlations row */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '14px',
        padding: '14px',
        background: `${color}08`,
        border: `1px solid ${color}20`,
        borderRadius: '8px',
        marginBottom: '16px',
      }}>
        <RhoDisplay value={topRho} label={topRhoLabel} color={color} isBig />
        <RhoDisplay value={secondRho} label={secondRhoLabel} color={color} isBig={false} />
      </div>

      {/* Top loading */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '8px 12px',
        background: 'rgba(255,255,255,0.03)',
        border: '1px solid rgba(255,255,255,0.06)',
        borderRadius: '6px',
        marginBottom: '14px',
      }}>
        <div>
          <div style={{ fontSize: '10px', opacity: 0.4, marginBottom: '2px', color: 'var(--ifm-font-color-base)' }}>
            Top loading absolu
          </div>
          <div style={{
            fontFamily: 'monospace',
            fontSize: '13px',
            fontWeight: '600',
            color: 'var(--ifm-font-color-base)',
          }}>{topLoading}</div>
        </div>
        <div style={{
          fontFamily: 'monospace',
          fontWeight: '800',
          fontSize: '16px',
          color: color,
        }}>
          {topLoadingW > 0 ? '+' : ''}{topLoadingW.toFixed(3)}
        </div>
      </div>

      {/* Family contributions */}
      <div style={{ marginBottom: warning ? '14px' : '0' }}>
        <div style={{ fontSize: '10px', opacity: 0.4, marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--ifm-font-color-base)' }}>
          Contribution par famille
        </div>
        <FamilyBar families={families} />
      </div>

      {/* Warning */}
      {warning && (
        <div style={{
          padding: '8px 12px',
          background: 'rgba(245,158,11,0.08)',
          border: '1px solid rgba(245,158,11,0.25)',
          borderRadius: '6px',
          fontSize: '11px',
          color: 'var(--ifm-font-color-base)',
          lineHeight: 1.5,
        }}>
          ⚠️ {warning}
        </div>
      )}
    </div>
  );
}

export default function PCAxisCards() {
  return (
    <div style={{ display: 'flex', gap: '16px', margin: '28px 0', flexWrap: 'wrap' }}>
      <PCCard
        pcLabel="PC1"
        axiName="Axe Thermique"
        axiSub="Temperature effective dominante"
        color="#38BDF8"
        variancePct={16.9}
        lambda="30,19"
        topRho={+0.831}
        topRhoLabel="Teff"
        secondRho={-0.768}
        secondRhoLabel="GBP−GRP"
        families={PC1_FAMILIES}
        topLoading="W(Hα)"
        topLoadingW={+0.173}
        warning={null}
      />
      <PCCard
        pcLabel="PC2"
        axiName="Axe Métallicité"
        axiSub="Composition chimique + contamination SNR"
        color="#F59E0B"
        variancePct={12.0}
        lambda="21,39"
        topRho={-0.548}
        topRhoLabel="[Fe/H]"
        secondRho={-0.310}
        secondRhoLabel="SNR_r"
        families={PC2_FAMILIES}
        topLoading="[Fe/H]proxy"
        topLoadingW={+0.205}
        warning="ρ(SNRr) = −0,310 : PC2 contient une part de variance instrumentale — ce n'est pas un axe de métallicité pur."
      />
    </div>
  );
}
