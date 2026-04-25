import React from 'react';

const FAMILY_COLORS = {
  balmer:    '#60A5FA',
  caii:      '#C084FC',
  mg:        '#34D399',
  fe:        '#FB923C',
  lick:      '#FBBF24',
  continuum: '#F87171',
};

const PC1_LOADINGS = [
  { rank: 1, descriptor: 'W(Hα)',              w: +0.173, family: 'balmer',    familyLabel: 'Balmer' },
  { rank: 2, descriptor: 'Asymétrie continuum', w: +0.172, family: 'continuum', familyLabel: 'Continuum' },
  { rank: 3, descriptor: 'W(Mg b)',             w: -0.171, family: 'mg',        familyLabel: 'Magnésium' },
  { rank: 4, descriptor: 'B−V synthétique',    w: -0.163, family: 'continuum', familyLabel: 'Continuum' },
  { rank: 5, descriptor: 'fbleu / frouge',      w: +0.161, family: 'continuum', familyLabel: 'Continuum' },
];

const PC2_LOADINGS = [
  { rank: 1, descriptor: '[Fe/H]proxy',         w: +0.205, family: 'fe',   familyLabel: 'Fer & métaux' },
  { rank: 2, descriptor: 'Idx métal combiné',   w: +0.198, family: 'fe',   familyLabel: 'Fer & métaux' },
  { rank: 3, descriptor: 'Idx α',               w: +0.194, family: 'fe',   familyLabel: 'Fer & métaux' },
  { rank: 4, descriptor: 'Idx métal pauvre',    w: -0.193, family: 'fe',   familyLabel: 'Fer & métaux' },
  { rank: 5, descriptor: 'W(Fe λ5270)',          w: +0.174, family: 'fe',   familyLabel: 'Fer & métaux' },
];

const MAX_W = 0.22; // slightly above 0.205

function LoadingBar({ rank, descriptor, w, family, familyLabel, maxW }) {
  const [hov, setHov] = React.useState(false);
  const color = FAMILY_COLORS[family];
  const pct = (Math.abs(w) / maxW) * 100;
  const isPos = w > 0;

  return (
    <div
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'grid',
        gridTemplateColumns: '20px 140px 1fr 60px',
        alignItems: 'center',
        gap: '10px',
        padding: '8px 12px',
        background: hov ? `${color}0c` : 'transparent',
        borderRadius: '6px',
        transition: 'background 0.12s',
        cursor: 'default',
      }}
    >
      {/* Rank */}
      <div style={{
        fontFamily: 'monospace',
        fontSize: '11px',
        color: color,
        opacity: 0.6,
        textAlign: 'right',
      }}>{rank}</div>

      {/* Descriptor name */}
      <div style={{
        fontSize: '12.5px',
        fontFamily: 'monospace',
        color: 'var(--ifm-font-color-base)',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
      }}>{descriptor}</div>

      {/* Signed bar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        height: '16px',
        position: 'relative',
      }}>
        {/* Zero line */}
        <div style={{
          position: 'absolute',
          left: '50%',
          top: 0,
          bottom: 0,
          width: '1px',
          background: 'rgba(255,255,255,0.08)',
        }} />

        {/* Bar */}
        <div style={{
          position: 'absolute',
          left: isPos ? '50%' : `calc(50% - ${pct / 2}%)`,
          width: `${pct / 2}%`,
          height: '10px',
          top: '50%',
          transform: 'translateY(-50%)',
          background: hov ? color : `${color}cc`,
          borderRadius: isPos ? '0 3px 3px 0' : '3px 0 0 3px',
          transition: 'background 0.12s',
        }} />
      </div>

      {/* Value */}
      <div style={{
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: '13px',
        fontWeight: '700',
        color: color,
        textAlign: 'right',
      }}>
        {w > 0 ? '+' : ''}{w.toFixed(3)}
      </div>
    </div>
  );
}

function PCPanel({ title, sub, color, loadings, indicator }) {
  return (
    <div style={{
      background: 'var(--ifm-card-background-color)',
      border: `1px solid ${color}28`,
      borderTop: `3px solid ${color}`,
      borderRadius: '10px',
      overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        padding: '16px 18px 12px',
        borderBottom: '1px solid rgba(255,255,255,0.06)',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <div style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontWeight: '900',
              fontSize: '20px',
              color: color,
              lineHeight: 1,
              marginBottom: '4px',
            }}>{title}</div>
            <div style={{
              fontSize: '12px',
              color: 'var(--ifm-font-color-base)',
              opacity: 0.55,
            }}>{sub}</div>
          </div>
          <div style={{
            background: `${color}18`,
            border: `1px solid ${color}40`,
            borderRadius: '6px',
            padding: '6px 12px',
            textAlign: 'center',
          }}>
            <div style={{ fontSize: '10px', opacity: 0.6, color: 'var(--ifm-font-color-base)', marginBottom: '2px' }}>
              interprétation
            </div>
            <div style={{
              fontWeight: '700',
              fontSize: '12px',
              color: color,
            }}>{indicator}</div>
          </div>
        </div>
      </div>

      {/* Column labels */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '20px 140px 1fr 60px',
        gap: '10px',
        padding: '6px 12px',
        borderBottom: '1px solid rgba(255,255,255,0.04)',
      }}>
        <div style={{ fontSize: '9px', opacity: 0.35, textAlign: 'right', color: 'var(--ifm-font-color-base)' }}>#</div>
        <div style={{ fontSize: '9px', opacity: 0.35, color: 'var(--ifm-font-color-base)' }}>Descripteur</div>
        <div style={{ fontSize: '9px', opacity: 0.35, textAlign: 'center', color: 'var(--ifm-font-color-base)' }}>
          ← négatif · · · · positif →
        </div>
        <div style={{ fontSize: '9px', opacity: 0.35, textAlign: 'right', color: 'var(--ifm-font-color-base)' }}>w</div>
      </div>

      {/* Bars */}
      <div style={{ padding: '6px 0 10px' }}>
        {loadings.map((l) => (
          <LoadingBar key={l.rank} {...l} maxW={MAX_W} />
        ))}
      </div>
    </div>
  );
}

export default function LoadingsBars() {
  return (
    <div style={{ margin: '28px 0', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
      <PCPanel
        title="PC1"
        sub="axe thermique · 16,9 % de variance · ρ(Teff) = +0,831"
        color="#38BDF8"
        loadings={PC1_LOADINGS}
        indicator="Température T_eff"
      />
      <PCPanel
        title="PC2"
        sub="axe métallicité · 12,0 % de variance"
        color="#F59E0B"
        loadings={PC2_LOADINGS}
        indicator="Métallicité [Fe/H]"
      />
    </div>
  );
}
