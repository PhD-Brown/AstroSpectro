import React from 'react';

const CONFIG = [
  {
    key: 'Jeu de données',
    value: 'LAMOST DR5 × Gaia DR3',
    sub: 'N = 43 019 spectres',
    color: '#38BDF8',
    icon: '◎',
  },
  {
    key: 'Descripteurs',
    value: '183',
    sub: 'mode spectro_only=True',
    color: '#34D399',
    icon: 'p',
    mono: true,
  },
  {
    key: 'Classes cibles',
    value: 'A · F · G · K · M',
    sub: '5 types spectraux stellaires',
    color: '#FBBF24',
    icon: '★',
  },
  {
    key: 'Métrique',
    value: 'balanced_accuracy',
    sub: 'optimisation & évaluation',
    color: '#A78BFA',
    icon: '≡',
    mono: true,
  },
  {
    key: 'Split train/test',
    value: '80 / 20',
    sub: 'stratifié par classe',
    color: '#FB923C',
    icon: '⊘',
  },
];

const HYPERPARAMS = [
  { name: 'n_estimators', value: '600',   color: '#38BDF8' },
  { name: 'max_depth',    value: '6',     color: '#34D399' },
  { name: 'learning_rate',value: '0.1',   color: '#FBBF24' },
  { name: 'colsample_bytree', value: '0.8', color: '#A78BFA' },
  { name: 'subsample',    value: '0.8',   color: '#FB923C' },
  { name: 'tree_method',  value: '"hist"',color: '#F87171' },
];

function ConfigRow({ k, value, sub, color, icon, mono }) {
  const [hov, setHov] = React.useState(false);
  return (
    <div
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'grid',
        gridTemplateColumns: '36px 160px 1fr',
        alignItems: 'center',
        gap: '12px',
        padding: '10px 14px',
        borderBottom: '1px solid rgba(255,255,255,0.04)',
        background: hov ? `${color}09` : 'transparent',
        transition: 'background 0.12s',
        cursor: 'default',
      }}
    >
      {/* Icon */}
      <div style={{
        width: '28px',
        height: '28px',
        borderRadius: '6px',
        background: `${color}18`,
        border: `1px solid ${color}30`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontFamily: 'monospace',
        fontSize: '13px',
        fontWeight: '700',
        color: color,
        flexShrink: 0,
      }}>{icon}</div>

      {/* Key */}
      <div style={{
        fontSize: '12px',
        color: 'var(--ifm-font-color-base)',
        opacity: 0.6,
        fontWeight: '500',
      }}>{k}</div>

      {/* Value */}
      <div>
        <div style={{
          fontFamily: mono ? "'JetBrains Mono', monospace" : 'inherit',
          fontSize: mono ? '13px' : '14px',
          fontWeight: '700',
          color: hov ? color : 'var(--ifm-font-color-base)',
          transition: 'color 0.12s',
        }}>{value}</div>
        {sub && (
          <div style={{ fontSize: '10.5px', opacity: 0.45, color: 'var(--ifm-font-color-base)', marginTop: '1px' }}>
            {sub}
          </div>
        )}
      </div>
    </div>
  );
}

function HyperParam({ name, value, color }) {
  const [hov, setHov] = React.useState(false);
  return (
    <div
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '4px',
        padding: '10px 12px',
        background: hov ? `${color}10` : `${color}06`,
        border: `1px solid ${hov ? color + '44' : color + '20'}`,
        borderRadius: '7px',
        transition: 'all 0.12s',
        cursor: 'default',
      }}
    >
      <div style={{
        fontFamily: 'monospace',
        fontSize: '10px',
        color: 'var(--ifm-font-color-base)',
        opacity: 0.5,
      }}>{name}</div>
      <div style={{
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: '16px',
        fontWeight: '800',
        color: color,
        lineHeight: 1,
      }}>{value}</div>
    </div>
  );
}

export default function TrainingConfig() {
  return (
    <div style={{ margin: '24px 0', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>

      {/* Left: config */}
      <div style={{
        background: 'var(--ifm-card-background-color)',
        border: '1px solid rgba(255,255,255,0.07)',
        borderRadius: '10px',
        overflow: 'hidden',
      }}>
        <div style={{
          padding: '10px 14px',
          background: 'rgba(255,255,255,0.03)',
          borderBottom: '1px solid rgba(255,255,255,0.06)',
          fontSize: '11px',
          fontWeight: '700',
          textTransform: 'uppercase',
          letterSpacing: '0.07em',
          opacity: 0.4,
          color: 'var(--ifm-font-color-base)',
        }}>Configuration expérimentale</div>

        {CONFIG.map((c) => <ConfigRow key={c.key} k={c.key} {...c} />)}
      </div>

      {/* Right: hyperparams */}
      <div style={{
        background: 'var(--ifm-card-background-color)',
        border: '1px solid rgba(255,255,255,0.07)',
        borderRadius: '10px',
        overflow: 'hidden',
      }}>
        <div style={{
          padding: '10px 14px',
          background: 'rgba(255,255,255,0.03)',
          borderBottom: '1px solid rgba(255,255,255,0.06)',
          fontSize: '11px',
          fontWeight: '700',
          textTransform: 'uppercase',
          letterSpacing: '0.07em',
          opacity: 0.4,
          color: 'var(--ifm-font-color-base)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}>
          <span>Hyperparamètres XGBoost</span>
          <span style={{ fontSize: '9px', opacity: 0.6, textTransform: 'none', letterSpacing: 0, fontWeight: '400', fontFamily: 'monospace' }}>
            run 20260213T225019Z
          </span>
        </div>

        <div style={{
          padding: '14px',
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '8px',
        }}>
          {HYPERPARAMS.map((p) => <HyperParam key={p.name} {...p} />)}
        </div>

        <div style={{
          padding: '10px 14px',
          borderTop: '1px solid rgba(255,255,255,0.05)',
          fontSize: '11px',
          opacity: 0.5,
          color: 'var(--ifm-font-color-base)',
          fontFamily: 'monospace',
        }}>
          Tuning via GridSearchCV · tree_method hist = GPU/CPU
        </div>
      </div>
    </div>
  );
}
