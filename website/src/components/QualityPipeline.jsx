import React from 'react';

const PIPELINE = [
  {
    step: 1,
    name: 'Rapport signal / bruit',
    criterion: 'SNR_r > 10',
    desc: 'Seuls les spectres avec un SNR en bande rouge supérieur à 10 sont conservés. Ce seuil garantit que les raies spectrales sont détectables et que les mesures de largeur équivalente sont fiables.',
    source: 'LAMOST DR5',
    color: '#38BDF8',
    icon: '〜',
  },
  {
    step: 2,
    name: 'Qualité astrométrique',
    criterion: 'RUWE < 1.4',
    desc: 'Le Renormalised Unit Weight Error (RUWE) de Gaia mesure la qualité de l\'ajustement astrométrique. Une valeur < 1.4 garantit une solution de mouvement propre simple, excluant les étoiles doubles non résolues.',
    source: 'Gaia DR3',
    color: '#34D399',
    icon: '◎',
  },
];

function Arrow() {
  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '28px',
        color: 'var(--ifm-font-color-base)',
        opacity: 0.25,
        fontSize: '20px',
        userSelect: 'none',
      }}
    >
      ↓
    </div>
  );
}

function FilterStep({ step, name, criterion, desc, source, color, icon }) {
  const [hovered, setHovered] = React.useState(false);

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: 'flex',
        gap: '0',
        background: 'var(--ifm-card-background-color)',
        border: `1px solid ${hovered ? color + '55' : color + '25'}`,
        borderRadius: '10px',
        overflow: 'hidden',
        transition: 'border-color 0.15s, box-shadow 0.15s',
        boxShadow: hovered ? `0 6px 20px ${color}14` : 'none',
      }}
    >
      {/* Left badge column */}
      <div
        style={{
          background: `${color}14`,
          borderRight: `2px solid ${color}33`,
          padding: '20px 16px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '8px',
          minWidth: '72px',
          flexShrink: 0,
        }}
      >
        {/* Step circle */}
        <div
          style={{
            width: '36px',
            height: '36px',
            borderRadius: '50%',
            background: color,
            color: '#000',
            fontWeight: '900',
            fontSize: '17px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: `0 0 0 4px ${color}22`,
          }}
        >
          {step}
        </div>

        {/* Source pill */}
        <div
          style={{
            background: `${color}22`,
            border: `1px solid ${color}44`,
            borderRadius: '4px',
            padding: '2px 6px',
            fontSize: '9.5px',
            fontFamily: 'monospace',
            fontWeight: '700',
            color: color,
            textAlign: 'center',
            whiteSpace: 'nowrap',
            letterSpacing: '0.04em',
          }}
        >
          {source}
        </div>
      </div>

      {/* Content */}
      <div style={{ padding: '18px 22px', flex: 1 }}>
        <div
          style={{
            fontWeight: '700',
            fontSize: '14px',
            color: 'var(--ifm-font-color-base)',
            marginBottom: '10px',
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            fontSize: '12px',
            opacity: 0.8,
          }}
        >
          {name}
        </div>

        {/* Criterion chip */}
        <div
          style={{
            fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
            fontSize: '18px',
            fontWeight: '800',
            color: color,
            background: `${color}14`,
            border: `1px solid ${color}38`,
            borderRadius: '7px',
            padding: '8px 16px',
            display: 'inline-block',
            marginBottom: '12px',
            letterSpacing: '0.02em',
          }}
        >
          {criterion}
        </div>

        <div
          style={{
            fontSize: '13px',
            color: 'var(--ifm-font-color-base)',
            opacity: 0.65,
            lineHeight: 1.65,
          }}
        >
          {desc}
        </div>
      </div>
    </div>
  );
}

export default function QualityPipeline() {
  return (
    <div style={{ margin: '24px 0' }}>

      {/* Flow header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          flexWrap: 'wrap',
          padding: '12px 18px',
          background: 'rgba(255,255,255,0.025)',
          border: '1px solid rgba(255,255,255,0.07)',
          borderRadius: '8px',
          marginBottom: '20px',
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: '13px',
        }}
      >
        <span style={{ color: '#94A3B8' }}>Entrée :</span>
        <span style={{ color: '#F59E0B', fontWeight: '700' }}>~250k</span>
        <span style={{ color: '#94A3B8', opacity: 0.5 }}>candidats LAMOST DR5</span>
        <span style={{ margin: '0 6px', color: '#475569' }}>→→</span>
        <span style={{ color: '#94A3B8' }}>SNR</span>
        <span style={{ margin: '0 6px', color: '#475569' }}>→</span>
        <span style={{ color: '#94A3B8' }}>RUWE</span>
        <span style={{ margin: '0 6px', color: '#475569' }}>→→</span>
        <span style={{ color: '#34D399', fontWeight: '700' }}>43 019</span>
        <span style={{ color: '#94A3B8', opacity: 0.5 }}>retenus</span>
      </div>

      {/* Filter steps */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
        {PIPELINE.map((f, i) => (
          <React.Fragment key={f.step}>
            <FilterStep {...f} />
            {i < PIPELINE.length - 1 && <Arrow />}
          </React.Fragment>
        ))}
      </div>

      {/* Result footer */}
      <div
        style={{
          marginTop: '16px',
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          padding: '14px 18px',
          background: 'rgba(52, 211, 153, 0.06)',
          border: '1px solid rgba(52, 211, 153, 0.2)',
          borderRadius: '8px',
        }}
      >
        <div
          style={{
            width: '10px',
            height: '10px',
            borderRadius: '50%',
            background: '#34D399',
            boxShadow: '0 0 0 3px rgba(52,211,153,0.25)',
            flexShrink: 0,
          }}
        />
        <div style={{ fontSize: '13px', color: 'var(--ifm-font-color-base)', opacity: 0.8 }}>
          Jeu de données final :{' '}
          <strong style={{ color: '#34D399', fontFamily: 'monospace' }}>N = 43 019</strong>
          {' '}spectres · <strong style={{ fontFamily: 'monospace' }}>p = 183</strong> descripteurs ·{' '}
          matrice <strong style={{ fontFamily: 'monospace' }}>X ∈ ℝ^(43019 × 183)</strong>
        </div>
      </div>
    </div>
  );
}
