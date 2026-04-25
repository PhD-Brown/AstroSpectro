import React from 'react';

const UMAP_PARAMS = [
  { value: 5,   label: '5',   behavior: 'Fragmentation en amas très compacts, structure globale perdue', quality: 'bad' },
  { value: 15,  label: '15',  behavior: 'Meilleur compromis local/global — valeur retenue',               quality: 'best' },
  { value: 30,  label: '30',  behavior: 'Structure robuste, légère coalescence',                          quality: 'good' },
  { value: 50,  label: '50',  behavior: 'Structure qualitative préservée',                                quality: 'good' },
  { value: 100, label: '100', behavior: 'Coalescence des clusters, perte de détail',                      quality: 'bad' },
];

const TSNE_PARAMS = [
  { value: 5,   label: '5',   behavior: 'Micro-amas instables, structure globale perdue',                 quality: 'bad' },
  { value: 15,  label: '15',  behavior: 'Transition — populations majeures visibles',                    quality: 'ok' },
  { value: 30,  label: '30',  behavior: 'Meilleur compromis — valeur retenue',                            quality: 'best' },
  { value: 50,  label: '50',  behavior: 'Structure qualitative préservée',                                quality: 'good' },
  { value: 100, label: '100', behavior: 'Coalescence — t-SNE traite tout comme une seule population',     quality: 'bad' },
];

const QUALITY_STYLES = {
  best:  { color: '#34D399', label: '✓ Retenu',    bg: '#34D39914', border: '#34D39940' },
  good:  { color: '#FBBF24', label: '◎ Robuste',   bg: '#FBBF2414', border: '#FBBF2440' },
  ok:    { color: '#94A3B8', label: '△ Transition', bg: '#94A3B814', border: '#94A3B840' },
  bad:   { color: '#F87171', label: '✗ Éviter',    bg: '#F8717114', border: '#F8717140' },
};

// Spectrum bar — shows where the value sits between "too local" and "too global"
function SpectrumBar({ value, min, max, color }) {
  const pct = ((Math.log(value) - Math.log(min)) / (Math.log(max) - Math.log(min))) * 100;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
      <span style={{ fontSize: '9px', opacity: 0.4, whiteSpace: 'nowrap', width: '50px' }}>trop local</span>
      <div style={{
        flex: 1,
        height: '5px',
        background: 'linear-gradient(90deg, rgba(56,189,248,0.3), rgba(167,139,250,0.3))',
        borderRadius: '3px',
        position: 'relative',
      }}>
        <div style={{
          position: 'absolute',
          left: `${pct}%`,
          top: '-3px',
          width: '11px',
          height: '11px',
          borderRadius: '50%',
          background: color,
          transform: 'translateX(-50%)',
          boxShadow: `0 0 0 3px ${color}33`,
        }} />
      </div>
      <span style={{ fontSize: '9px', opacity: 0.4, whiteSpace: 'nowrap', width: '52px', textAlign: 'right' }}>trop global</span>
    </div>
  );
}

function ParamTable({ params, paramName, color, unit, min, max }) {
  return (
    <div>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '12px',
      }}>
        <div style={{ fontSize: '12px', opacity: 0.5, color: 'var(--ifm-font-color-base)' }}>
          Structure robuste pour {paramName} ∈ [15, 50]
        </div>
      </div>

      {/* Spectrum visual */}
      <div style={{
        background: 'rgba(255,255,255,0.02)',
        border: '1px solid rgba(255,255,255,0.06)',
        borderRadius: '7px',
        padding: '12px 16px',
        marginBottom: '14px',
      }}>
        <div style={{ fontSize: '11px', opacity: 0.5, marginBottom: '8px', color: 'var(--ifm-font-color-base)' }}>
          Position sur le spectre local ↔ global
        </div>
        {params.map((p) => (
          <div key={p.value} style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '4px' }}>
            <span style={{
              width: '26px',
              fontFamily: 'monospace',
              fontSize: '11px',
              fontWeight: p.quality === 'best' ? '800' : '400',
              color: p.quality === 'best' ? color : 'var(--ifm-font-color-base)',
              opacity: p.quality === 'best' ? 1 : 0.5,
              textAlign: 'right',
            }}>{p.label}</span>
            <div style={{ flex: 1 }}>
              <SpectrumBar value={p.value} min={min} max={max} color={QUALITY_STYLES[p.quality].color} />
            </div>
          </div>
        ))}
      </div>

      {/* Rows */}
      <div style={{
        border: '1px solid rgba(255,255,255,0.07)',
        borderRadius: '8px',
        overflow: 'hidden',
      }}>
        {params.map((p, i) => {
          const q = QUALITY_STYLES[p.quality];
          const isBest = p.quality === 'best';
          return (
            <div key={p.value} style={{
              display: 'grid',
              gridTemplateColumns: '56px 1fr auto',
              gap: '12px',
              alignItems: 'center',
              padding: '11px 14px',
              borderBottom: i < params.length - 1 ? '1px solid rgba(255,255,255,0.05)' : 'none',
              background: isBest
                ? `${q.color}0c`
                : 'transparent',
              borderLeft: isBest ? `3px solid ${q.color}` : '3px solid transparent',
            }}>
              <div style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: '14px',
                fontWeight: isBest ? '800' : '500',
                color: isBest ? color : 'var(--ifm-font-color-base)',
                opacity: isBest ? 1 : 0.7,
              }}>{p.label}</div>
              <div style={{
                fontSize: '13px',
                color: 'var(--ifm-font-color-base)',
                opacity: isBest ? 0.9 : 0.6,
                fontWeight: isBest ? '500' : '400',
              }}>{p.behavior}</div>
              <div style={{
                background: q.bg,
                border: `1px solid ${q.border}`,
                borderRadius: '4px',
                padding: '2px 8px',
                fontSize: '10px',
                fontWeight: '700',
                color: q.color,
                whiteSpace: 'nowrap',
              }}>{q.label}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function HyperparamSensitivity() {
  const [active, setActive] = React.useState('umap');

  const tabs = [
    { id: 'umap',  label: 'UMAP — n_neighbors', color: '#F59E0B' },
    { id: 'tsne',  label: 't-SNE — Perplexité',  color: '#A78BFA' },
  ];

  const activeColor = active === 'umap' ? '#F59E0B' : '#A78BFA';

  return (
    <div style={{ margin: '24px 0' }}>
      {/* Tabs */}
      <div style={{
        display: 'flex',
        gap: '4px',
        marginBottom: '0',
        borderBottom: '1px solid rgba(255,255,255,0.08)',
      }}>
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setActive(t.id)}
            style={{
              background: active === t.id ? `${t.color}14` : 'transparent',
              border: 'none',
              borderBottom: active === t.id ? `2px solid ${t.color}` : '2px solid transparent',
              padding: '10px 18px',
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: '13px',
              fontWeight: active === t.id ? '700' : '400',
              color: active === t.id ? t.color : 'var(--ifm-font-color-base)',
              opacity: active === t.id ? 1 : 0.5,
              cursor: 'pointer',
              transition: 'all 0.15s',
              borderRadius: '6px 6px 0 0',
              marginBottom: '-1px',
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{
        background: 'var(--ifm-card-background-color)',
        border: `1px solid ${activeColor}25`,
        borderTop: 'none',
        borderRadius: '0 0 10px 10px',
        padding: '20px',
        transition: 'border-color 0.15s',
      }}>
        {active === 'umap' ? (
          <ParamTable
            params={UMAP_PARAMS}
            paramName="n_neighbors"
            color="#F59E0B"
            unit=""
            min={5}
            max={100}
          />
        ) : (
          <ParamTable
            params={TSNE_PARAMS}
            paramName="perplexité"
            color="#A78BFA"
            unit=""
            min={5}
            max={100}
          />
        )}
      </div>
    </div>
  );
}
