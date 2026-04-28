import React from 'react';

const ENCODER_LAYERS = [
  { dim: 183, label: 'Entrée X', color: '#94A3B8', desc: '183 descripteurs normalisés', role: 'input' },
  { dim: 256, label: 'Dense 256', color: '#38BDF8', desc: 'Linear → BatchNorm → ReLU → Drop(0.1)', role: 'hidden' },
  { dim: 128, label: 'Dense 128', color: '#60A5FA', desc: 'Linear → BatchNorm → ReLU → Drop(0.1)', role: 'hidden' },
  { dim: 64,  label: 'Dense 64',  color: '#818CF8', desc: 'Linear → BatchNorm → ReLU → Drop(0.1)', role: 'hidden' },
  { dim: 2,   label: 'z  (latent)', color: '#C084FC', desc: 'Linear — espace latent 2D', role: 'latent' },
];

const DECODER_LAYERS = [
  { dim: 64,  label: 'Dense 64',  color: '#818CF8', desc: 'Linear → BatchNorm → ReLU → Drop(0.1)', role: 'hidden' },
  { dim: 128, label: 'Dense 128', color: '#60A5FA', desc: 'Linear → BatchNorm → ReLU → Drop(0.1)', role: 'hidden' },
  { dim: 256, label: 'Dense 256', color: '#38BDF8', desc: 'Linear → BatchNorm → ReLU → Drop(0.1)', role: 'hidden' },
  { dim: 183, label: 'Sortie X̂', color: '#34D399', desc: 'Linear — reconstruction', role: 'output' },
];

const MAX_DIM = 256;
const MAX_HEIGHT = 80;
const MIN_HEIGHT = 6;

function barH(dim) {
  return MIN_HEIGHT + ((dim / MAX_DIM) * (MAX_HEIGHT - MIN_HEIGHT));
}

function Layer({ layer, isActive, onClick }) {
  const h = barH(layer.dim);
  const [hov, setHov] = React.useState(false);
  const highlighted = isActive || hov;

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '6px',
        cursor: 'pointer',
        flex: 1,
        position: 'relative',
      }}
    >
      {/* Dim label top */}
      <div style={{
        fontFamily: 'monospace',
        fontSize: '11px',
        fontWeight: '800',
        color: layer.color,
        opacity: highlighted ? 1 : 0.6,
        transition: 'opacity 0.12s',
      }}>{layer.dim}</div>

      {/* Bar */}
      <div style={{
        width: layer.role === 'latent' ? '18px' : '32px',
        height: `${h}px`,
        background: highlighted
          ? layer.color
          : layer.role === 'latent' ? `${layer.color}cc` : `${layer.color}55`,
        borderRadius: '4px',
        transition: 'all 0.15s',
        boxShadow: highlighted ? `0 0 12px ${layer.color}60` : 'none',
        border: layer.role === 'latent' ? `2px solid ${layer.color}` : 'none',
      }} />

      {/* Label bottom */}
      <div style={{
        fontSize: '9px',
        fontWeight: layer.role === 'latent' ? '800' : '500',
        color: highlighted ? layer.color : 'var(--ifm-font-color-base)',
        opacity: highlighted ? 1 : 0.5,
        textAlign: 'center',
        whiteSpace: 'nowrap',
        transition: 'all 0.12s',
        lineHeight: 1.3,
      }}>{layer.label}</div>
    </div>
  );
}

function Arrow({ label, color }) {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      flexShrink: 0,
      gap: '2px',
      paddingBottom: '30px',
    }}>
      <div style={{ width: '1px', height: '20px', background: `${color}40` }} />
      <div style={{ fontSize: '10px', color, opacity: 0.5, fontFamily: 'monospace', fontSize: '14px' }}>›</div>
    </div>
  );
}

export default function AEArchitecture() {
  const [activeLayer, setActiveLayer] = React.useState(4); // latent by default
  const allLayers = [...ENCODER_LAYERS, ...DECODER_LAYERS];
  const active = allLayers[activeLayer < ENCODER_LAYERS.length ? activeLayer : activeLayer];

  // combined for display
  const combined = [...ENCODER_LAYERS, ...DECODER_LAYERS];

  return (
    <div style={{ margin: '28px 0' }}>
      <div style={{
        background: 'var(--ifm-card-background-color)',
        border: '1px solid rgba(255,255,255,0.07)',
        borderRadius: '12px',
        overflow: 'hidden',
      }}>
        {/* Header */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '12px 18px',
          background: 'rgba(255,255,255,0.02)',
          borderBottom: '1px solid rgba(255,255,255,0.06)',
          flexWrap: 'wrap',
          gap: '8px',
        }}>
          <div>
            <div style={{ fontSize: '13px', fontWeight: '700', color: 'var(--ifm-font-color-base)' }}>
              Architecture SpectralAutoencoder
            </div>
            <div style={{ fontSize: '10px', opacity: 0.4, color: 'var(--ifm-font-color-base)' }}>
              MLP symétrique · BatchNorm + Dropout(0.1) · Adam lr=1e-3
            </div>
          </div>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            {[
              { label: 'Encodeur', color: '#38BDF8' },
              { label: 'Espace latent z=2', color: '#C084FC' },
              { label: 'Décodeur', color: '#34D399' },
            ].map((t) => (
              <div key={t.label} style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '10px', opacity: 0.6, color: t.color }}>
                <div style={{ width: '10px', height: '10px', borderRadius: '2px', background: t.color }} />
                {t.label}
              </div>
            ))}
          </div>
        </div>

        {/* Architecture diagram */}
        <div style={{ padding: '24px 20px 10px' }}>
          {/* Section labels */}
          <div style={{ display: 'flex', marginBottom: '8px' }}>
            <div style={{ flex: ENCODER_LAYERS.length, textAlign: 'center', fontSize: '9px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.06em', color: '#38BDF8', opacity: 0.6 }}>
              ← ENCODEUR
            </div>
            <div style={{ width: '20px' }} />
            <div style={{ flex: DECODER_LAYERS.length, textAlign: 'center', fontSize: '9px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.06em', color: '#34D399', opacity: 0.6 }}>
              DÉCODEUR →
            </div>
          </div>

          {/* Layers */}
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: '0' }}>
            {ENCODER_LAYERS.map((layer, i) => (
              <React.Fragment key={`enc-${i}`}>
                <Layer
                  layer={layer}
                  isActive={activeLayer === i}
                  onClick={() => setActiveLayer(i)}
                />
                {i < ENCODER_LAYERS.length - 1 && (
                  <Arrow color={layer.color} />
                )}
                {i === ENCODER_LAYERS.length - 1 && (
                  <div style={{
                    display: 'flex', flexDirection: 'column', alignItems: 'center',
                    justifyContent: 'center', flexShrink: 0, paddingBottom: '30px', width: '20px',
                  }}>
                    <div style={{ fontFamily: 'monospace', fontSize: '12px', color: '#C084FC', opacity: 0.6 }}>⇌</div>
                  </div>
                )}
              </React.Fragment>
            ))}
            {DECODER_LAYERS.map((layer, i) => (
              <React.Fragment key={`dec-${i}`}>
                <Layer
                  layer={layer}
                  isActive={activeLayer === ENCODER_LAYERS.length + i}
                  onClick={() => setActiveLayer(ENCODER_LAYERS.length + i)}
                />
                {i < DECODER_LAYERS.length - 1 && (
                  <Arrow color={layer.color} />
                )}
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* Detail panel */}
        <div style={{
          margin: '0 16px 16px',
          padding: '12px 14px',
          background: `${combined[activeLayer].color}0a`,
          border: `1px solid ${combined[activeLayer].color}25`,
          borderRadius: '8px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '8px',
        }}>
          <div>
            <div style={{ fontFamily: 'monospace', fontWeight: '800', fontSize: '14px', color: combined[activeLayer].color }}>
              {combined[activeLayer].label} — dim {combined[activeLayer].dim}
            </div>
            <div style={{ fontSize: '11.5px', color: 'var(--ifm-font-color-base)', opacity: 0.6, marginTop: '3px' }}>
              {combined[activeLayer].desc}
            </div>
          </div>
          {combined[activeLayer].role === 'latent' && (
            <div style={{
              background: '#C084FC18', border: '1px solid #C084FC35',
              borderRadius: '7px', padding: '8px 14px', textAlign: 'center',
            }}>
              <div style={{ fontSize: '10px', opacity: 0.5, marginBottom: '3px', color: 'var(--ifm-font-color-base)' }}>Taux de compression</div>
              <div style={{ fontFamily: 'monospace', fontSize: '20px', fontWeight: '900', color: '#C084FC' }}>183 → 2</div>
              <div style={{ fontSize: '9px', opacity: 0.4, color: 'var(--ifm-font-color-base)' }}>91,5× (vs 91 pour PCA@95%)</div>
            </div>
          )}
        </div>

        {/* Params row */}
        <div style={{
          padding: '10px 18px 14px',
          borderTop: '1px solid rgba(255,255,255,0.05)',
          display: 'flex',
          flexWrap: 'wrap',
          gap: '10px',
        }}>
          {[
            { k: 'Paramètres total',  v: '~330k' },
            { k: 'Batch size',         v: '512' },
            { k: 'Optimiseur',         v: 'Adam · lr=1e-3' },
            { k: 'Perte',              v: 'MSE (reconstruction)' },
            { k: 'Early stopping',     v: 'patience=15 · val_loss' },
            { k: 'Device',             v: 'CPU (Ryzen 9 5950X)' },
          ].map((p) => (
            <div key={p.k} style={{
              display: 'flex', flexDirection: 'column', gap: '2px',
              padding: '7px 10px',
              background: 'rgba(255,255,255,0.02)',
              border: '1px solid rgba(255,255,255,0.05)',
              borderRadius: '6px',
            }}>
              <div style={{ fontSize: '9px', opacity: 0.35, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--ifm-font-color-base)' }}>{p.k}</div>
              <div style={{ fontFamily: 'monospace', fontSize: '11.5px', fontWeight: '600', color: 'var(--ifm-font-color-base)' }}>{p.v}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
