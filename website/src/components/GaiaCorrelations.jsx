import React from 'react';

const CORRELATIONS = [
  {
    param: 'Teff',
    unit: 'K',
    symbol: 'T_eff',
    rho1: +0.831,
    rho2: +0.156,
    note1: 'axe thermique dominant',
    note2: null,
  },
  {
    param: 'log g',
    unit: 'dex',
    symbol: 'log g',
    rho1: -0.155,
    rho2: +0.081,
    note1: null,
    note2: null,
  },
  {
    param: '[Fe/H]',
    unit: 'dex',
    symbol: '[Fe/H]',
    rho1: -0.548,
    rho2: -0.082,
    note1: 'PC1 capture la métallicité',
    note2: null,
  },
  {
    param: 'G_BP − G_RP',
    unit: 'mag',
    symbol: 'G_BP-G_RP',
    rho1: -0.768,
    rho2: -0.025,
    note1: 'couleur photométrique ∝ Teff',
    note2: null,
  },
  {
    param: 'SNR_r',
    unit: '',
    symbol: 'SNR',
    rho1: +0.071,
    rho2: -0.310,
    note1: null,
    note2: 'contamination instrumentale',
    warn2: true,
  },
  {
    param: 'A_G',
    unit: 'mag',
    symbol: 'A_G',
    rho1: -0.070,
    rho2: +0.196,
    note1: null,
    note2: null,
  },
];

// Color based on rho value: strong positive = warm, strong negative = cool, near zero = neutral
function rhoColor(rho) {
  const abs = Math.abs(rho);
  if (abs >= 0.7) return rho > 0 ? '#F59E0B' : '#38BDF8';
  if (abs >= 0.4) return rho > 0 ? '#FBBF2488' : '#60A5FA88';
  return 'rgba(255,255,255,0.1)';
}

function rhoTextColor(rho) {
  const abs = Math.abs(rho);
  if (abs >= 0.7) return rho > 0 ? '#F59E0B' : '#38BDF8';
  if (abs >= 0.4) return rho > 0 ? '#FBBF24' : '#60A5FA';
  return 'var(--ifm-font-color-base)';
}

function CorrelationCell({ rho, note, warn }) {
  const [hov, setHov] = React.useState(false);
  const abs = Math.abs(rho);
  const isSignificant = abs >= 0.4;
  const color = rhoColor(rho);
  const textColor = rhoTextColor(rho);

  return (
    <div
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        padding: '10px 14px',
        background: hov ? color : 'transparent',
        transition: 'background 0.15s',
        cursor: 'default',
        textAlign: 'center',
        position: 'relative',
      }}
    >
      {/* Filled bar background */}
      <div style={{
        position: 'absolute',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        width: `${abs * 100}%`,
        height: '70%',
        background: color,
        borderRadius: '4px',
        opacity: hov ? 0 : 1,
        transition: 'opacity 0.15s',
      }} />

      {/* Value */}
      <div style={{
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: '15px',
        fontWeight: isSignificant ? '800' : '400',
        color: hov ? (abs > 0.5 ? '#000' : textColor) : textColor,
        position: 'relative',
        zIndex: 1,
        opacity: isSignificant ? 1 : 0.6,
        transition: 'color 0.15s',
      }}>
        {rho > 0 ? '+' : ''}{rho.toFixed(3)}
      </div>

      {/* Note tooltip */}
      {note && hov && (
        <div style={{
          position: 'absolute',
          bottom: 'calc(100% + 4px)',
          left: '50%',
          transform: 'translateX(-50%)',
          background: warn ? '#92400E' : 'var(--ifm-card-background-color)',
          border: `1px solid ${warn ? '#F59E0B' : 'rgba(255,255,255,0.2)'}`,
          borderRadius: '5px',
          padding: '4px 8px',
          fontSize: '10px',
          color: warn ? '#FCD34D' : 'var(--ifm-font-color-base)',
          whiteSpace: 'nowrap',
          zIndex: 20,
          boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
        }}>
          {note}
        </div>
      )}
    </div>
  );
}

export default function GaiaCorrelations() {
  return (
    <div style={{ margin: '28px 0' }}>
      {/* Scale legend */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        marginBottom: '14px',
        fontSize: '11px',
        color: 'var(--ifm-font-color-base)',
        opacity: 0.6,
      }}>
        <span>Intensité de couleur ∝ |ρ| :</span>
        {[
          { label: '|ρ| ≥ 0,7', bg: '#F59E0B', text: 'fort positif' },
          { label: '|ρ| ≥ 0,7', bg: '#38BDF8', text: 'fort négatif' },
          { label: '|ρ| ≥ 0,4', bg: '#FBBF2455', text: 'modéré' },
          { label: '|ρ| < 0,4', bg: 'rgba(255,255,255,0.08)', text: 'faible' },
        ].map((s, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <div style={{ width: '14px', height: '14px', borderRadius: '3px', background: s.bg, flexShrink: 0 }} />
            <span style={{ fontSize: '10px' }}>{s.text}</span>
          </div>
        ))}
      </div>

      <div style={{
        border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: '10px',
        overflow: 'hidden',
      }}>
        {/* Header */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '160px 60px 1fr 1fr',
          background: 'rgba(255,255,255,0.03)',
          borderBottom: '1px solid rgba(255,255,255,0.08)',
        }}>
          <div style={{ padding: '10px 14px', fontSize: '11px', fontWeight: '700', opacity: 0.5, color: 'var(--ifm-font-color-base)' }}>
            Paramètre Gaia DR3
          </div>
          <div style={{ padding: '10px 6px', fontSize: '11px', fontWeight: '700', opacity: 0.5, textAlign: 'center', color: 'var(--ifm-font-color-base)' }}>
            Unité
          </div>
          <div style={{
            padding: '10px 14px',
            fontSize: '12px',
            fontWeight: '800',
            color: '#38BDF8',
            textAlign: 'center',
            borderBottom: '2px solid #38BDF8',
          }}>
            ρ(PC1, ·)
          </div>
          <div style={{
            padding: '10px 14px',
            fontSize: '12px',
            fontWeight: '800',
            color: '#F59E0B',
            textAlign: 'center',
            borderBottom: '2px solid #F59E0B',
          }}>
            ρ(PC2, ·)
          </div>
        </div>

        {/* Rows */}
        {CORRELATIONS.map((row, ri) => (
          <div
            key={row.param}
            style={{
              display: 'grid',
              gridTemplateColumns: '160px 60px 1fr 1fr',
              borderBottom: ri < CORRELATIONS.length - 1
                ? '1px solid rgba(255,255,255,0.04)' : 'none',
              background: ri % 2 === 0 ? 'rgba(255,255,255,0.01)' : 'transparent',
            }}
          >
            <div style={{
              padding: '0 14px',
              display: 'flex',
              alignItems: 'center',
              borderRight: '1px solid rgba(255,255,255,0.04)',
            }}>
              <span style={{
                fontFamily: 'monospace',
                fontSize: '13px',
                fontWeight: '600',
                color: 'var(--ifm-font-color-base)',
              }}>{row.param}</span>
            </div>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              borderRight: '1px solid rgba(255,255,255,0.04)',
            }}>
              <span style={{
                fontSize: '10px',
                fontFamily: 'monospace',
                color: 'var(--ifm-font-color-base)',
                opacity: 0.4,
              }}>{row.unit}</span>
            </div>
            <div style={{ borderRight: '1px solid rgba(255,255,255,0.04)' }}>
              <CorrelationCell rho={row.rho1} note={row.note1} />
            </div>
            <div>
              <CorrelationCell rho={row.rho2} note={row.note2} warn={row.warn2} />
            </div>
          </div>
        ))}
      </div>

      {/* Interpretation footer */}
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
          <strong style={{ color: '#38BDF8' }}>PC1 = axe thermique</strong> — dominé par Balmer (31,5 %) + Ca II (22,4 %). Gradient continu de Teff (étoiles chaudes A-F à droite, froides K-M à gauche).
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
          <strong style={{ color: '#F59E0B' }}>PC2 = axe métallicité</strong> — dominé par métaux (43,3 %). ⚠️ Partiellement contaminé par SNR (ρ = −0,310) : artefact instrumental inhérent à la PCA non supervisée.
        </div>
      </div>
    </div>
  );
}
