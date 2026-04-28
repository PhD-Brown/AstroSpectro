import React from 'react';

const WL_MIN = 3690;
const WL_MAX = 9100;
const N_PIXELS = 3921;

const FALLBACKS = [
  {
    priority: 1,
    method: 'COEFF0 + COEFF1',
    source: 'Header FITS',
    formula: 'λ = 10^(COEFF0 + i × COEFF1)',
    color: '#34D399',
    status: 'Primaire — ~99 % des spectres LAMOST DR5',
    available: true,
  },
  {
    priority: 2,
    method: 'loglam colonne',
    source: 'Extension données',
    formula: 'λ = 10^loglam[i]',
    color: '#F59E0B',
    status: 'Fallback — certaines campagnes DR3/DR4',
    available: true,
  },
  {
    priority: 3,
    method: 'WCS standard',
    source: 'CRVAL1 + CDELT1',
    formula: 'λ = CRVAL1 + i × CDELT1',
    color: '#A78BFA',
    status: 'Fallback 2 — formats spéciaux',
    available: true,
  },
  {
    priority: 4,
    method: 'Échec de chargement',
    source: 'Aucune grille trouvée',
    formula: 'Spectre ignoré, log dans errors.csv',
    color: '#F87171',
    status: 'Gestion d\'erreur — très rare',
    available: false,
  },
];

// Key spectral lines to display on the ruler
const KEY_LINES = [
  { lambda: 3933, label: 'Ca K', color: '#C084FC' },
  { lambda: 3968, label: 'Ca H', color: '#C084FC' },
  { lambda: 4861, label: 'Hβ',   color: '#60A5FA' },
  { lambda: 5175, label: 'Mg b', color: '#34D399' },
  { lambda: 5895, label: 'Na D', color: '#FB923C' },
  { lambda: 6563, label: 'Hα',   color: '#60A5FA' },
  { lambda: 8498, label: 'CaIR', color: '#C084FC' },
];

// LAMOST spectral regions
const REGIONS = [
  { from: 3690, to: 5900, label: 'Bras bleu', color: '#38BDF8' },
  { from: 5900, to: 9100, label: 'Bras rouge', color: '#F87171' },
];

function wlPct(lambda) {
  return ((lambda - WL_MIN) / (WL_MAX - WL_MIN)) * 100;
}

function SpectralRuler() {
  const [hovLine, setHovLine] = React.useState(null);

  return (
    <div style={{
      background: '#06080f',
      border: '1px solid rgba(255,255,255,0.08)',
      borderRadius: '10px',
      padding: '16px 20px',
      margin: '16px 0',
    }}>
      <div style={{
        fontSize: '10px',
        fontWeight: '700',
        textTransform: 'uppercase',
        letterSpacing: '0.06em',
        opacity: 0.35,
        marginBottom: '14px',
        color: 'var(--ifm-font-color-base)',
        fontFamily: 'monospace',
      }}>
        Grille reconstruite — 3 921 canaux · 3 690–9 100 Å
      </div>

      {/* Spectral regions */}
      <div style={{ display: 'flex', height: '16px', borderRadius: '6px', overflow: 'hidden', marginBottom: '8px' }}>
        {REGIONS.map((r) => {
          const leftPct = wlPct(r.from);
          const widthPct = wlPct(r.to) - leftPct;
          return (
            <div key={r.label} style={{
              flex: `0 0 ${widthPct}%`,
              background: `${r.color}30`,
              borderRight: r.label === 'Bras bleu' ? `2px solid rgba(255,255,255,0.3)` : 'none',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <span style={{ fontSize: '9px', fontFamily: 'monospace', color: r.color, opacity: 0.8 }}>
                {r.label}
              </span>
            </div>
          );
        })}
      </div>

      {/* Spectral line ticks */}
      <div style={{ position: 'relative', height: '36px', marginBottom: '8px' }}>
        {/* Background track */}
        <div style={{
          position: 'absolute',
          left: 0, right: 0, top: '50%',
          height: '2px',
          background: 'rgba(255,255,255,0.05)',
        }} />

        {KEY_LINES.map((line) => {
          const pct = wlPct(line.lambda);
          const isHov = hovLine === line.lambda;
          return (
            <div
              key={line.lambda}
              onMouseEnter={() => setHovLine(line.lambda)}
              onMouseLeave={() => setHovLine(null)}
              style={{
                position: 'absolute',
                left: `${pct}%`,
                top: 0,
                transform: 'translateX(-50%)',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                cursor: 'default',
                zIndex: isHov ? 10 : 1,
              }}
            >
              <div style={{
                fontSize: isHov ? '9.5px' : '8.5px',
                fontFamily: 'monospace',
                fontWeight: '700',
                color: line.color,
                marginBottom: '2px',
                background: isHov ? '#06080f' : 'transparent',
                padding: isHov ? '0 3px' : '0',
                borderRadius: '2px',
                whiteSpace: 'nowrap',
                transition: 'all 0.1s',
              }}>
                {isHov ? `${line.label} ${line.lambda}Å` : line.label}
              </div>
              <div style={{
                width: isHov ? '2px' : '1px',
                height: isHov ? '20px' : '14px',
                background: line.color,
                opacity: isHov ? 1 : 0.7,
                transition: 'all 0.1s',
              }} />
            </div>
          );
        })}
      </div>

      {/* Wavelength axis */}
      <div style={{ position: 'relative', height: '16px', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
        {[3700, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000].map((wl) => (
          <div key={wl} style={{
            position: 'absolute',
            left: `${wlPct(wl)}%`,
            top: '3px',
            transform: 'translateX(-50%)',
            fontSize: '8.5px',
            fontFamily: 'monospace',
            color: 'var(--ifm-font-color-base)',
            opacity: 0.3,
            whiteSpace: 'nowrap',
          }}>{wl}</div>
        ))}
      </div>

      {/* Pixel count */}
      <div style={{
        marginTop: '8px',
        fontSize: '10px',
        fontFamily: 'monospace',
        opacity: 0.3,
        color: 'var(--ifm-font-color-base)',
        textAlign: 'right',
      }}>
        N = {N_PIXELS.toLocaleString('fr-FR')} canaux · Δλ/pixel ≈ 1,36 Å · Survoler les raies
      </div>
    </div>
  );
}

export default function WavelengthReconstruction() {
  const [checkedIdx, setCheckedIdx] = React.useState(0);

  return (
    <div style={{ margin: '28px 0' }}>
      {/* Fallback chain */}
      <div style={{
        background: 'var(--ifm-card-background-color)',
        border: '1px solid rgba(255,255,255,0.07)',
        borderRadius: '10px',
        overflow: 'hidden',
        marginBottom: '0',
      }}>
        <div style={{
          padding: '10px 16px',
          background: 'rgba(255,255,255,0.02)',
          borderBottom: '1px solid rgba(255,255,255,0.06)',
          fontSize: '11px',
          fontWeight: '700',
          textTransform: 'uppercase',
          letterSpacing: '0.06em',
          opacity: 0.4,
          color: 'var(--ifm-font-color-base)',
        }}>
          Chaîne de fallback — reconstruction robuste de la grille λ
        </div>

        {FALLBACKS.map((f, i) => {
          const isActive = checkedIdx === i;
          return (
            <div
              key={i}
              onClick={() => setCheckedIdx(i)}
              style={{
                display: 'grid',
                gridTemplateColumns: '32px 50px 160px 1fr 1fr',
                alignItems: 'center',
                gap: '12px',
                padding: '12px 16px',
                borderBottom: i < FALLBACKS.length - 1 ? '1px solid rgba(255,255,255,0.04)' : 'none',
                background: isActive ? `${f.color}0a` : 'transparent',
                cursor: 'pointer',
                transition: 'background 0.12s',
                opacity: f.available ? 1 : 0.5,
              }}
            >
              {/* Priority */}
              <div style={{
                width: '24px',
                height: '24px',
                borderRadius: '50%',
                background: isActive ? f.color : `${f.color}22`,
                border: `2px solid ${f.color}50`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '11px',
                fontWeight: '800',
                color: isActive ? '#000' : f.color,
                flexShrink: 0,
                transition: 'all 0.12s',
              }}>{f.priority}</div>

              {/* Available? */}
              <div style={{
                fontSize: '14px',
                textAlign: 'center',
              }}>
                {f.available
                  ? <span style={{ color: '#34D399' }}>✓</span>
                  : <span style={{ color: '#F87171' }}>✗</span>
                }
              </div>

              {/* Method */}
              <code style={{
                fontSize: '12px',
                fontWeight: '700',
                color: isActive ? f.color : 'var(--ifm-font-color-base)',
                opacity: isActive ? 1 : 0.8,
                transition: 'color 0.12s',
              }}>{f.method}</code>

              {/* Formula */}
              <code style={{
                fontSize: '11.5px',
                color: f.color,
                opacity: isActive ? 0.9 : 0.55,
                fontFamily: 'monospace',
              }}>{f.formula}</code>

              {/* Status */}
              <div style={{
                fontSize: '11.5px',
                color: 'var(--ifm-font-color-base)',
                opacity: 0.55,
                fontStyle: 'italic',
              }}>{f.status}</div>
            </div>
          );
        })}
      </div>

      {/* Spectral ruler */}
      <SpectralRuler />

      {/* Code block for selected method */}
      <div style={{
        background: '#06080f',
        border: `1px solid ${FALLBACKS[checkedIdx].color}25`,
        borderRadius: '8px',
        padding: '16px',
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: '12.5px',
        lineHeight: 1.7,
      }}>
        <div style={{ fontSize: '10px', opacity: 0.35, marginBottom: '8px', color: 'var(--ifm-font-color-base)' }}>
          src/pipeline/preprocessor.py — Méthode {FALLBACKS[checkedIdx].priority} : {FALLBACKS[checkedIdx].method}
        </div>
        {checkedIdx === 0 && (
          <pre style={{ margin: 0, color: '#34D399' }}>{`# Méthode primaire — COEFF0 + COEFF1 (log-linéaire)
coeff0 = header['COEFF0']      # λ₀ du premier pixel (log10)
coeff1 = header['COEFF1']      # pas logarithmique / pixel
n_pix  = flux.shape[0]         # 3921 canaux
loglam  = coeff0 + np.arange(n_pix) * coeff1
wavelength = 10 ** loglam      # → Ångströms [3690 ... 9100]`}</pre>
        )}
        {checkedIdx === 1 && (
          <pre style={{ margin: 0, color: '#F59E0B' }}>{`# Fallback — colonne loglam dans l'extension de données
loglam     = hdul[0].data[4]   # colonne log-wavelength stockée
wavelength = 10 ** loglam      # → Ångströms`}</pre>
        )}
        {checkedIdx === 2 && (
          <pre style={{ margin: 0, color: '#A78BFA' }}>{`# Fallback 2 — WCS standard
crval1 = header['CRVAL1']      # λ au pixel de référence
cdelt1 = header['CDELT1']      # pas linéaire / pixel
crpix1 = header.get('CRPIX1', 1)
pixel  = np.arange(n_pix) - (crpix1 - 1)
wavelength = crval1 + pixel * cdelt1`}</pre>
        )}
        {checkedIdx === 3 && (
          <pre style={{ margin: 0, color: '#F87171' }}>{`# Gestion d'erreur — aucun format reconnu
logger.warning(f"Impossible de reconstruire λ pour {obsid}")
error_log.append({'obsid': obsid, 'reason': 'no_wavelength_grid'})
return None  # spectre ignoré du pipeline`}</pre>
        )}
      </div>
    </div>
  );
}
