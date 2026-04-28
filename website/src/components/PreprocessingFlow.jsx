import React from 'react';

const STEPS = [
  {
    id: 1,
    name: 'Lecture FITS',
    sub: 'fits.gz → mémoire',
    color: '#38BDF8',
    icon: '📄',
    details: [
      'Ouverture avec astropy.io.fits',
      'Extraction flux + invvar depuis extension 0',
      'Lecture du header pour les métadonnées',
      'Gestion robuste des valeurs manquantes',
    ],
    code: 'hdul = fits.open(fits_path)\nflux   = hdul[0].data[0]\ninvvar = hdul[0].data[1]',
  },
  {
    id: 2,
    name: 'Reconstruction λ',
    sub: 'grille de longueurs d\'onde',
    color: '#A78BFA',
    icon: 'λ',
    details: [
      'Priorité 1 : COEFF0 + COEFF1 (log-linéaire)',
      'Fallback : colonne loglam dans les données',
      'Fallback 2 : WCS standard (CRVAL1 + CDELT1)',
      'Résultat : 3 921 valeurs de 3 690 à 9 100 Å',
    ],
    code: 'loglam = coeff0 + np.arange(n) * coeff1\nwavelength = 10 ** loglam  # en Ångströms',
  },
  {
    id: 3,
    name: 'Normalisation',
    sub: 'flux / médiane → continuum ≈ 1',
    color: '#F59E0B',
    icon: 'σ',
    details: [
      'Division par la médiane du flux',
      'Robuste aux raies d\'émission et aux outliers',
      'Ramène le continuum à ~1.0',
      'Élimine les différences de magnitude absolue',
    ],
    code: 'median = np.median(flux[flux > 0])\nflux_norm = flux / median  if median > 0  else flux',
  },
  {
    id: 4,
    name: 'Contrôle qualité',
    sub: 'SNR · RUWE · masques',
    color: '#34D399',
    icon: '✓',
    details: [
      'SNR_r > 10 — raies détectables',
      'RUWE < 1,4 — astrométrie Gaia fiable',
      'Pixels masqués exclus du calcul des raies',
      '43 019 / ~250 000 spectres retenus',
    ],
    code: 'valid = (snr_r > 10) & (ruwe < 1.4)\ncatalog_clean = catalog[valid]',
  },
  {
    id: 5,
    name: 'Extraction features',
    sub: '183 descripteurs physiques',
    color: '#C084FC',
    icon: 'p',
    details: [
      'Raies : prominence, FWHM, EW par specutils',
      'Pentes continuum sigma-clippées (σ = 2,5)',
      'Courbures Savitzky-Golay (dérivée seconde)',
      'Indices Lick, bandes moléculaires, couleurs',
    ],
    code: 'vec = fe.extract_features(\n    matched_lines, wl,\n    flux_norm, invvar\n)',
  },
];

function Step({ step, isActive, onClick }) {
  return (
    <div
      onClick={onClick}
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '6px',
        cursor: 'pointer',
        flex: 1,
        minWidth: 0,
      }}
    >
      {/* Circle */}
      <div style={{
        width: '48px',
        height: '48px',
        borderRadius: '50%',
        background: isActive ? step.color : `${step.color}20`,
        border: `2px solid ${isActive ? step.color : step.color + '40'}`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '18px',
        transition: 'all 0.15s',
        boxShadow: isActive ? `0 0 0 4px ${step.color}22` : 'none',
        flexShrink: 0,
      }}>
        <span style={{ fontSize: '16px', lineHeight: 1, color: isActive ? '#000' : step.color }}>{step.icon}</span>
      </div>

      {/* Label */}
      <div style={{ textAlign: 'center' }}>
        <div style={{
          fontSize: '11.5px',
          fontWeight: isActive ? '700' : '500',
          color: isActive ? step.color : 'var(--ifm-font-color-base)',
          opacity: isActive ? 1 : 0.65,
          lineHeight: 1.3,
          transition: 'color 0.15s',
        }}>{step.name}</div>
        <div style={{
          fontSize: '9.5px',
          color: 'var(--ifm-font-color-base)',
          opacity: 0.4,
          marginTop: '2px',
          fontFamily: 'monospace',
        }}>{step.sub}</div>
      </div>
    </div>
  );
}

function Connector({ color }) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      flexShrink: 0,
      paddingBottom: '22px',
    }}>
      <div style={{
        width: '28px',
        height: '2px',
        background: `linear-gradient(90deg, ${color}60, rgba(255,255,255,0.1))`,
        borderRadius: '1px',
      }} />
    </div>
  );
}

export default function PreprocessingFlow() {
  const [active, setActive] = React.useState(1);
  const step = STEPS.find((s) => s.id === active);

  return (
    <div style={{ margin: '28px 0' }}>
      {/* Step selector */}
      <div style={{
        background: 'var(--ifm-card-background-color)',
        border: '1px solid rgba(255,255,255,0.07)',
        borderRadius: '12px 12px 0 0',
        padding: '20px 20px 14px',
        borderBottom: 'none',
      }}>
        <div style={{
          fontSize: '10px',
          fontWeight: '700',
          textTransform: 'uppercase',
          letterSpacing: '0.07em',
          opacity: 0.35,
          marginBottom: '16px',
          color: 'var(--ifm-font-color-base)',
        }}>
          Pipeline de prétraitement — cliquer sur une étape
        </div>

        <div style={{ display: 'flex', alignItems: 'center' }}>
          {STEPS.map((s, i) => (
            <React.Fragment key={s.id}>
              <Step step={s} isActive={active === s.id} onClick={() => setActive(s.id)} />
              {i < STEPS.length - 1 && <Connector color={STEPS[i + 1].color} />}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Detail panel */}
      <div style={{
        background: `${step.color}06`,
        border: `1px solid ${step.color}30`,
        borderTop: `2px solid ${step.color}`,
        borderRadius: '0 0 12px 12px',
        padding: '20px',
        transition: 'all 0.2s',
      }}>
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '16px',
          alignItems: 'flex-start',
        }}>
          {/* Left: details */}
          <div>
            <div style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontWeight: '800',
              fontSize: '15px',
              color: step.color,
              marginBottom: '12px',
            }}>
              Étape {step.id} — {step.name}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {step.details.map((d, i) => (
                <div key={i} style={{
                  display: 'flex',
                  gap: '8px',
                  fontSize: '12.5px',
                  color: 'var(--ifm-font-color-base)',
                  opacity: 0.8,
                  lineHeight: 1.5,
                }}>
                  <span style={{ color: step.color, flexShrink: 0, fontWeight: '700' }}>›</span>
                  {d}
                </div>
              ))}
            </div>
          </div>

          {/* Right: code */}
          <div style={{
            background: '#06080f',
            border: `1px solid ${step.color}22`,
            borderRadius: '8px',
            padding: '14px',
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: '12px',
            color: step.color,
            lineHeight: 1.7,
            whiteSpace: 'pre',
            overflow: 'auto',
          }}>
            {step.code}
          </div>
        </div>
      </div>
    </div>
  );
}
