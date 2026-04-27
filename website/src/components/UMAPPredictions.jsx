import React from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';

// Class positions in UMAP (approximate, for the annotation overlay)
const CLASS_POSITIONS = [
  {
    letter: 'A',
    label: 'Péninsule chaude',
    color: '#93C5FD',
    desc: 'Supérieure gauche — étoiles A-F chaudes',
    teff: '> 7 500 K',
    x: '15%',
    y: '12%',
  },
  {
    letter: 'F',
    label: 'Transition chaude',
    color: '#FEF9C3',
    desc: 'Zone intermédiaire haute',
    teff: '6 000–7 500 K',
    x: '30%',
    y: '25%',
  },
  {
    letter: 'G',
    label: 'Masse centrale',
    color: '#FDE68A',
    desc: 'Masse principale centrale',
    teff: '5 200–6 000 K',
    x: '52%',
    y: '42%',
  },
  {
    letter: 'K',
    label: 'Périphérie froide',
    color: '#FB923C',
    desc: 'Bordure froide de la masse centrale',
    teff: '3 700–5 200 K',
    x: '68%',
    y: '58%',
  },
  {
    letter: 'M',
    label: 'Extrémité froide',
    color: '#F87171',
    desc: 'Branche inférieure — naines M',
    teff: '< 3 700 K',
    x: '80%',
    y: '75%',
  },
];

function ClassBadge({ letter, label, color, desc, teff }) {
  const [hov, setHov] = React.useState(false);
  return (
    <div
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'flex',
        gap: '10px',
        alignItems: 'flex-start',
        padding: '9px 12px',
        background: hov ? `${color}15` : `${color}08`,
        border: `1px solid ${hov ? color + '50' : color + '25'}`,
        borderLeft: `3px solid ${color}`,
        borderRadius: '7px',
        transition: 'all 0.13s',
        cursor: 'default',
      }}
    >
      {/* Letter bubble */}
      <div style={{
        width: '28px',
        height: '28px',
        borderRadius: '50%',
        background: `radial-gradient(circle at 35% 35%, ${color}cc, ${color}88)`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontFamily: 'Georgia, serif',
        fontWeight: '900',
        fontSize: '14px',
        color: '#000',
        flexShrink: 0,
        boxShadow: `0 0 8px ${color}44`,
      }}>{letter}</div>

      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{
          fontWeight: '700',
          fontSize: '12px',
          color: color,
          marginBottom: '1px',
        }}>{label}</div>
        <div style={{
          fontSize: '10.5px',
          color: 'var(--ifm-font-color-base)',
          opacity: 0.6,
          lineHeight: 1.3,
        }}>{desc}</div>
        <div style={{
          fontFamily: 'monospace',
          fontSize: '10px',
          color: color,
          opacity: 0.75,
          marginTop: '3px',
        }}>{teff}</div>
      </div>

      {/* Checkmark */}
      <div style={{
        color: '#34D399',
        fontSize: '16px',
        fontWeight: '800',
        flexShrink: 0,
        alignSelf: 'center',
      }}>✓</div>
    </div>
  );
}

export default function UMAPPredictions() {
  const [activeImg, setActiveImg] = React.useState('predictions');
  const [imgLoaded, setImgLoaded] = React.useState(false);
  const [imgError, setImgError] = React.useState(null);

  const predictionsSrc = useBaseUrl('/img/dimred/umap_xgboost_predictions.png');
  const fgConfusionSrc = useBaseUrl('/img/dimred/umap_xgboost_FG_confusion.png');

  const imgs = {
    predictions: {
      src: predictionsSrc,
      label: 'Prédictions XGBoost',
      desc: 'Chaque point coloré par le type spectral prédit par XGBoost',
    },
    fg_confusion: {
      src: fgConfusionSrc,
      label: 'Confusion F/G',
      desc: 'Zone de confusion F/G — continuité physique à 5 500–6 200 K',
    },
  };

  const current = imgs[activeImg];

  return (
    <div style={{ margin: '28px 0' }}>
      {/* Tab switcher */}
      <div style={{
        display: 'flex',
        gap: '4px',
        marginBottom: '14px',
        flexWrap: 'wrap',
      }}>
        {Object.entries(imgs).map(([key, img]) => (
          <button
            key={key}
            onClick={() => {
              setActiveImg(key);
              setImgLoaded(false);
              setImgError(null);
            }}
            style={{
              background: activeImg === key ? 'rgba(245,158,11,0.12)' : 'transparent',
              border: `1px solid ${activeImg === key ? '#F59E0B55' : 'rgba(255,255,255,0.1)'}`,
              borderBottom: activeImg === key ? '2px solid #F59E0B' : '2px solid transparent',
              borderRadius: '6px 6px 0 0',
              padding: '8px 16px',
              fontSize: '12px',
              fontWeight: activeImg === key ? '700' : '400',
              color: activeImg === key ? '#F59E0B' : 'var(--ifm-font-color-base)',
              opacity: activeImg === key ? 1 : 0.6,
              cursor: 'pointer',
              transition: 'all 0.13s',
            }}
          >
            {img.label}
          </button>
        ))}
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: '1.6fr 1fr',
        gap: '16px',
        alignItems: 'flex-start',
      }}>
        {/* Image panel */}
        <div style={{
          background: 'var(--ifm-card-background-color)',
          border: '1px solid rgba(245,158,11,0.2)',
          borderRadius: '0 10px 10px 10px',
          overflow: 'hidden',
        }}>
          {/* Image */}
          <div style={{
            background: '#0a0a0f',
            position: 'relative',
            minHeight: '220px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}>
            {!imgLoaded && !imgError && (
              <div style={{
                position: 'absolute',
                inset: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'rgba(255,255,255,0.2)',
                fontSize: '12px',
                fontFamily: 'monospace',
              }}>
                Chargement…
              </div>
            )}
            {imgError && (
              <div style={{
                position: 'absolute',
                inset: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'rgba(248,113,113,0.9)',
                fontSize: '12px',
                fontFamily: 'monospace',
                textAlign: 'center',
                padding: '12px',
              }}>
                {imgError}
              </div>
            )}
            <img
              key={current.src}
              src={current.src}
              alt={current.label}
              onLoad={() => {
                setImgLoaded(true);
                setImgError(null);
              }}
              onError={() => {
                setImgLoaded(false);
                setImgError('Image introuvable ou chemin invalide.');
              }}
              style={{
                width: '100%',
                height: 'auto',
                display: 'block',
                opacity: imgLoaded ? 1 : 0,
                transition: 'opacity 0.3s',
              }}
            />
          </div>
          {/* Caption */}
          <div style={{
            padding: '10px 14px',
            borderTop: '1px solid rgba(255,255,255,0.05)',
            fontSize: '11px',
            color: 'var(--ifm-font-color-base)',
            opacity: 0.6,
            fontStyle: 'italic',
          }}>{current.desc}</div>
        </div>

        {/* Legend / geography */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <div style={{
            fontSize: '11px',
            fontWeight: '700',
            textTransform: 'uppercase',
            letterSpacing: '0.06em',
            opacity: 0.4,
            color: 'var(--ifm-font-color-base)',
            marginBottom: '4px',
          }}>
            Géographie des classes
          </div>
          {CLASS_POSITIONS.map((c) => <ClassBadge key={c.letter} {...c} />)}

          {/* Coherence note */}
          <div style={{
            marginTop: '4px',
            padding: '10px 12px',
            background: 'rgba(52,211,153,0.06)',
            border: '1px solid rgba(52,211,153,0.2)',
            borderRadius: '7px',
            fontSize: '11px',
            color: 'var(--ifm-font-color-base)',
            opacity: 0.8,
            lineHeight: 1.5,
          }}>
            <strong style={{ color: '#34D399' }}>Confiance médiane 96,3 %</strong><br/>
            La géographie XGBoost dans UMAP est cohérente avec la structure non supervisée : supervisé et non supervisé capturent la <em>même physique</em>.
          </div>
        </div>
      </div>
    </div>
  );
}
