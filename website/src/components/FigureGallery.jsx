import React from 'react';
import { useBaseUrlUtils } from '@docusaurus/useBaseUrl';

/**
 * FigureGallery — image gallery with lightbox for reference figures
 *
 * Props:
 *   figures : array of { src, title, desc, tag }
 *   columns : number of grid columns (default 3)
 */
export default function FigureGallery({ figures, columns = 3 }) {
  const [lightbox, setLightbox] = React.useState(null); // index of open image
  const [loaded, setLoaded] = React.useState({});
  const { withBaseUrl } = useBaseUrlUtils();

  function open(i)  { setLightbox(i); }
  function close()  { setLightbox(null); }
  function prev()   { setLightbox((l) => (l - 1 + figures.length) % figures.length); }
  function next()   { setLightbox((l) => (l + 1) % figures.length); }

  React.useEffect(() => {
    function onKey(e) {
      if (lightbox === null) return;
      if (e.key === 'Escape')      close();
      if (e.key === 'ArrowLeft')   prev();
      if (e.key === 'ArrowRight')  next();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [lightbox]);

  // Prevent body scroll when lightbox open
  React.useEffect(() => {
    if (lightbox !== null) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => { document.body.style.overflow = ''; };
  }, [lightbox]);

  const active = lightbox !== null ? figures[lightbox] : null;

  return (
    <>
      {/* Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: `repeat(${columns}, 1fr)`,
        gap: '12px',
        margin: '20px 0',
      }}>
        {figures.map((fig, i) => (
          <div
            key={i}
            onClick={() => open(i)}
            style={{
              background: '#0a0c14',
              border: '1px solid rgba(255,255,255,0.07)',
              borderRadius: '9px',
              overflow: 'hidden',
              cursor: 'pointer',
              transition: 'transform 0.15s, border-color 0.15s, box-shadow 0.15s',
              position: 'relative',
              display: 'flex',
              flexDirection: 'column',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-2px)';
              e.currentTarget.style.borderColor = 'rgba(56,189,248,0.35)';
              e.currentTarget.style.boxShadow = '0 6px 20px rgba(56,189,248,0.1)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'none';
              e.currentTarget.style.borderColor = 'rgba(255,255,255,0.07)';
              e.currentTarget.style.boxShadow = 'none';
            }}
          >
            {/* Image */}
            <div style={{
              background: '#06080f',
              aspectRatio: '4/3',
              overflow: 'hidden',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              position: 'relative',
            }}>
              {!loaded[i] && (
                <div style={{ fontSize: '11px', opacity: 0.25, color: 'var(--ifm-font-color-base)', fontFamily: 'monospace' }}>
                  ◎
                </div>
              )}
              <img
                src={withBaseUrl(fig.src)}
                alt={fig.title}
                loading="lazy"
                onLoad={() => setLoaded((p) => ({ ...p, [i]: true }))}
                style={{
                  width: '100%',
                  height: '100%',
                  objectFit: 'contain',
                  padding: '6px',
                  opacity: loaded[i] ? 1 : 0,
                  transition: 'opacity 0.3s',
                  display: 'block',
                }}
              />

              {/* Zoom hint on hover */}
              <div style={{
                position: 'absolute',
                top: '8px',
                right: '8px',
                background: 'rgba(0,0,0,0.6)',
                borderRadius: '4px',
                padding: '2px 6px',
                fontSize: '10px',
                color: '#38BDF8',
                fontFamily: 'monospace',
                opacity: 0,
                transition: 'opacity 0.15s',
                pointerEvents: 'none',
              }} className="zoom-hint">⤢</div>
            </div>

            {/* Caption */}
            <div style={{ padding: '10px 12px', flex: 1 }}>
              {fig.tag && (
                <div style={{
                  display: 'inline-block',
                  background: 'rgba(56,189,248,0.12)',
                  border: '1px solid rgba(56,189,248,0.25)',
                  borderRadius: '4px',
                  padding: '1px 7px',
                  fontSize: '9.5px',
                  fontWeight: '700',
                  color: '#38BDF8',
                  fontFamily: 'monospace',
                  marginBottom: '5px',
                }}>{fig.tag}</div>
              )}
              <div style={{
                fontSize: '12px',
                fontWeight: '600',
                color: 'var(--ifm-font-color-base)',
                marginBottom: '3px',
                lineHeight: 1.3,
              }}>{fig.title}</div>
              {fig.desc && (
                <div style={{
                  fontSize: '10.5px',
                  color: 'var(--ifm-font-color-base)',
                  opacity: 0.5,
                  lineHeight: 1.4,
                }}>{fig.desc}</div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Lightbox */}
      {lightbox !== null && (
        <div
          onClick={close}
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.88)',
            zIndex: 9999,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backdropFilter: 'blur(6px)',
          }}
        >
          {/* Content box */}
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              background: '#0d1117',
              border: '1px solid rgba(255,255,255,0.12)',
              borderRadius: '12px',
              maxWidth: '90vw',
              maxHeight: '90vh',
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
              boxShadow: '0 24px 64px rgba(0,0,0,0.6)',
            }}
          >
            {/* Lightbox header */}
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '12px 16px',
              borderBottom: '1px solid rgba(255,255,255,0.07)',
            }}>
              <div>
                <div style={{ fontWeight: '700', fontSize: '14px', color: 'var(--ifm-font-color-base)' }}>
                  {active.title}
                </div>
                {active.desc && (
                  <div style={{ fontSize: '11px', opacity: 0.5, marginTop: '2px', color: 'var(--ifm-font-color-base)' }}>
                    {active.desc}
                  </div>
                )}
              </div>
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <span style={{ fontSize: '11px', opacity: 0.35, fontFamily: 'monospace', color: 'var(--ifm-font-color-base)' }}>
                  {lightbox + 1} / {figures.length}
                </span>
                <button
                  onClick={close}
                  style={{
                    background: 'rgba(255,255,255,0.08)',
                    border: '1px solid rgba(255,255,255,0.12)',
                    borderRadius: '6px',
                    color: 'var(--ifm-font-color-base)',
                    cursor: 'pointer',
                    padding: '4px 10px',
                    fontSize: '14px',
                  }}
                >✕</button>
              </div>
            </div>

            {/* Image */}
            <div style={{
              flex: 1,
              overflow: 'auto',
              padding: '16px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: '#06080f',
              minHeight: '300px',
            }}>
              <img
                src={withBaseUrl(active.src)}
                alt={active.title}
                style={{
                  maxWidth: '100%',
                  maxHeight: '70vh',
                  objectFit: 'contain',
                  display: 'block',
                }}
              />
            </div>

            {/* Lightbox nav */}
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '10px 16px',
              borderTop: '1px solid rgba(255,255,255,0.07)',
            }}>
              <button
                onClick={prev}
                style={{
                  background: 'rgba(255,255,255,0.06)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '6px',
                  color: 'var(--ifm-font-color-base)',
                  cursor: 'pointer',
                  padding: '6px 16px',
                  fontSize: '13px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                }}
              >← Préc.</button>

              {/* Thumbnail strip */}
              <div style={{
                display: 'flex',
                gap: '4px',
                overflowX: 'auto',
                maxWidth: '400px',
                padding: '2px',
              }}>
                {figures.map((f, i) => (
                  <div
                    key={i}
                    onClick={() => setLightbox(i)}
                    style={{
                      width: '36px',
                      height: '36px',
                      borderRadius: '4px',
                      overflow: 'hidden',
                      flexShrink: 0,
                      border: `2px solid ${i === lightbox ? '#38BDF8' : 'rgba(255,255,255,0.1)'}`,
                      cursor: 'pointer',
                      background: '#06080f',
                      transition: 'border-color 0.1s',
                    }}
                  >
                    <img src={withBaseUrl(f.src)} alt="" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                  </div>
                ))}
              </div>

              <button
                onClick={next}
                style={{
                  background: 'rgba(255,255,255,0.06)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '6px',
                  color: 'var(--ifm-font-color-base)',
                  cursor: 'pointer',
                  padding: '6px 16px',
                  fontSize: '13px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                }}
              >Suiv. →</button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
