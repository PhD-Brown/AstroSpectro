import React from 'react';

export default function PCAVarianceMilestones() {
  // Fusion des données de tes deux tableaux
  const milestones = [
    {
      k: 2,
      variance: 28.8, // %
      mse: '0,696',
      desc: 'PC1 + PC2 uniquement — visualisation basique 2D.',
      color: '#64748B', // Gris ardoise
      highlight: false
    },
    {
      k: 10,
      variance: 50.6,
      mse: '0,494',
      desc: 'Capture de la moitié de l\'information physique.',
      color: '#8B5CF6', // Violet
      highlight: false
    },
    {
      k: 51,
      variance: 80.0,
      mse: '0,196',
      desc: 'Seuil d\'analyse rapide — bon compromis vitesse/précision.',
      color: '#F59E0B', // Ambre
      highlight: false
    },
    {
      k: 91,
      variance: 95.0,
      mse: '~0,050',
      desc: 'SEUIL RETENU POUR UMAP/t-SNE — Capture l\'essentiel de la variance.',
      color: '#38BDF8', // Bleu AstroSpectro
      highlight: true
    },
    {
      k: 100,
      variance: 99.0,
      mse: '<0,010',
      desc: 'Quasi-totalité de la variance (inclut le bruit résiduel).',
      color: '#10B981', // Vert
      highlight: false
    }
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', margin: '24px 0' }}>
      {milestones.map((item, index) => (
        <div 
          key={item.k}
          style={{
            display: 'flex',
            alignItems: 'center',
            background: item.highlight ? 'rgba(56, 189, 248, 0.05)' : 'var(--ifm-background-surface-color, #0a0c14)',
            border: `1px solid ${item.highlight ? item.color : 'rgba(255, 255, 255, 0.08)'}`,
            borderRadius: '8px',
            padding: '16px 20px',
            boxShadow: item.highlight ? `0 0 20px ${item.color}20` : 'none',
            position: 'relative',
            overflow: 'hidden',
            flexWrap: 'wrap',
            gap: '16px'
          }}
        >
          {/* Effet de lueur sur la carte highlight */}
          {item.highlight && (
            <div style={{
              position: 'absolute', left: 0, top: 0, bottom: 0, width: '4px',
              background: item.color, boxShadow: `0 0 10px ${item.color}`
            }}></div>
          )}

          {/* Bloc K composantes */}
          <div style={{ minWidth: '70px', textAlign: 'center' }}>
            <div style={{ fontSize: '11px', textTransform: 'uppercase', opacity: 0.5, fontWeight: 'bold' }}>K =</div>
            <div style={{ fontSize: '24px', fontWeight: '900', color: item.highlight ? item.color : 'var(--ifm-font-color-base)' }}>
              {item.k}
            </div>
          </div>

          {/* Bloc Barre de Variance & MSE */}
          <div style={{ flex: '1 1 200px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', fontWeight: 'bold' }}>
              <span>Variance : {item.variance}%</span>
              <span style={{ opacity: 0.7, fontWeight: 'normal', fontFamily: 'monospace' }}>MSE: {item.mse}</span>
            </div>
            {/* Barre de progression */}
            <div style={{ 
              width: '100%', height: '8px', background: 'rgba(255, 255, 255, 0.1)', 
              borderRadius: '4px', overflow: 'hidden' 
            }}>
              <div style={{ 
                width: `${item.variance}%`, height: '100%', background: item.color,
                transition: 'width 1s ease-in-out'
              }}></div>
            </div>
          </div>

          {/* Bloc Description */}
          <div style={{ 
            flex: '2 1 250px', 
            fontSize: '13.5px', 
            color: item.highlight ? '#fff' : 'var(--ifm-font-color-base)',
            opacity: item.highlight ? 1 : 0.8,
            fontWeight: item.highlight ? '600' : '400',
            borderLeft: '1px solid rgba(255, 255, 255, 0.1)',
            paddingLeft: '16px'
          }}>
            {item.desc}
          </div>
        </div>
      ))}
    </div>
  );
}