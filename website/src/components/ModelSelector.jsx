import React from 'react';

const MODELS = [
  {
    key: '"XGBoost"',
    algo: 'XGBClassifier',
    desc: 'Gradient boosting sur arbres — champion du pipeline AstroSpectro',
    color: '#34D399',
    badge: '87 % BA',
    champion: true,
    tag: 'Champion',
  },
  {
    key: '"LightGBM"',
    algo: 'LGBMClassifier',
    desc: 'Gradient boosting leaf-wise — très rapide sur grands jeux de données',
    color: '#38BDF8',
    badge: null,
    tag: 'Rapide',
  },
  {
    key: '"CatBoost"',
    algo: 'CatBoostClassifier',
    desc: 'Gradient boosting ordonné — gestion native des valeurs manquantes et catégorielles',
    color: '#F59E0B',
    badge: null,
    tag: 'Robust',
  },
  {
    key: '"ExtraTrees"',
    algo: 'ExtraTreesClassifier',
    desc: 'Arbres extrêmement randomisés — très rapide, bonne diversification',
    color: '#FBBF24',
    badge: null,
    tag: 'Rapide',
  },
  {
    key: '"RandomForest"',
    algo: 'RandomForestClassifier',
    desc: 'Forêts aléatoires — référence historique du projet (baseline)',
    color: '#94A3B8',
    badge: null,
    tag: 'Baseline',
  },
  {
    key: '"Ensemble"',
    algo: 'VotingClassifier',
    desc: 'Vote souple XGB + LightGBM + ExtraTrees — meilleure généralisation',
    color: '#C084FC',
    badge: null,
    tag: 'Combiné',
  },
  {
    key: '"SVM"',
    algo: 'SVC (calibré)',
    desc: 'Support Vector Machine calibré — bon sur petits datasets, lent à grande échelle',
    color: '#FB923C',
    badge: null,
    tag: 'Classique',
  },
  {
    key: '"MLP"',
    algo: 'MLPClassifier',
    desc: 'Réseau dense multicouche — alternative neuronale légère',
    color: '#F87171',
    badge: null,
    tag: 'Neural',
  },
  {
    key: '"LogisticRegression"',
    algo: 'LR multiclasse',
    desc: 'Régression logistique OvR — baseline linéaire interprétable',
    color: '#6EE7B7',
    badge: null,
    tag: 'Linéaire',
  },
  {
    key: '"KNN"',
    algo: 'KNeighborsClassifier',
    desc: 'K plus proches voisins — utile pour l\'analyse de structure spectrale',
    color: '#67E8F9',
    badge: null,
    tag: 'Géométrique',
  },
  {
    key: '"LDA"',
    algo: 'LinearDiscriminantAnalysis',
    desc: 'Analyse discriminante linéaire — référence classique projective',
    color: '#A3E635',
    badge: null,
    tag: 'Classique',
  },
];

function ModelCard({ modelKey, algo, desc, color, badge, champion, tag }) {
  const [hov, setHov] = React.useState(false);

  return (
    <div
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        background: hov ? `${color}0e` : champion ? `${color}05` : 'var(--ifm-card-background-color)',
        border: `1px solid ${hov ? color + '55' : champion ? color + '35' : 'rgba(255,255,255,0.07)'}`,
        borderLeft: `3px solid ${color}`,
        borderRadius: '8px',
        padding: '12px 14px',
        transition: 'all 0.15s',
        boxShadow: champion ? `0 2px 12px ${color}18` : 'none',
        cursor: 'default',
        display: 'flex',
        flexDirection: 'column',
        gap: '6px',
        position: 'relative',
      }}
    >
      {/* Top row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '6px' }}>
        <code style={{
          fontSize: '12px',
          fontWeight: champion ? '800' : '600',
          color: color,
          background: `${color}14`,
          padding: '2px 7px',
          borderRadius: '4px',
          lineHeight: 1.4,
        }}>{modelKey}</code>

        <div style={{ display: 'flex', gap: '5px', flexShrink: 0 }}>
          {badge && (
            <div style={{
              background: '#34D39918',
              border: '1px solid #34D39944',
              borderRadius: '12px',
              padding: '1px 8px',
              fontSize: '10px',
              fontWeight: '800',
              color: '#34D399',
              fontFamily: 'monospace',
            }}>{badge}</div>
          )}
          <div style={{
            background: 'rgba(255,255,255,0.06)',
            borderRadius: '4px',
            padding: '1px 6px',
            fontSize: '9.5px',
            color: 'var(--ifm-font-color-base)',
            opacity: 0.5,
          }}>{tag}</div>
        </div>
      </div>

      {/* Algo */}
      <div style={{
        fontSize: '11px',
        fontFamily: 'monospace',
        color: 'var(--ifm-font-color-base)',
        opacity: 0.5,
      }}>{algo}</div>

      {/* Desc */}
      <div style={{
        fontSize: '11.5px',
        color: 'var(--ifm-font-color-base)',
        opacity: hov ? 0.85 : 0.6,
        lineHeight: 1.4,
        transition: 'opacity 0.12s',
      }}>{desc}</div>
    </div>
  );
}

export default function ModelSelector() {
  return (
    <div style={{ margin: '24px 0' }}>
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
        gap: '10px',
      }}>
        {MODELS.map((m) => (
          <ModelCard key={m.key} modelKey={m.key} {...m} />
        ))}
      </div>

      <div style={{
        marginTop: '12px',
        padding: '10px 14px',
        background: 'rgba(52,211,153,0.06)',
        border: '1px solid rgba(52,211,153,0.2)',
        borderRadius: '7px',
        fontSize: '12px',
        color: 'var(--ifm-font-color-base)',
        opacity: 0.8,
      }}>
        <strong style={{ color: '#34D399' }}>Changer de modèle est trivial :</strong>
        {' '}<code>SpectralClassifier(model_type="LightGBM")</code> — le pipeline, le tuning et le logging W&B s'adaptent automatiquement.
      </div>
    </div>
  );
}
