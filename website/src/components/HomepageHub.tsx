import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faRocket, faBook, faVial, faUsers } from '@fortawesome/free-solid-svg-icons';

// 1. On définit les données pour nos cartes dans un tableau
const cardData = [
  {
    title: "Getting Started",
    icon: faRocket,
    description: "Suivez un tutoriel complet pour installer le projet et lancer votre première classification en moins de 10 minutes.",
    link: "/AstroSpectro/docs/getting-started",
    buttonText: "Commencer le Tutoriel"
  },
  {
    title: "User Guides",
    icon: faBook,
    description: "Des tutoriels détaillés pour chaque étape clé : téléchargement, traitement, entraînement, et visualisation.",
    link: "/AstroSpectro/docs/user-guides",
    buttonText: "Consulter les Guides"
  },
  {
    title: "Scientific Context",
    icon: faVial,
    description: "Plongez dans le contexte astrophysique : catalogue LAMOST, types spectraux, et méthodologie.",
    link: "/AstroSpectro/docs/science",
    buttonText: "Explorer le Contexte"
  },
  {
    title: "Community & Contributing",
    icon: faUsers,
    description: "Vous souhaitez contribuer, citer le projet ou poser une question ? Toutes les ressources sont ici.",
    link: "/AstroSpectro/docs/community",
    buttonText: "Rejoindre la Communauté"
  }
];

// 2. On crée un sous-composant pour une seule carte
const HubCard = ({ title, icon, description, link, buttonText }) => (
  <div className="col col--6 margin-bottom--lg">
    <div className="card hub-card">
      <div className="card__header">
        <h3><FontAwesomeIcon icon={icon} /> {title}</h3>
      </div>
      <div className="card__body">
        <p>{description}</p>
      </div>
      <div className="card__footer">
        <a href={link} className="button button--secondary button--block">
          {buttonText}
        </a>
      </div>
    </div>
  </div>
);

// 3. On crée le composant principal qui va générer la grille de cartes
export default function HomepageHub() {
  return (
    <div className="container">
      <div className="row justify-content--center">
        {cardData.map((card, idx) => (
          <HubCard key={idx} {...card} />
        ))}
      </div>
    </div>
  );
}