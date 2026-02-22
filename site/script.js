// ===================================
// Navigation
// ===================================
const nav = document.getElementById('nav');
const navToggle = document.getElementById('navToggle');
const navLinks = document.getElementById('navLinks');

window.addEventListener('scroll', () => {
    nav.classList.toggle('scrolled', window.scrollY > 10);
});

navToggle.addEventListener('click', () => {
    navLinks.classList.toggle('open');
});

navLinks.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => navLinks.classList.remove('open'));
});

// ===================================
// Active nav link highlighting
// ===================================
const sections = document.querySelectorAll('section[id]');
const navAnchors = document.querySelectorAll('.nav-links a:not(.nav-cta)');

function updateActiveLink() {
    let current = '';
    sections.forEach(section => {
        const top = section.offsetTop - 120;
        if (window.scrollY >= top) {
            current = section.getAttribute('id');
        }
    });
    navAnchors.forEach(a => {
        a.classList.toggle('active', a.getAttribute('href') === '#' + current);
    });
}

window.addEventListener('scroll', updateActiveLink);
updateActiveLink();

// ===================================
// Scroll-triggered fade-in animations
// ===================================
const observerOpts = { root: null, rootMargin: '0px 0px -60px 0px', threshold: 0.1 };

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
            observer.unobserve(entry.target);
        }
    });
}, observerOpts);

// Individual elements
document.querySelectorAll(
    '.hero-left, .hero-right, .section-tag, .section-title, .section-desc, ' +
    '.intro-grid, .feature-card, .data-card, .pipeline-step, .result-card, ' +
    '.team-card, .accordion, .conclusion-block, .stats-banner, .mentors-row'
).forEach(el => {
    el.classList.add('fade-up');
    observer.observe(el);
});

// Stagger groups
document.querySelectorAll('.feature-cards, .data-grid, .results-grid, .team-grid, .stats-row').forEach(group => {
    group.classList.add('stagger');
});

// ===================================
// Animate hero score bar on load
// ===================================
window.addEventListener('load', () => {
    const bar = document.querySelector('.hero-card-bar-fill');
    if (bar) {
        bar.style.width = '0%';
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                bar.style.width = '75.4%';
            });
        });
    }
});
