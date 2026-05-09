/* ── Counter animation for home page stats ── */
document.addEventListener('DOMContentLoaded', () => {

  // Animate counting numbers
  const counters = document.querySelectorAll('.stat-number, .metric-value');
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        animateCounter(entry.target);
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.3 });

  counters.forEach(el => observer.observe(el));

  function animateCounter(el) {
    const target = parseFloat(el.dataset.target || el.textContent);
    if (isNaN(target)) return;
    const isDecimal = target % 1 !== 0;
    const duration = 1400;
    const start = performance.now();
    const update = (now) => {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const val = eased * target;
      el.textContent = isDecimal ? val.toFixed(2) : Math.floor(val);
      if (progress < 1) requestAnimationFrame(update);
      else el.textContent = isDecimal ? target.toFixed(2) : target;
    };
    requestAnimationFrame(update);
  }

  // Animate metric fill bars
  const fills = document.querySelectorAll('.metric-fill');
  const fillObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const w = entry.target.dataset.width;
        setTimeout(() => { entry.target.style.width = w + '%'; }, 200);
        fillObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.3 });
  fills.forEach(el => fillObserver.observe(el));

  // Navbar scroll effect
  const nav = document.getElementById('mainNav');
  if (nav) {
    window.addEventListener('scroll', () => {
      nav.style.boxShadow = window.scrollY > 30 ? '0 4px 30px rgba(0,0,0,0.5)' : '';
    });
  }

  // Tooltip init (Bootstrap)
  document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => {
    new bootstrap.Tooltip(el);
  });
});
