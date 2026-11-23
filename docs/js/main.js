document.documentElement.classList.add('js-enabled');
const revealItems = document.querySelectorAll('.reveal');

if ('IntersectionObserver' in window) {

  const revealObserver = new IntersectionObserver(
    (entries, observer) => {

      entries.forEach(entry => {

        if (entry.isIntersecting) {
          entry.target.classList.add('visible');

          observer.unobserve(entry.target);
        }

      });

    },
    {
      threshold: 0.12
    }
  );

  revealItems.forEach(item => {
    revealObserver.observe(item);
  });

} else {

  revealItems.forEach(item => {
    item.classList.add('visible');
  });

}


const sections = document.querySelectorAll('main section[id]');
const navLinks = document.querySelectorAll('.nav-links a');

if ('IntersectionObserver' in window) {

  const sectionObserver = new IntersectionObserver(
    entries => {

      entries.forEach(entry => {

        if (entry.isIntersecting) {

          navLinks.forEach(link => {

            link.classList.toggle(
              'active',
              link.getAttribute('href') === `#${entry.target.id}`
            );

          });

        }

      });

    },
    {
      rootMargin: '-35% 0px -55% 0px'
    }
  );

  sections.forEach(section => {
    sectionObserver.observe(section);
  });

}
