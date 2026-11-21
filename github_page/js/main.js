
// scroll reveal
document.body.classList.add("js-enabled");

const revealElements = document.querySelectorAll(
    ".section, .algorithm-card"
);

revealElements.forEach((element) => {
    element.classList.add("reveal");
});

const observer = new IntersectionObserver(
    (entries, observer) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                entry.target.classList.add("visible");
                observer.unobserve(entry.target);
            }
        });
    },
    {
        threshold: 0.12
    }
);

revealElements.forEach((element) => {
    observer.observe(element);
});


// active navigation
const sections = document.querySelectorAll(
    "#overview, #algorithms, #methodology, #results"
);

const navLinks = document.querySelectorAll(".nav-links a");

const navObserver = new IntersectionObserver(
    (entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {

                navLinks.forEach((link) => {
                    link.classList.remove("active");
                });

                const activeLink = document.querySelector(
                    `.nav-links a[href="#${entry.target.id}"]`
                );

                if (activeLink) {
                    activeLink.classList.add("active");
                }
            }
        });
    },
    {
        threshold: 0.4
    }
);

sections.forEach((section) => {
    navObserver.observe(section);
});
