# Security policy

dotFly is a simulation library and demo; it opens local files and runs no network services.
The one class of input that deserves care is a `.dfb` checkpoint from an untrusted source: the
reader validates section bounds, but treat checkpoints like any other binary file you did not build.

To report a vulnerability, use GitHub's private vulnerability reporting on this repository
(Security → Report a vulnerability) rather than a public issue. You will get an acknowledgement
within a week.
