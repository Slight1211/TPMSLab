# Before public release

The local package is a release candidate. These are publication preparation
steps, not automated uploads.

- Finalize author names, affiliations, contact, contributors and copyright.
- Confirm the distribution name on PyPI and repository ownership; a 404 is not
  a reservation. If renamed, update pyproject metadata, documentation and CLI.
- Review LattGen formula provenance and the supplied user-code contributions;
  retain all applicable notices. Complete human review of AI-assisted code.
- Create a public version-controlled repository and run the configured CI on
  every advertised OS/Python combination. Current local evidence covers only
  Windows and Python 3.12.
- Build from a clean checkout, run tests, inspect both wheel and source archive,
  install the wheel into an empty environment, run the public API and CLI, and
  run the optional web checks. Use twine check before an upload.
- Test a TestPyPI release, then publish the approved release with the account
  owner's credentials. This folder contains no automatic publish workflow.
- Archive the actual released source and benchmark data with a persistent DOI.
  Complete CITATION.cff only after author and archive metadata are known.
- For Software Impacts, finalize the official article template, permanent code
  and reproducible-capsule links, support address, declarations and the list of
  research publications enabled by this software. Check current author guidance.
- Add a documented application and mesh-convergence study before claiming
  validated mechanical predictions. Adoption and user impact require evidence.

The core example can be run without proprietary software in a reproducibility
capsule. Licensed COMSOL verification must be documented separately; it should
not be represented as executable on a generic public capsule without a licence.
