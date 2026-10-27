# Data provenance and usage record

This study uses the archived Oxford-Man Realized Library CSV hosted by the academic highfrequency project.

- Revision: d6eafaefee68e4bb191e1d834ae67c2dd353614c
- SHA-256: 42f46a5f569bb30e1bfc8922e0ee857c4152ff1c6e95dbe08b55cf26319dcb99
- Instrument: .SPX, S&P 500 cash index, not SPY ETF.
- Measure: rv5, realised variance based on five-minute intraday returns; squared decimal-return units, not annualised.
- Instrument coverage: 5,052 observations, 2000-01-03 through 2020-02-21.
- Fourteen missing XNYS sessions; zero duplicate dates, unexpected sessions or invalid variance values.
- Daily civil date labels are preserved without UTC conversion of midnight timestamps.
- Source verification metadata is written to data/raw/source_manifest.json. Initial retrieval time was not captured; the manifest explicitly labels its timestamp as verification time.

The source repository's data preparation script attributes its original download to Oxford-Man and explicitly selects .SPX:
https://github.com/jonathancornelissen/highfrequency/blob/master/data-raw/realized_library_data.R

Pinned archive:
https://github.com/jonathancornelissen/highfrequency/blob/d6eafaefee68e4bb191e1d834ae67c2dd353614c/data-raw/oxfordmanrealizedvolatilityindices.csv

Original provider withdrawal notice:
https://oxford-man.ox.ac.uk/research/realized-library/

The highfrequency software declares GPL >= 2, but that does not independently establish third-party data redistribution terms. A separate data-specific licence has not been verified. The raw archive and row-level outcome data are excluded from this repository. The project supplies acquisition code and attribution, and keeps raw files and detailed predictions local. Any future raw-data release would require a separate review of redistribution terms.

An historical snapshot is not a point-in-time feed. Vendor publication delays and retrospective corrections cannot be reconstructed. The archive ends before the main COVID shock; no COVID performance claim is possible. Holdout values are used only for integrity/coverage auditing, not model selection or performance analysis.
