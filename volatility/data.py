"""Pinned acquisition and a conservative session-calendar audit."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import exchange_calendars as xc
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / 'configs/development.json').read_text(encoding='utf-8-sig'))


def acquire(cfg):
    path = ROOT / 'data/raw/oxfordman.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    url = ('https://raw.githubusercontent.com/jonathancornelissen/highfrequency/'
           + cfg['source_revision'] + '/data-raw/oxfordmanrealizedvolatilityindices.csv')
    if not path.exists():
        with urlopen(url, timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != cfg['source_sha256']:
            raise ValueError('Downloaded archive checksum differs from pinned source')
        path.write_bytes(content)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    if checksum != cfg['source_sha256']:
        raise ValueError('Local archive checksum differs from pinned source; refusing to overwrite')
    manifest = path.parent / 'source_manifest.json'
    if not manifest.exists():
        manifest.write_text(json.dumps({
            'url': url, 'revision': cfg['source_revision'], 'sha256': checksum,
            'verified_at_utc': datetime.now(timezone.utc).isoformat(),
            'note': 'Verification timestamp; initial retrieval timestamp was not captured.',
            'source': 'Oxford-Man Realized Library archived by highfrequency',
            'raw_redistribution': 'Excluded from repository'
        }, indent=2), encoding='utf-8')
    return path


def session_dates(labels):
    """Preserve civil date labels; these are not UTC event timestamps."""
    text = labels.astype(str)
    if not text.str.match(r'^\d{4}-\d{2}-\d{2} 00:00:00[+-]\d{2}:\d{2}$').all():
        raise ValueError('Unexpected daily date-label format')
    return pd.to_datetime(text.str.slice(0, 10), format='%Y-%m-%d')

def audit_and_load(cfg):
    path = acquire(cfg)
    raw = pd.read_csv(path)
    selected = raw.loc[raw.Symbol == cfg['symbol']].copy()
    selected['date'] = session_dates(selected.iloc[:, 0])
    if selected.empty or selected.date.duplicated().any():
        raise ValueError('Empty instrument selection or duplicate dates')
    selected = selected.set_index('date').sort_index()
    calendar = xc.get_calendar(cfg['calendar'], start='1999-01-01', end='2021-01-01')
    expected = calendar.sessions_in_range(selected.index.min(), selected.index.max()).tz_localize(None)
    missing = expected.difference(selected.index)
    unexpected = selected.index.difference(expected)
    values = pd.to_numeric(selected[cfg['measure']], errors='raise')
    invalid = (~np.isfinite(values)) | (values <= 0)
    if invalid.any():
        raise ValueError('Invalid variance observations need explicit investigation')
    # Preserve absent sessions as NaN; never compress or forward-fill the calendar.
    aligned = values.reindex(expected)
    development = aligned.loc[:cfg['development_end']]
    audit = {
        'symbol': cfg['symbol'], 'measure': cfg['measure'],
        'source_sha256': cfg['source_sha256'], 'rows': len(selected),
        'first_date': str(selected.index.min().date()), 'last_date': str(selected.index.max().date()),
        'duplicate_dates': 0, 'invalid_variances': int(invalid.sum()),
        'calendar': cfg['calendar'], 'calendar_version': xc.__version__,
        'missing_sessions': missing.strftime('%Y-%m-%d').tolist(),
        'unexpected_sessions': unexpected.strftime('%Y-%m-%d').tolist(),
        'policy': 'Reindex expected sessions; drop examples with incomplete feature/target windows; no imputation.',
        'holdout_policy': 'Coverage/integrity metadata only; no holdout forecasts or performance inspected.',
        'development_observed_rows': int(development.notna().sum()),
        'limitation': 'Historical archive, not a point-in-time publication feed; no COVID evaluation.'
    }
    if len(unexpected):
        raise ValueError('Unexpected sessions require investigation: ' + str(audit))
    report_dir = ROOT / 'reports'
    report_dir.mkdir(exist_ok=True)
    (report_dir / 'data_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return development, audit


def make_frame(series, decay=0.94):
    """Build one-session labels on the complete calendar, preserving gaps."""
    if not series.index.is_monotonic_increasing or series.index.has_duplicates:
        raise ValueError('Session index must be strictly increasing')
    frame = pd.DataFrame({'daily': series, 'weekly': series.rolling(5).mean(),
                          'monthly': series.rolling(22).mean(), 'target': series.shift(-1)})
    frame['target_end'] = pd.Series(series.index, index=series.index).shift(-1)
    ewma, state = [], None
    for value in series:
        if pd.isna(value):
            state = None
            ewma.append(np.nan)
        else:
            state = float(value) if state is None else decay * state + (1 - decay) * value
            ewma.append(state)
    frame['ewma'] = ewma
    frame.index.name = 'origin'
    return frame.dropna()


if __name__ == '__main__':
    _, audit = audit_and_load(config())
    print(json.dumps(audit, indent=2))

