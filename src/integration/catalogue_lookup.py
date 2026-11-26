"""I8.4b: reusable, read-only product lookup, independent of Spark and FastAPI."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SQL = """
SELECT p.*, b.record_count AS conflict_count, to_jsonb(c) AS curated
FROM catalogue.off_products p
LEFT JOIN catalogue.barcode_conflicts b ON p.gtin14 = b.gtin14
LEFT JOIN catalogue.curated_links c ON p.gtin14 = c.gtin14
WHERE p.gtin14 = %s
ORDER BY p.record_id
"""


def normalize_barcode(value):
    """Return canonical GTIN-14 or None; never convert the whole code to int."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if len(value) not in (8, 12, 13, 14) or not re.fullmatch(r'[0-9]+', value):
        return None
    if not value.strip('0'):
        return None
    total = sum(int(c) * (3 if i % 2 == 0 else 1)
                for i, c in enumerate(reversed(value[:-1])))
    if (-total) % 10 != int(value[-1]):
        return None
    return value.zfill(14)


def connect_catalogue():
    """Local prototype connection. Read password from our generated ignored file."""
    import psycopg
    from psycopg.rows import dict_row
    lines = (ROOT / 'infra/.env').read_text(encoding='utf-8').splitlines()
    passwords = [line.split('=', 1)[1] for line in lines if line.startswith('POSTGRES_PASSWORD=')]
    if len(passwords) != 1 or not re.fullmatch(r'[0-9a-f]{48}', passwords[0]):
        raise RuntimeError('Expected one locally generated hex database password; inspect infra/.env privately.')
    return psycopg.connect(host='127.0.0.1', port=5432, dbname='foodfacts',
                           user='food_admin', password=passwords[0], connect_timeout=5,
                           autocommit=True, row_factory=dict_row,
                           options='-c default_transaction_read_only=on -c statement_timeout=10000')


def classify(key, rows):
    if not rows:
        return {'status': 'not_found', 'gtin14': key, 'matching_rows': 0}
    counts = {r['conflict_count'] for r in rows}
    if len(rows) > 1:
        if counts != {len(rows)}:
            raise RuntimeError('Catalogue conflict audit differs from actual matching rows')
        return {'status': 'ambiguous', 'gtin14': key, 'matching_rows': len(rows)}
    if counts != {None}:
        raise RuntimeError('Unexpected conflict metadata for a singleton')
    product = dict(rows[0])
    product.pop('conflict_count')
    curated = product.pop('curated')
    status = curated['link_status'] if curated else 'no_curated_link'
    # Only the checked linkage is exposed; even that is not verified formulation identity.
    comparison = curated if curated and status == 'consistent_checked_fields' and curated['record_count'] == 1 else None
    return {'status': 'found', 'gtin14': key, 'matching_rows': 1,
            'product': product, 'curated_link_status': status,
            'source_comparison': comparison}


def lookup(conn, barcode):
    key = normalize_barcode(barcode)
    if key is None:
        return {'status': 'invalid_barcode', 'gtin14': None, 'matching_rows': 0}
    # Bind the barcode as a parameter, not a piece of SQL code.
    rows = conn.execute(SQL, (key,)).fetchall()
    return classify(key, rows)


def self_test():
    assert normalize_barcode('036000291452') == '00036000291452'
    assert normalize_barcode('0036000291452') == '00036000291452'
    assert normalize_barcode(' 00036000291452 ') == '00036000291452'
    for bad in ('036000291453', '00000000', '１２３４５６７８', '', 'abc', '036000291452\n0', None, 36000291452):
        assert normalize_barcode(bad) is None
    key = '00036000291452'
    assert lookup(None, 'abc')['status'] == 'invalid_barcode'
    assert classify(key, [])['status'] == 'not_found'
    assert classify(key, [{'conflict_count':2}]*2)['status'] == 'ambiguous'
    row = {'gtin14':key, 'recorded_nova':None, 'conflict_count':None, 'curated':None}
    assert classify(key, [row])['product']['recorded_nova'] is None
    row['curated'] = {'link_status':'missing_english_comparison','record_count':1}
    assert classify(key,[row])['source_comparison'] is None
    row['curated']['link_status'] = 'consistent_checked_fields'
    assert classify(key,[row])['source_comparison'] is not None
    print('SUCCESS: barcode validation and lookup decision self-tests passed.')


if __name__ == '__main__':
    self_test()
