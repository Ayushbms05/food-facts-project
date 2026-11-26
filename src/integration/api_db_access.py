"""I8.5a: provision and verify a restricted local API login (run from repo)."""
import os
import re
import secrets
import subprocess
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from .catalogue_lookup import lookup

ROOT = Path(__file__).resolve().parents[2]
SECRET = ROOT / 'infra/.env.api'
ROLE = 'food_api'
TABLES = ('off_products', 'curated_links', 'barcode_conflicts')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read_password(path, name):
    lines = path.read_text(encoding='utf-8').splitlines()
    values = [line.split('=', 1)[1] for line in lines if line.startswith(name + '=')]
    require(len(values) == 1 and re.fullmatch(r'[0-9a-f]{48}', values[0]),
            f'Expected one generated hex password in {path.name}; inspect privately.')
    return values[0]


def api_connection_kwargs():
    return dict(host='127.0.0.1', port=5432, dbname='foodfacts', user=ROLE,
                password=read_password(SECRET, 'API_DB_PASSWORD'),
                autocommit=True, row_factory=dict_row, connect_timeout=5,
                options='-c default_transaction_read_only=on -c statement_timeout=10000')


def verify():
    with psycopg.connect(**api_connection_kwargs()) as conn:
        role = conn.execute('SELECT current_user AS name, rolsuper, rolcreatedb, rolcreaterole, rolreplication, rolbypassrls FROM pg_roles WHERE rolname=current_user').fetchone()
        require(role['name'] == ROLE and not any(v for k,v in role.items() if k != 'name'), 'Unexpected role privileges')
        require(conn.execute('SHOW transaction_read_only').fetchone()['transaction_read_only'] == 'on', 'Expected read-only session')
        for table in TABLES:
            name = 'catalogue.' + table
            def has(priv):
                return conn.execute('SELECT has_table_privilege(current_user, %s, %s) AS allowed', (name, priv)).fetchone()['allowed']
            require(has('SELECT'), f'Missing SELECT on {name}')
            require(not any(has(p) for p in ('INSERT','UPDATE','DELETE','TRUNCATE','REFERENCES','TRIGGER')), f'Unexpected write/other privileges on {name}')
            print(f'{name}: SELECT allowed; write privileges denied')
        require(not conn.execute("SELECT has_schema_privilege(current_user, 'catalogue', 'CREATE') AS allowed").fetchone()['allowed'], 'Unexpected schema CREATE')
        require(lookup(conn, '00000105000011')['status'] == 'found', 'Known fixture lookup failed')
        require(lookup(conn, '00059527070430')['status'] == 'ambiguous', 'Ambiguous fixture lookup failed')
    print('SUCCESS: restricted role and actual catalogue lookups verified.')


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        verify()
        return
    require(subprocess.run(['git','check-ignore','-q','infra/.env.api'], cwd=ROOT).returncode == 0,
            'First ignore infra/.env.api in Git.')
    require(not subprocess.check_output(['git','ls-files','--','infra/.env.api'], cwd=ROOT).strip(), 'Secret path already tracked; stop.')
    require(not SECRET.exists(), 'Credential file already exists; use --verify-only or inspect before retry.')
    admin = read_password(ROOT / 'infra/.env', 'POSTGRES_PASSWORD')
    password = secrets.token_hex(24)
    with psycopg.connect(host='127.0.0.1', port=5432, dbname='foodfacts', user='food_admin', password=admin, connect_timeout=5) as conn:
        require(not conn.execute('SELECT 1 FROM pg_roles WHERE rolname=%s', (ROLE,)).fetchone(), 'Role already exists; no reset performed.')
        # Use a SCRAM verifier in DDL instead of sending plaintext in a SQL statement.
        verifier = conn.pgconn.encrypt_password(password.encode(), ROLE.encode(), b'scram-sha-256').decode()
        conn.execute(sql.SQL('CREATE ROLE food_api LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS CONNECTION LIMIT 8 PASSWORD {}').format(sql.Literal(verifier)))
        conn.execute('GRANT CONNECT ON DATABASE foodfacts TO food_api')
        conn.execute('GRANT USAGE ON SCHEMA catalogue TO food_api')
        conn.execute('GRANT SELECT ON catalogue.off_products, catalogue.curated_links, catalogue.barcode_conflicts TO food_api')
        conn.execute('ALTER ROLE food_api SET default_transaction_read_only = on')
        conn.execute("ALTER ROLE food_api SET statement_timeout = '10s'")
        # Create the local file exclusively, with owner-only permissions; no password printed.
        with os.fdopen(os.open(SECRET, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as f:
            f.write('API_DB_PASSWORD=' + password + '\n')
        # Context manager commits DDL when successful. Keep the file if commit outcome is uncertain.
    verify()


if __name__ == '__main__':
    main()
