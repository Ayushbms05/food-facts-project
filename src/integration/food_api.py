"""I8.5b: local read-only HTTP API. No model inference or analytics events yet."""
from contextlib import asynccontextmanager
import logging

import psycopg
from psycopg_pool import ConnectionPool, PoolTimeout
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from .api_db_access import api_connection_kwargs
from .catalogue_lookup import lookup, normalize_barcode

log = logging.getLogger(__name__)
HTTP_CODES = {'found': 200, 'invalid_barcode': 400, 'not_found': 404, 'ambiguous': 409}


@asynccontextmanager
async def lifespan(app):
    pool = ConnectionPool(kwargs=api_connection_kwargs(), min_size=1, max_size=4,
                          timeout=5, open=False, name='food-api')
    try:
        pool.open(wait=True, timeout=15)
        app.state.pool = pool
        yield
    finally:
        pool.close()


app = FastAPI(title='Food Facts API', version='0.1.0', lifespan=lifespan)


def unavailable():
    return JSONResponse(status_code=503, content={'status': 'database_unavailable'})


@app.get('/health')
def health(request: Request):
    try:
        with request.app.state.pool.connection() as conn:
            row = conn.execute('SELECT record_id FROM catalogue.off_products LIMIT 1').fetchone()
            if row is None:
                return unavailable()
        return {'status': 'ok', 'database': 'ready'}
    except (psycopg.Error, PoolTimeout) as exc:
        log.warning('Health check failed: %s', type(exc).__name__)
        return unavailable()


@app.get('/products/{barcode}', responses={400:{'description':'Invalid barcode'},404:{'description':'Not found in local catalogue'},409:{'description':'Ambiguous barcode'},503:{'description':'Database unavailable'}})
def product(barcode: str, request: Request):
    if normalize_barcode(barcode) is None:
        result = lookup(None, barcode)
    else:
        try:
            with request.app.state.pool.connection() as conn:
                result = lookup(conn, barcode)
        except (psycopg.Error, PoolTimeout) as exc:
            log.warning('Lookup database failure: %s', type(exc).__name__)
            return unavailable()
        except RuntimeError:
            log.error('Catalogue consistency check failed')
            return JSONResponse(status_code=500, content={'status':'catalogue_inconsistent'})
    return JSONResponse(status_code=HTTP_CODES[result['status']], content=jsonable_encoder(result))
