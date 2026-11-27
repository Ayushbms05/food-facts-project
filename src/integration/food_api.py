"""I8.7b: enforce ingredient quality triage before original-mode prediction."""
from contextlib import asynccontextmanager
import logging
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictBool
from fastapi.exceptions import RequestValidationError
from .nova_service import NovaService, InputError, MAX_CHARS, validate_text
from .ingredient_quality import assess_ingredients, RULE_VERSION

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
        app.state.nova = None
        try:
            app.state.nova = NovaService()
            log.info('Original NOVA model ready on %s', app.state.nova.device)
        except Exception as exc:
            # Keep verified barcode lookup available if the model cannot load.
            log.error('NOVA model unavailable: %s', type(exc).__name__)
        yield
    finally:
        app.state.nova = None
        pool.close()


app = FastAPI(title='Food Facts API', version='0.3.0', lifespan=lifespan)


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


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    ingredient_text: str = Field(max_length=MAX_CHARS)
    source: Literal['manual', 'ocr'] = 'manual'
    reviewed: StrictBool = False


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    # Avoid returning raw ingredient text in validation errors.
    return JSONResponse(status_code=422, content={'status': 'invalid_request'})


@app.post('/predict', responses={400: {'description': 'Invalid ingredient text or text needing correction'},
                                503: {'description': 'Model unavailable'}})
def predict(payload: PredictionInput, request: Request):
    try:
        quality = assess_ingredients(payload.ingredient_text, payload.source, payload.reviewed)
        if quality['decision'] != 'accept':
            messages = {
                'heading_without_ingredients': 'Enter the ingredients, not just the heading.',
                'unrecognized_short_fragment': 'Check the short ingredient text and correct or re-enter it.',
                'possible_non_ingredient_label_text': 'Enter only the ingredient list; remove unrelated label information.',
            }
            return JSONResponse(status_code=400, content={
                'status': 'needs_review' if quality['decision'] == 'review' else 'invalid_ingredients',
                'reason': quality['reason'],
                'message': quality.get('message') or messages[quality['reason']],
                'quality_rule_version': RULE_VERSION,
            })
        text = validate_text(payload.ingredient_text, payload.source, payload.reviewed)
        service = request.app.state.nova
        if service is None:
            return JSONResponse(status_code=503, content={'status': 'model_unavailable'})
        return service.predict(text, payload.source, payload.reviewed)
    except InputError as exc:
        return JSONResponse(status_code=400, content={'status': 'invalid_ingredients', 'message': str(exc)})
    except Exception as exc:
        log.error('NOVA prediction failed: %s', type(exc).__name__)
        return JSONResponse(status_code=503, content={'status': 'model_unavailable'})
