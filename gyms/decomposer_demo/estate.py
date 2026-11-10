"""Demo-compatible tools over fictional Python records, without SQLite or live APIs."""
from copy import deepcopy
from typing import Annotated, Literal
from pydantic import Field
import hashlib
from itertools import zip_longest
import json
import re

from .tasks import task_spec


CITY_ALIASES = {'москва': 'moscow', 'moscow': 'moscow', 'санкт-петербург': 'saint-petersburg',
                'санкт петербург': 'saint-petersburg', 'спб': 'saint-petersburg',
                'saint-petersburg': 'saint-petersburg', 'saint petersburg': 'saint-petersburg',
                'st petersburg': 'saint-petersburg', 'st. petersburg': 'saint-petersburg'}
CAPTURED_AT = '2026-10-30T00:00:00+00:00'


def snapshot():
    rows = []
    for city in ('moscow', 'saint-petersburg'):
        for deal in ('rent', 'sale'):
            for i in range(24):
                id = str(10000001 + len(rows))
                row = dict(id=id, city=city, deal_type=deal,
                    url=f'https://listings.example.invalid/{id}',
                    address=f'Fictional {city} street {i+1}',
                    price=(60000+5000*i) if deal=='rent' else (300000000+5000000*i),
                    rooms=1+i%4, area=45+7*i, furnished=i%3!=0, desks=i%4,
                    lease_months=None if i%5==0 else 12, central=i%3!=0,
                    condition='finished and habitable', synthetic=True)
                if deal == 'rent':
                    row['price_period'] = 'month'
                rows.append(row)
    return rows


def matches(row, filters):
    return (row['city']==filters['city'] and row['deal_type']==filters['deal']
            and filters['min_price']<=row['price']<=filters['max_price']
            and row['rooms']>=filters['min_rooms']
            and (not filters['furnished'] or row['furnished'])
            and row['desks']>=filters['min_desks']
            and (not filters['central'] or row['central'])
            and (not filters['min_lease_months'] or
                 row['lease_months'] is not None and row['lease_months']>=filters['min_lease_months']))


def expected(task_id):
    task = task_spec(task_id)
    rows = [r for r in snapshot() if matches(r, task['filters'])]
    return [r['id'] for r in sorted(rows,key=lambda r: (-r['area'] if task['order']=='area' else r['price'],r['id']))[:3]]


def diverse_order(rows, filters):
    """Same stable price-band discovery ordering as demo snapshot.py."""
    if not rows:
        return []
    low, high = min(r['price'] for r in rows), max(r['price'] for r in rows)
    bands = [[] for _ in range(5)]
    seed = json.dumps(filters, separators=(',', ':'))
    for row in rows:
        band = min(4, (row['price'] - low) * 5 // (high - low + 1))
        key = hashlib.blake2b((seed + ':' + row['id']).encode(), digest_size=8).digest()
        bands[band].append((key, row['id']))
    ordered = [[id for _, id in sorted(band)] for band in bands]
    return [id for group in zip_longest(*ordered) for id in group if id is not None]


class Estate:
    def __init__(self):
        self.rows = {r['id']:r for r in snapshot()}
        self.answer = None
        self.fetched = set()
        self.searches = []
        self.actions = []

    def search(self, city: str, rooms: Annotated[int, Field(ge=1, le=4)], max_price: Annotated[int, Field(gt=0, le=10**15)],
               deal_type: Literal['sale', 'rent'] = 'sale',
               sort: Literal['diverse', 'price_ascending', 'price_descending'] = 'diverse',
               min_price: Annotated[int, Field(ge=0, le=10**15)] = 0,
               page: Annotated[int, Field(ge=1, le=2147483647)] = 1) -> dict:
        """Search Moscow or Saint Petersburg apartments. rooms is an exact count (1-4). RUB price bounds filter advertised amounts; rent price_period must be checked. Default diverse ordering explores price bands, not quality. Returns up to 20 compact records. Use price_ascending or price_descending when appropriate; follow next_page while has_more. Fetch details before recommendations."""
        city = CITY_ALIASES.get(city.strip().casefold())
        if city is None:
            raise ValueError('Unsupported city')
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (rooms, max_price, min_price, page)):
            raise ValueError('Expected integer filters')
        if (deal_type not in ('rent', 'sale') or rooms not in (1, 2, 3, 4)
                or not 1 <= page <= 2**31-1 or not 0 <= min_price <= max_price <= 10**15 or max_price == 0):
            raise ValueError('Invalid filters')
        if sort not in ('diverse', 'price_ascending', 'price_descending'):
            raise ValueError('Invalid sort')
        rows = [r for r in self.rows.values() if r['city']==city and r['deal_type']==deal_type
                and r['rooms']==rooms and min_price<=r['price']<=max_price]
        if sort == 'diverse':
            rows = [self.rows[id] for id in diverse_order(rows, (city, deal_type, rooms, min_price, max_price))]
        else:
            rows.sort(key=lambda r: (r['price'] if sort=='price_ascending' else -r['price'], r['id']))
        filters = dict(city=city, deal_type=deal_type, rooms=rooms, min_price=min_price, max_price=max_price, page=page, sort=sort)
        self.searches.append({**filters, 'total': len(rows)})
        result = {'status':'ok', 'source':'snapshot', 'total':len(rows), 'page':page,
                  'total_pages':(len(rows)+19)//20, 'has_more':page*20<len(rows),
                  'next_page':page+1 if page*20<len(rows) else None, 'sort':sort,
                  'min_price':min_price, 'max_price':max_price,
                  'price_unit':'RUB; rental period is specified per listing' if deal_type=='rent' else 'RUB',
                  'message':'Fictional archived snapshot, not live availability. Discovery ordering is not a suitability ranking. Fetch candidate details.',
                  'results':[{**{k:r[k] for k in ('id','url','address','price','price_period','rooms','area') if k in r},
                              'captured_at':CAPTURED_AT} for r in rows[(page-1)*20:page*20]]}
        self.actions.append({'name':'search', 'args':filters, 'result':deepcopy(result)})
        return result

    def fetch(self, id: Annotated[str, Field(pattern=r'^\d{1,12}$')]) -> dict:
        """Read all archived details by exact numeric listing ID from search results."""
        if not re.fullmatch(r'\d{1,12}', id):
            raise ValueError('Invalid listing ID')
        if id not in self.rows:
            result = {'status':'not_archived', 'source':'snapshot', 'message':'Listing is not in this snapshot. No live request was made.'}
        else:
            self.fetched.add(id)
            result = {'status':'ok', 'source':'snapshot', 'url':self.rows[id]['url'],
                      'captured_at':CAPTURED_AT, 'coverage':'search_page',
                      'text':json.dumps(self.rows[id], ensure_ascii=False)}
        self.actions.append({'name':'fetch', 'args':{'id':id}, 'result':deepcopy(result)})
        return result

    def inspect_photos(self, id: Annotated[str, Field(pattern=r'^\d{1,12}$')], question: Annotated[str, Field(min_length=1, max_length=4000)] | None = None) -> dict:
        """Inspect archived photos for a listing. This synthetic snapshot has no photos; returns explicit unavailable evidence, never fabricated observations."""
        if question is not None and (not question.strip() or len(question.strip()) > 4000):
            raise ValueError('Question must contain 1-4000 characters')
        if id not in self.fetched:
            result = self.fetch(id)
            if result['status'] != 'ok':
                return result
        result = {'status':'vision_unavailable', 'requested_images':0, 'inspected_images':0,
                  'message':'No photos exist in this fictional benchmark snapshot. No visual assessment is available; do not infer condition from images.'}
        self.actions.append({'name':'inspect_photos', 'args':{'id':id,'question':question}, 'result':deepcopy(result)})
        return result

    def submit_answer(self, answer):
        """Harness-only submission of the final coordinator answer, not an agent tool."""
        self.answer = answer

    def state(self):
        urls = list(dict.fromkeys(url.rstrip('.,;:!?') for url in re.findall(r'https?://[^\s<>()\[\]]+', self.answer or '')))
        return {'answer':self.answer, 'submitted_urls':urls, 'fetched':sorted(self.fetched),
                'searches':deepcopy(self.searches)}


def grade(task_id, state):
    ids = expected(task_id)
    rows = {r['id']:r for r in snapshot()}
    passed = bool((state.get('answer') or '').strip()) and state.get('submitted_urls') == [rows[id]['url'] for id in ids]
    passed = passed and all(id in state.get('fetched', []) for id in ids)
    if not ids:
        filters = task_spec(task_id)['filters']
        covered = {s['rooms'] for s in state.get('searches', []) if s['total']==0
                   and s['city']==filters['city'] and s['deal_type']==filters['deal']
                   and s['min_price']<=filters['min_price'] and s['max_price']>=filters['max_price']}
        says_no_matches = re.search(
            r'\bno\s+(?:(?:suitable|matching|qualifying)\s+)?(?:matches|listings|apartments|properties)\b'
            r'|\bnone\b|нет\s+(?:подходящих|совпадений|квартир|вариантов|объявлений)'
            r'|не\s+(?:найден[а-я]*|наш[а-я]*)', state.get('answer') or '', re.IGNORECASE)
        passed = passed and bool(says_no_matches) and set(range(filters['min_rooms'], 5)) <= covered
    return {'score':float(passed), 'passed':passed}
