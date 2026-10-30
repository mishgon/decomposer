"""Twenty frozen validation tasks; EN/RU are two views of the same task IDs."""
CALENDAR = [
('calendar-airi-only', 'Keep only talks and panel discussions with at least one AIRI-affiliated speaker in my calendar. Delete the other events.', 'Оставь в календаре только выступления и дискуссии, среди спикеров которых есть сотрудник AIRI. Остальные события удали.', lambda e: any(p.get('organization') == 'AIRI' and p.get('role', '').startswith('speaker') for p in e.get('people', []))),
('calendar-remove-alice', 'Delete events whose title contains Alice or Алиса (including inflected forms). Keep every other event unchanged.', 'Удали события, в заголовке которых упоминается Alice или Алиса (включая склонения). Остальные события оставь без изменений.', lambda e: not any(s in e['title'].lower() for s in ['alice', 'алис'])),
('calendar-online', 'Keep only online events (tag online). Delete all other events, preserving all metadata of the retained ones.', 'Оставь только онлайн-события с тегом online. Остальные удали; все метаданные оставшихся сохрани.', lambda e: 'online' in e.get('tags', [])),
('calendar-nlp', 'Keep only talks and keynotes tagged NLP; remove programme events and everything else.', 'Оставь только доклады и ключевые выступления с тегом NLP; программные события и всё остальное удали.', lambda e: 'NLP' in e.get('tags', []) and bool({'talk', 'keynote'} & set(e.get('tags', [])))),
('calendar-before-five', 'Delete every event starting at or after 17:00 local time. Leave earlier events unchanged, even if they end later.', 'Удали все события, которые начинаются в 17:00 или позже по местному времени. Более ранние оставь без изменений, даже если они заканчиваются позже.', lambda e: e['start'][11:16] < '17:00'),
('calendar-no-programme', 'Remove every event tagged programme, but keep all talks, keynotes and panels unchanged.', 'Удали все события с тегом programme, а доклады, ключевые выступления и дискуссии сохрани без изменений.', lambda e: 'programme' not in e.get('tags', [])),
('calendar-recsys', 'Keep only events tagged RecSys. Preserve their complete metadata and delete the rest.', 'Оставь только события с тегом RecSys. Сохрани все их метаданные, остальные события удали.', lambda e: 'RecSys' in e.get('tags', [])),
('calendar-yandex-afternoon', 'Keep events starting before 17:00 with a speaker whose organization contains Яндекс or Yandex. Delete all others.', 'Оставь события с началом до 17:00, среди спикеров которых есть представитель организации с Яндекс или Yandex в названии. Остальные удали.', lambda e: e['start'][11:16] < '17:00' and any(p.get('role','').startswith('speaker') and any(x in p.get('organization','').lower() for x in ['яндекс','yandex']) for p in e.get('people', []))),
]

# Scope, filters and a single unambiguous ordering are visible to the agent.
ESTATE = [
('estate-moscow-work', 'moscow', 'rent', 120000, 150000, 2, True, 2, 6, 'price', 'Three apartments for remote work in Moscow', 'Три квартиры для удалённой работы в Москве'),
('estate-petersburg-students', 'saint-petersburg', 'rent', 80000, 100000, 3, True, 3, 6, 'price', 'Three apartments for three students in Saint Petersburg', 'Три квартиры для трёх студентов в Санкт-Петербурге'),
('estate-moscow-buy', 'moscow', 'sale', 320000000, 400000000, 3, False, 0, 0, 'price', 'Three apartments to buy in central Moscow', 'Три квартиры для покупки в центре Москвы'),
('estate-budget', 'moscow', 'rent', 60000, 100000, 1, True, 1, 12, 'price', 'Three furnished budget apartments in Moscow', 'Три недорогие меблированные квартиры в Москве'),
('estate-large', 'saint-petersburg', 'rent', 80000, 160000, 3, False, 2, 6, 'area', 'Three spacious apartments in Saint Petersburg', 'Три просторные квартиры в Санкт-Петербурге'),
('estate-no-matches', 'moscow', 'sale', 1000000, 2000000, 4, True, 4, 0, 'price', 'Check whether this unusually low purchase budget has matches', 'Проверь, есть ли варианты при необычно низком бюджете покупки'),
]

FRONTEND = [
('frontend-notebook', 'Make the chat demo feel like a cozy study notebook', 'Сделай демо чата похожим на уютную учебную тетрадь', {(':root','--paper'):'#fff8e7',(':root','--ink'):'#302b24',(':root','--accent'):'#a65f35',(':root','--radius'):'8px'}),
('frontend-confetti', 'Restyle the demo as a vivid Confetti as a Service page', 'Оформи демо как яркую страницу «Конфетти как сервис»', {(':root','--paper'):'#160f29',(':root','--surface'):'#24183b',(':root','--ink'):'#ffffff',(':root','--accent'):'#ff6bd6',(':root','--accent-ink'):'#160f29'}),
('frontend-magazine', 'Give the demo a clean technical magazine layout', 'Придай демо вид аккуратного технического журнала', {(':root','--paper'):'#ffffff',(':root','--ink'):'#111827',(':root','--accent'):'#2563eb',('.page','max-width'):'960px',('.description','max-width'):'72ch'}),
('frontend-warm-dark', 'Create a warm dark theme for the chat demo', 'Создай тёплую тёмную тему для демо чата', {(':root','--paper'):'#1c1917',(':root','--surface'):'#292524',(':root','--soft'):'#44403c',(':root','--ink'):'#fafaf9',(':root','--muted'):'#d6d3d1',(':root','--line'):'#57534e'}),
('frontend-roomy', 'Make the editing and conversation area more spacious', 'Сделай область редактирования и переписки просторнее', {('.page','max-width'):'1100px',('textarea','min-height'):'180px',('.main','padding'):'32px',('.message','padding'):'20px'}),
('frontend-compact', 'Make a compact, square-cornered version for a narrow workspace', 'Сделай компактную версию с прямыми углами для узкой рабочей области', {(':root','--radius'):'0px',('.page','max-width'):'640px',('.controls','gap'):'8px',('textarea','min-height'):'72px'}),
]


def task_ids():
    return [t[0] for group in (CALENDAR, ESTATE, FRONTEND) for t in group]


def task_spec(task_id, language='ru'):
    if language not in {'en', 'ru'}:
        raise ValueError('language must be en or ru')
    for id, en, ru, keep in CALENDAR:
        if id == task_id:
            return {'task_id': id, 'environment': 'calendar', 'prompt': ru if language == 'ru' else en, 'language': language}
    for id, city, deal, low, high, rooms, furnished, desks, months, order, en, ru in ESTATE:
        if id == task_id:
            filters = dict(city=city, deal=deal, min_price=low, max_price=high, min_rooms=rooms,
                           furnished=furnished, min_desks=desks, min_lease_months=months, central=deal=='sale')
            if language == 'en':
                city_name = 'Moscow' if city == 'moscow' else 'Saint Petersburg'
                amount = f'{low:,}-{high:,} RUB' + (' per month' if deal == 'rent' else ' purchase price')
                prompt = f'{en}. Find up to three apartments in {city_name} for {amount}, with at least {rooms} rooms. '
                if furnished:
                    prompt += 'They must be furnished. '
                if desks:
                    prompt += f'Require space for at least {desks} desks. '
                if months:
                    prompt += f'The minimum supported lease term is {months} months. '
                if deal == 'sale':
                    prompt += 'Only central locations qualify. '
                prompt += ('Rank by area descending' if order == 'area' else 'Rank by price ascending')
                prompt += ', breaking ties by listing ID ascending. Fetch candidate details and present the final ranked shortlist with exact source links. Only include qualifying listings; do not add rejected candidates to the answer. If none match, say so. Unknown fields do not satisfy requirements. Photos are unavailable in this fictional snapshot: use the archived fields and state that visual conditions were not verified.'
            else:
                city_name = 'Москве' if city == 'moscow' else 'Санкт-Петербурге'
                amount = f'{low:,}-{high:,} рублей' + (' в месяц' if deal == 'rent' else ' за покупку')
                prompt = f'{ru}. Найди до трёх квартир в {city_name} за {amount}, не менее {rooms} комнат. '
                if furnished:
                    prompt += 'Обязательна мебель. '
                if desks:
                    prompt += f'Нужно место минимум для {desks} рабочих столов. '
                if months:
                    prompt += f'Подтверждённый минимальный срок аренды должен быть не меньше {months} месяцев. '
                if deal == 'sale':
                    prompt += 'Подходят только центральные районы. '
                prompt += ('Сортируй по убыванию площади' if order == 'area' else 'Сортируй по возрастанию цены')
                prompt += ', при равенстве по ID объявления по возрастанию. Прочитай детали кандидатов и приведи итоговый список в нужном порядке с точными ссылками на источники. В ответе перечисли только подходящие варианты, без отклонённых кандидатов. Если совпадений нет, сообщи об этом. Неизвестные значения не подтверждают соответствие условиям. Фото в этом вымышленном снимке недоступны: используй сохранённые поля и укажи, что внешний вид не проверен.'
            return {'task_id': id, 'environment':'estate', 'prompt':prompt, 'filters':filters, 'order':order, 'language':language}
    for id, en, ru, targets in FRONTEND:
        if id == task_id:
            requirements = '; '.join(f'{selector}: {key}={value}' for (selector,key),value in targets.items())
            prompt = ((f'{ru}. Измени /styles.css по точной спецификации: {requirements}. '
                       'Сохрани все остальные CSS-свойства и остальные файлы без изменений. Используй обычные CSS-правила без !important. Это проверка исходников; браузер и скриншоты недоступны.') if language=='ru' else
                      (f'{en}. Change /styles.css to this exact specification: {requirements}. '
                       'Preserve all other CSS properties and every other file. Use ordinary CSS rules without !important. This is source validation; no browser or screenshots are available.'))
            return {'task_id':id, 'environment':'frontend', 'prompt':prompt, 'targets':targets, 'language':language}
    raise ValueError(f'Unknown decomposer-demo task: {task_id}')
