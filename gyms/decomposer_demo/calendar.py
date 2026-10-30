"""The demo's calendar tools, backed by private state for each episode."""
from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path
import re


TASK_ID = 'calendar-airi-only'
KEEP_IDS = frozenset({'pml-2026-24', 'pml-2026-36', 'pml-2026-39'})


def load_task(task_id=TASK_ID, language='ru'):
    from .tasks import task_spec
    task = task_spec(task_id, language)
    if task['environment'] != 'calendar':
        raise ValueError('Not a calendar task')
    calendar = json.loads(Path(__file__).with_name('timetable.json').read_text())
    return {**task, 'benchmark': 'decomposer-demo', 'split': 'validation', 'calendar': calendar}


class Calendar:
    def __init__(self, task):
        self.calendar = deepcopy(task['calendar'])
        self.events = self.calendar.pop('events')
        self.actions = []
        self._next_id = 1

    def _time(self, value):
        if not isinstance(value, str) or not re.fullmatch(r'(?:[01]\d|2[0-3]):[0-5]\d|24:00', value):
            raise ValueError('Use local HH:mm; 24:00 is allowed only as an end time')
        midnight = datetime.fromisoformat(self.calendar['date'] + 'T00:00:00' + self.calendar['utcOffset'])
        hour, minute = map(int, value.split(':'))
        return midnight + timedelta(hours=hour, minutes=minute)

    def _range(self, start, end, max_hours):
        a, b = self._time(start), self._time(end)
        if start == '24:00' or not timedelta(0) < b - a <= timedelta(hours=max_hours):
            raise ValueError(f'Provide an increasing range of at most {max_hours} hours')
        return a, b

    def get_events(self, start: str, end: str) -> list[dict]:
        """Read overlapping events in a local HH:mm interval of at most six hours."""
        a, b = self._range(start, end, 6)
        result = sorted((e for e in self.events if datetime.fromisoformat(e['end']) > a
                         and datetime.fromisoformat(e['start']) < b), key=lambda e: e['start'])
        self.actions.append({'name': 'get_events', 'args': {'from': start, 'to': end},
                             'result': deepcopy(result)})
        return deepcopy(result)

    def remove_event(self, id: str) -> dict:
        """Remove an event by its exact ID, returning its complete metadata."""
        event = next((e for e in self.events if e['id'] == id), None)
        if event is None:
            raise ValueError(f'Event not found: {id}')
        self.events.remove(event)
        self.actions.append({'name': 'remove_event', 'args': {'id': id}, 'result': deepcopy(event)})
        return deepcopy(event)

    def create_event(self, title: str, start: str, end: str, description: str = '',
                     location: str = '', people: list[dict] | None = None,
                     tags: list[str] | None = None, url: str = '') -> dict:
        """Create an explicitly requested event, with local HH:mm times and known metadata."""
        a, b = self._range(start, end, 24)
        if not title.strip():
            raise ValueError('Event title must be nonempty')
        event = {'id': f'created-{self._next_id}', 'title': title.strip(),
                 'start': a.isoformat(), 'end': b.isoformat()}
        for key, value in {'description': description, 'location': location, 'people': people,
                           'tags': tags, 'url': url}.items():
            if value:
                event[key] = deepcopy(value)
        self._next_id += 1
        self.events.append(event)
        self.actions.append({'name': 'create_event',
                             'args': {'title': title, 'start': start, 'end': end,
                                      'description': description, 'location': location,
                                      'people': deepcopy(people), 'tags': deepcopy(tags), 'url': url},
                             'result': deepcopy(event)})
        return deepcopy(event)


def grade(task_id, events):
    """Exact final-state reward; changed metadata and extra events also fail."""
    task = load_task(task_id)
    from .tasks import CALENDAR
    keep = next(t[3] for t in CALENDAR if t[0] == task_id)
    expected = [e for e in task['calendar']['events'] if keep(e)]
    passed = sorted(events, key=lambda e: e['id']) == sorted(expected, key=lambda e: e['id'])
    return {'score': float(passed), 'passed': passed}
