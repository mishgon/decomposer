"""Dockerless demo environments and native result checking."""
from .calendar import Calendar, load_task
from .tasks import task_ids, task_spec


def grade(task_id, final_state, language='ru'):
    environment=task_spec(task_id,language)['environment']
    if environment=='calendar':
        from .calendar import grade as check
        return check(task_id,final_state)
    if environment=='estate':
        from .estate import grade as check
        return check(task_id,final_state)
    from .frontend import grade as check
    return check(task_id,final_state,language)
