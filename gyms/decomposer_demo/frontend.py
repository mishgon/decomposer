"""In-memory demo project. Generated source is never executed."""
from copy import deepcopy
from gyms.local_workers import worker_id
from pathlib import Path
from typing import Annotated
from pydantic import Field

from .tasks import task_spec

ProjectPath = Annotated[str, Field(max_length=160, pattern=r'^/(?:[\w.-]+/)*[\w.-]+$')]
FileContent = Annotated[str, Field(max_length=100000)]
Revision = Annotated[int, Field(ge=0)]



def seed(language='ru'):
    root=Path(__file__).with_name('frontend_seed')
    files={'/'+name:(root/name).read_text() for name in ('index.js','api.js','tools.js','styles.css','README.md')}
    files.update({'/App.jsx':(root/f'App.{language}.jsx').read_text(),
                  '/ChatDemo.jsx':(root/f'ChatDemo.{language}.jsx').read_text(),
                  '/public/index.html':f'<!doctype html><html lang="{language}"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>{"Демо чата" if language=="ru" else "Chat demo"}</title></head><body><div id="root"></div></body></html>'})
    import json
    files['/package.json']=json.dumps({'name':'chat-demo','version':'1.0.0','main':'/index.js','dependencies':{'react':'18.3.1','react-dom':'18.3.1','react-confetti':'6.1.0'}},indent=2)
    return files


class Frontend:
    def __init__(self, language):
        self.files = seed(language)
        self.revisions = {p: 1 for p in self.files}
        self.version = 0
        self.seen = {}
        self.actions = []

    @staticmethod
    def _path(path):
        import re
        if len(path) > 160 or not re.fullmatch(r'/(?:[\w.-]+/)*[\w.-]+', path, re.ASCII) or any(p in {'.', '..'} for p in path.split('/')):
            raise ValueError('Use an absolute project path without . or ..')

    @staticmethod
    def _characters(content):
        # JavaScript String.length counts UTF-16 code units.
        return len(content.encode('utf-16-le')) // 2

    def list_files(self) -> dict:
        """List every project file and its current revision. Files are shared by all frontend agents in this session."""
        return {'version': self.version,
                'files': [{'path': p, 'revision': self.revisions[p], 'characters': self._characters(self.files[p])} for p in sorted(self.files)],
                'deleted': [{'path': p, 'revision': revision} for p, revision in self.revisions.items() if p not in self.files]}

    def read_file(self, path: ProjectPath) -> dict:
        """Read a complete file and its revision. Read before changing a file."""
        self._path(path)
        if path not in self.files:
            raise ValueError(f'File not found: {path}')
        self.seen[worker_id.get(), path] = self.revisions[path]
        return {'path': path, 'revision': self.revisions[path], 'content': self.files[path]}

    def _change(self, path, expected_revision, transform):
        self._path(path)
        actual = self.revisions.get(path, 0)
        expected = expected_revision if expected_revision is not None else self.seen.get((worker_id.get(), path), 0 if actual == 0 else None)
        if expected is None:
            raise ValueError(f'Read {path} before changing it, or pass expected_revision explicitly.')
        if actual != expected:
            raise ValueError(f'Revision conflict for {path}: expected {expected}, current {actual}. Read the file again.')
        content = transform(self.files.get(path))
        if content is not None and self._characters(content) > 100000:
            raise ValueError('File exceeds 100,000 characters')
        files = dict(self.files)
        if content is None:
            files.pop(path)
        else:
            files[path] = content
        if len(files) > 80 or sum(self._characters(v) for v in files.values()) > 1000000:
            raise ValueError('Project size limit exceeded')
        self.files = files
        self.revisions[path] = actual + 1
        self.version += 1
        self.seen[worker_id.get(), path] = actual + 1
        return {'path': path, 'revision': actual + 1, 'deleted': content is None}

    def write_file(self, path: ProjectPath, content: FileContent, expected_revision: Revision | None = None) -> dict:
        """Create or replace a file. Read existing files first. Automatically uses your last read/write revision; optionally pass expected_revision explicitly. New paths need no revision. Stale revisions are rejected. Returns the new revision."""
        result = self._change(path, expected_revision, lambda previous: content)
        self.actions.append({'name': 'write_file', 'path': path, 'content': content, **result})
        return result

    def edit_file(self, path: ProjectPath, old_text: Annotated[str, Field(min_length=1, max_length=100000)], new_text: FileContent, expected_revision: Revision | None = None) -> dict:
        """Replace one exact occurrence of old_text with new_text. Automatically uses your last read/write revision; expected_revision is optional. Fails if the text is missing, ambiguous, or another agent changed the file. Re-read and retry on conflicts."""
        def edit(previous):
            if previous is None:
                raise ValueError(f'File not found: {path}')
            at = previous.find(old_text)
            if not old_text or at < 0:
                raise ValueError('old_text was not found. Read the file again.')
            if previous.find(old_text, at + 1) >= 0:
                raise ValueError('old_text matches more than once. Include more surrounding text.')
            return previous[:at] + new_text + previous[at + len(old_text):]
        result = self._change(path, expected_revision, edit)
        self.actions.append({'name': 'edit_file', 'path': path, 'old_text': old_text, 'new_text': new_text, **result})
        return result

    def delete_file(self, path: ProjectPath, expected_revision: Revision | None = None) -> dict:
        """Read first, then delete using your last read/write revision; expected_revision is optional. Keep the application entry point and package.json working."""
        def delete(previous):
            if previous is None:
                raise ValueError(f'File not found: {path}')
            return None
        result = self._change(path, expected_revision, delete)
        self.actions.append({'name': 'delete_file', **result})
        return result

    def get_preview_state(self) -> dict:
        """Read preview diagnostics. This Python environment has no browser; status is always unavailable."""
        return {'project_version': self.version, 'status': 'unavailable', 'error': None, 'logs': []}

    def inspect_screenshot(self, question: Annotated[str, Field(min_length=1, max_length=4000)] | None = None) -> dict:
        """Inspect the preview screenshot. Unavailable in this Python-only environment; no visual evidence is produced."""
        raise ValueError('Preview and screenshot capture are unavailable in this Python-only environment.')

    def state(self):
        return deepcopy(self.files)

def css_map(source):
    import tinycss2
    result={}
    def walk(rules, scope=()):
        for rule in rules:
            if rule.type=='error':
                raise ValueError('Invalid CSS')
            if rule.type=='qualified-rule':
                selector=' '.join(tinycss2.serialize(rule.prelude).strip().split())
                for declaration in tinycss2.parse_declaration_list(rule.content,skip_comments=True,skip_whitespace=True):
                    if declaration.type!='declaration':
                        raise ValueError('Invalid declaration')
                    value=tinycss2.serialize(declaration.value).strip()
                    result[(*scope,selector,declaration.name)]=(value,declaration.important)
            elif rule.type=='at-rule' and rule.lower_at_keyword=='media' and rule.content is not None:
                media=' '.join(tinycss2.serialize(rule.prelude).strip().split())
                walk(tinycss2.parse_rule_list(rule.content,skip_comments=True,skip_whitespace=True),(*scope,'@media '+media))
            else:
                raise ValueError('Only ordinary CSS and media rules are supported by this source checker')
    walk(tinycss2.parse_stylesheet(source,skip_comments=True,skip_whitespace=True))
    return result


def grade(task_id, state, language='ru'):
    initial=seed(language);task=task_spec(task_id,language)
    if set(state)!=set(initial) or any(state[p]!=v for p,v in initial.items() if p!='/styles.css'):
        return {'score':0.,'passed':False}
    try:
        before,after=css_map(initial['/styles.css']),css_map(state['/styles.css'])
        expected={**before,**{key:(value,False) for key,value in task['targets'].items()}}
        passed=after==expected
    except (ValueError,TypeError):
        passed=False
    return {'score':float(passed),'passed':passed}
