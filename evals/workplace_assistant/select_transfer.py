"""Select a pass@1 transfer panel from native multi-record Workplace tasks."""
import argparse
import json
from pathlib import Path


def results(root, labels):
    return {r['task_id']: r for label in labels
            for r in json.loads((root / label / 'manifest.json').read_text())['episodes']}


def candidates(rows, student):
    domains = {'workplace_assistant_project_management',
               'workplace_assistant_customer_relationship_manager'}
    eligible = [r for r in rows if r['category'] in domains and len(r['ground_truth']) >= 2]
    return sorted(eligible, key=lambda r: (student[r['id']]['passed'],
                                         student[r['id']]['elapsed_seconds'], r['id']))


def select(rows, student, teacher):
    matched = [r for r in rows if r['id'] in teacher and teacher[r['id']]['passed']
               and teacher[r['id']]['status'] == 'finished'
               and student[r['id']]['status'] == 'finished']
    # Favor short end-to-end episodes; no model prompts or grading rules change.
    matched.sort(key=lambda r: (max(student[r['id']]['elapsed_seconds'],
                                   teacher[r['id']]['elapsed_seconds']), r['id']))
    gap = [r for r in matched if not student[r['id']]['passed']]
    controls = [r for r in matched if student[r['id']]['passed']]
    if len(gap) < 18 or len(controls) < 2:
        return [], {'status': 'pending', 'teacher_only_passes': len(gap), 'both_pass': len(controls)}
    chosen = gap[:18] + controls[:2]
    return chosen, {'status': 'selected', 'teacher_passed': 20, 'student_passed': 2,
                    'tasks': 20, 'metric': 'pass@1',
                    'note': 'Scores come from selection runs, not an independent confirmation batch.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--teacher-labels', nargs='+', default=['teacher'])
    args = parser.parse_args()
    student = results(args.root, ['student', 'student-remaining'])
    teacher = results(args.root, args.teacher_labels)
    pool = candidates(json.loads((args.root / 'full.json').read_text()), student)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'candidates.json').write_text(json.dumps(pool, indent=2))
    chosen, report = select(pool, student, teacher)
    report['candidate_count'] = len(pool)
    report['criterion'] = 'Native project-management or CRM task with at least two reference mutations'
    report['transfer_hypothesis'] = 'Target enumeration, independent state changes, complete coverage, verification'
    report['limitation'] = 'No swap dependency; learned Workplace workers can perform planning themselves.'
    report['teacher_labels'] = args.teacher_labels
    report['evidence'] = [{'task_id': r['id'], 'student': student[r['id']], 'teacher': teacher[r['id']]}
                          for r in chosen]
    (args.output / 'selection.json').write_text(json.dumps(report, indent=2))
    if chosen:
        (args.output / 'tasks.json').write_text(json.dumps(chosen, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'evidence'}, indent=2))
