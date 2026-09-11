"""Exercise the evaluator branch reached only after an agent creates a form."""
import argparse
import json
from pathlib import Path
from gyms.toolathlon_gym.episode import Episode
from tests.toolathlon_gym.check_environment import probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    episode = Episode("canvas-quiz-performance-gcal-gform", args.output,
                      subagent_port=8025, image=args.image)
    try:
        episode.start()
        def call(name, arguments):
            return json.loads(probe(episode, "google-forms-mcp", "build/index.js", name, arguments))
        form = call("create_form", {"title": "Quiz Improvement Feedback"})
        text = json.loads(form["content"][0]["text"])
        form_id = text["formId"]
        call("add_multiple_choice_question", {"formId": form_id,
             "questionTitle": "Which topic is most challenging?",
             "options": ["Programming Fundamentals", "Data Structures", "Algorithms", "User Interface Design"]})
        count = episode.command("exec", episode.pg, "psql", "-U", "eigent", "-d", "toolathlon_gym",
                                "-Atc", "SELECT count(*) FROM gform.question_options").stdout.strip()
        assert count == "4", count
        result = episode.score()
        assert "Traceback" not in result["stderr"]
        assert "MC options include all four prescribed topics" in result["stdout"]
        print("PASS: actual MCP form options are visible to the native evaluator")
    finally:
        episode.close()


if __name__ == "__main__":
    main()
