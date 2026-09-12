"""
Rule-based sequence checker — deliberately NOT another AI model.
For a mission-critical check ("was the protocol followed correctly"),
a plain, auditable state machine is the right tool: its behavior is
100% predictable and easy to verify, unlike a second neural network.
"""

import yaml


class Result:
    def __init__(self, status, step_index, expected):
        self.status = status  # "confirmed" | "out_of_order" | "unrecognized" | "complete"
        self.step_index = step_index
        self.expected = expected


class SequenceValidator:
    def __init__(self, protocol_path):
        with open(protocol_path) as f:
            protocol = yaml.safe_load(f)
        self.steps = protocol["steps"]
        self.current_step = 0

    def check(self, detected_activity):
        if self.current_step >= len(self.steps):
            return Result("complete", self.current_step, None)

        expected = self.steps[self.current_step]

        if detected_activity == expected:
            self.current_step += 1
            next_expected = (
                self.steps[self.current_step]
                if self.current_step < len(self.steps)
                else None
            )
            return Result("confirmed", self.current_step, next_expected)

        if detected_activity in self.steps:
            return Result("out_of_order", self.current_step, expected)

        return Result("unrecognized", self.current_step, expected)

    def reset(self):
        self.current_step = 0
