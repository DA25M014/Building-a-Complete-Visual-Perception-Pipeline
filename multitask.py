'''Re-export for autograder compatibility.

The autograder does 'from multitask import MultiTaskPerceptionModel'
which resolves to this top-level module.
'''

from models.multitask import MultiTaskPerceptionModel

__all__ = ["MultiTaskPerceptionModel"]
