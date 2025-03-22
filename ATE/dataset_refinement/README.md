This folder was used in an attempt of refining the dataset.

This is an optional, extra step, it's performance is heavily METRIC - dependant, and should be checked or tested.

The refinement consist in the model having access to a query and the model answer.
The framework gives these two elements to the llm and prompts him to:
    a) if the answer is correct, use it as your new answer.
    b) if it is incorrect, provide the corrected answer.
The model manages to correct some instances, while others are wrongly changed.
This happens because in the main cycle, the model tries to call the metric and actually modify its response based on the error feedback and his thought process, while this refinement is one-shot.

The suggested approach is to refine the dataset check by metric if the quality is improved.

Ideally, the new answers should be parsed again and re-tested with a evaluate call.