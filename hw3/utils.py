#!/usr/bin/env python3

import contractions
import numpy as np


def split_info(data):
    def to_numpy(*args):
        return [np.array(lst) for lst in args]

    context = []
    question = []
    answer_text = []
    answer_start = []

    for q in data:
        context.append(contractions.fix(q["context"].numpy().decode("UTF-8")))
        question.append(contractions.fix(q["question"].numpy().decode("UTF-8")))
        answer_text.append(
            contractions.fix(q["answers"]["text"].numpy()[0].decode("UTF-8"))
        )
        answer_start.append(q["answers"]["answer_start"].numpy()[0])

    return to_numpy(context, question, answer_text, answer_start)
