"""
Author: Subho Mukherjee (submukhe@microsoft.com)
Code for XtremeDistil for distilling massive multi-lingual models.
"""

import conlleval
import logging
import numpy as np

logger = logging.getLogger('xtremedistil')

def ner_evaluate(model, X_test, y_test, labels, special_tokens, MAX_SEQUENCE_LENGTH, batch_size=32):

    y_pred = model.predict(X_test, batch_size=batch_size)
    pred_tags_all = []
    true_tags_all = []
    for i, seq in enumerate(y_pred):
        for j in range(MAX_SEQUENCE_LENGTH):
            indx = y_test[i][j]
            true_label = labels[indx]
            if special_tokens["pad_token"] in true_label or special_tokens["bos_token"] in true_label or special_tokens["eos_token"] in true_label:
                continue
            true_tags_all.append(true_label)
            indx = np.argmax(seq[j])
            pred_label = labels[indx]
            pred_tags_all.append(pred_label)
    prec, rec, f1 = conlleval.evaluate(true_tags_all, pred_tags_all, special_tokens, verbose=True)
    logger.info ("Test scores {} {} {}".format(prec, rec, f1))

    return np.mean(f1)