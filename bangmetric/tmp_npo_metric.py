import numpy as np
from scipy.integrate import trapz

# TODO: consider using scikits.learn.metrics here
# see http://scikit-learn.sourceforge.net/modules/classes.html#module-scikits.learn.metrics
# e.g. accuracy = scikits.learn.metrics.zero_one_score

def accuracy_from_boolean(preds, gt):
    """
    **Description**
        computes the accuracy of a prediction given ground truth results
    **Inputs**
        preds: array of booleans
            contains the predicted values
        gt: array of booleans
            contains the ground truth baseline
    **Output**
        accuracy: float
            accuracy of the prediction as defined by the proportion
            of "True" results in the population.
    **Reference**
        http://en.wikipedia.org/wiki/Accuracy_and_precision#Accuracy_in_binary_classification
    """
    assert len(preds) == len(gt)
    # the array (preds == gt) contains "True" values only
    # for true positives and true negatives
    accuracy = (preds == gt).mean()
    return accuracy


def recall_from_boolean(preds, gt):
    """
    **Description**
        computes the recall value of a prediction given a ground
        truth baseline
    **Inputs**
        preds: array of booleans
            contains the predicted values
        gt: array of booleans
            contains the ground truth baseline
    **Output**
        recall: float
            global recall, i.e. the ratio between the number of true
            positives to the sum of true positives and false negative
    **Reference**
        http://en.wikipedia.org/wiki/Precision_and_recall
    """
    assert len(preds) == len(gt)
    tp = float(gt[np.where(preds)].sum())
    tp_plus_fn = float(gt.sum())
    recall = tp / tp_plus_fn

    return recall


def precision_from_boolean(preds, gt):
    """
    **Description**                                             
        computes the precision value of a prediction given a ground
        truth baseline                                          
    **Inputs**                                                  
        preds: array of booleans                                
            contains the predicted values                       
        gt: array of booleans                                   
            contains the ground truth baseline                  
    **Output**                                                  
        precision: float                                           
            global precision, i.e. the ratio between the number of true
            positives to the sum of true positives and false positive
    **Reference**                                               
        http://en.wikipedia.org/wiki/Precision_and_recall
    """
    assert len(preds) == len(gt)
    tp = float(gt[np.where(preds)].sum())
    tp_plus_fp = float(preds.sum())
    precision = tp / tp_plus_fp

    return precision


def average_precision(preds, gt, integration_type='trapz'):
    """
    **Description**
        computes the average precision of a prediction given
        ground truth results
    **Inputs**
        preds: array of floats
            array containing the predictions (the values should
            all be positive)
        gt: array of floats
            array containing the ground truth baseline (the values
            can be positive and negative)
        integration_type: string
            which type of integration method to use for
            computing the area under the curve precision(recall)
            default is trapezoidal rule 'trapz'
            another possibility is the innacurate method used in
            Pascal VOC 2007 'voc2007'
    **Ouputs**
        ap: float
            average precision of the prediction
    **References**
        Code converted from Pascal VOC 2007 devkit:
        http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
        
        I believe the average precision code has changed in Pascal VOC 2010+, check it out:
        http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2010/VOCdevkit_08-May-2010.tar
    **Warning**
        in this algorithm, a 0 value for the ground truth data is **ignored** in
        computing the average precision
    """
    assert len(preds) == len(gt)
    assert np.sum(preds >= 0.) == len(preds)
    # first find the "natural" ranking order by sorting
    # the predictions from highest score to lowest score
    si = np.argsort(-preds)
    # then compute the cumulative number of "true positive"
    # by comparing with the ground truth for the same ranking
    tp = np.cumsum(np.single(gt[si] > 0))
    # similarly compute the cumulative number of "false positive"
    # by comparing with the ground truth for the same ranking
    fp = np.cumsum(np.single(gt[si] < 0))
    # if all the examples are actually wrong (i.e. all the
    # gt values are negative) then the average precision can
    # only be zero since in this case there are no true positive
    if np.sum(gt > 0) == 0:
        ap = 0.
        return ap
    # compute the cumulative recall rate
    rec = tp / np.sum(gt > 0)
    # compute the cumulative precision rate
    prec = tp / (fp + tp)

    if integration_type == 'trapz':
        # in this case, we extend the recall array "to the left"
        # by adding 0 if and only if rec[0] is different from 0.
        if rec[0] != 0.:
            rec = np.array([0.] + rec.tolist())
            # a similar thing happens to the precision array where
            # we add the value prec[0] "to the left" if and only
            # if rec[0] is different from 0.
            prec = np.array([prec[0]] + prec.tolist())
        # now we can integrate the area under the precision(recall)
        # curve to obtain the 'exact' average precision
        ap = trapz(prec, rec)
    elif integration_type == 'voc2007':
        # the code below tries to integrate the area under
        # the curve precision(recall) by using a grid of recall
        # values and creating
        ap = 0
        rng = np.arange(0, 1.1, .1)
        for th in rng:
            p = prec[rec >= th].max()
            if p == []:
                p = 0
            ap += p / rng.size
    else:
        raise ValueError('integration type "%s" not recognized' %
                integration_type)

    return ap
