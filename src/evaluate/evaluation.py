import numpy as np
from sklearn import metrics
import torch
import scipy.stats

"""
Code taken directly from https://github.com/felipebravom/SemEval_2018_Task_1_Eval/blob/master/utils.py
"""
def cohen_kappa_score(y1, y2, labels=None, weights=None):
  """Cohen's kappa: a statistic that measures inter-annotator agreement.
  This function computes Cohen's kappa [1]_, a score that expresses the level
  of agreement between two annotators on a classification problem. It is
  defined as
  .. math::
      \kappa = (p_o - p_e) / (1 - p_e)
  where :math:`p_o` is the empirical probability of agreement on the label
  assigned to any sample (the observed agreement ratio), and :math:`p_e` is
  the expected agreement when both annotators assign labels randomly.
  :math:`p_e` is estimated using a per-annotator empirical prior over the
  class labels [2]_.
  Read more in the :ref:`User Guide <cohen_kappa>`.
  Parameters
  ----------
  y1 : array, shape = [n_samples]
      Labels assigned by the first annotator.
  y2 : array, shape = [n_samples]
      Labels assigned by the second annotator. The kappa statistic is
      symmetric, so swapping ``y1`` and ``y2`` doesn't change the value.
  labels : array, shape = [n_classes], optional
      List of labels to index the matrix. This may be used to select a
      subset of labels. If None, all labels that appear at least once in
      ``y1`` or ``y2`` are used.
  weights : str, optional
      List of weighting type to calculate the score. None means no weighted;
      "linear" means linear weighted; "quadratic" means quadratic weighted.
  Returns
  -------
  kappa : float
      The kappa statistic, which is a number between -1 and 1. The maximum
      value means complete agreement; zero or lower means chance agreement.
  References
  ----------
  .. [1] J. Cohen (1960). "A coefficient of agreement for nominal scales".
          Educational and Psychological Measurement 20(1):37-46.
          doi:10.1177/001316446002000104.
  .. [2] `R. Artstein and M. Poesio (2008). "Inter-coder agreement for
          computational linguistics". Computational Linguistics 34(4):555-596.
          <http://www.mitpressjournals.org/doi/abs/10.1162/coli.07-034-R2#.V0J1MJMrIWo>`_
  .. [3] `Wikipedia entry for the Cohen's kappa.
          <https://en.wikipedia.org/wiki/Cohen%27s_kappa>`_
  """
  y1 = y1.numpy()
  y2 = y2.numpy()
  confusion = metrics.confusion_matrix(y1, y2, labels=labels)
  n_classes = confusion.shape[0]
  sum0 = np.sum(confusion, axis=0)
  sum1 = np.sum(confusion, axis=1)
  expected = np.outer(sum0, sum1)*1.0 / np.sum(sum0)

  if weights is None:
    w_mat = np.ones([n_classes, n_classes], dtype=np.int)
    w_mat.flat[:: n_classes + 1] = 0
  elif weights == "linear" or weights == "quadratic":
    w_mat = np.zeros([n_classes, n_classes], dtype=np.int)
    w_mat += np.arange(n_classes)
    if weights == "linear":
      w_mat = np.abs(w_mat - w_mat.T)
    else:
      w_mat = (w_mat - w_mat.T) ** 2
  else:
    raise ValueError("Unknown kappa weighting type.")

  k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
  
  return 1 - k

def evaluate_PerEmotion (gold_scores, pred_scores):
  gold_scores_some=[]
  pred_scores_some=[]

  for i in range(len(gold_scores)):
    if(gold_scores[i]!=0):            
      gold_scores_some.append(gold_scores[i])
      pred_scores_some.append(pred_scores[i])
  pred_scores_some=torch.tensor(pred_scores_some)
  gold_scores_some=torch.tensor(gold_scores_some)

  if np.std(pred_scores.numpy())==0 or np.std(gold_scores.numpy())==0:
    return (0,0,0,0)

  pears_corr=scipy.stats.pearsonr(pred_scores,gold_scores)[0]
  pears_corr_some=scipy.stats.pearsonr(pred_scores_some,gold_scores_some)[0]  

  gold_labels=list(sorted(set(gold_scores.numpy())))

  kappa=cohen_kappa_score(pred_scores,gold_scores,labels=gold_labels, weights='quadratic')        
  kappa_some=cohen_kappa_score(pred_scores_some,gold_scores_some, labels=gold_labels, weights='quadratic')

  return (pears_corr,pears_corr_some,kappa,kappa_some)