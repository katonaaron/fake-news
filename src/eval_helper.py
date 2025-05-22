from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score


"""
	Utility functions for evaluating the model performance
"""


def eval_deep(log, loader):
	"""
	Evaluating the classification performance given mini-batch data
	"""

	# get the empirical batch_size for each mini-batch
	data_size = len(loader.dataset.indices)
	batch_size = loader.batch_size
	if data_size % batch_size == 0:
		size_list = [batch_size] * (data_size//batch_size)
	else:
		size_list = [batch_size] * (data_size // batch_size) + [data_size % batch_size]

	assert len(log) == len(size_list)

	accuracy, f1, precision, recall = 0, 0, 0, 0

	prob_log, label_log = [], []

	for batch, size in zip(log, size_list):
		pred_y, y = batch[0].tolist(), batch[1].tolist()
		# prob_log.extend(batch[0][:, 1].tolist())
		# label_log.extend(y)

		# pred_y = list(map(lambda val: 1 - val, pred_y))
		# y = list(map(lambda val: 1 - val, y))

		accuracy += accuracy_score(y, pred_y) * size
		# f1_macro += f1_score(y, pred_y, average='macro') * size
		# f1_micro += f1_score(y, pred_y, average='micro') * size
		f1 += f1_score(y, pred_y, average='binary') * size
		precision += precision_score(y, pred_y, zero_division=0) * size
		recall += recall_score(y, pred_y, zero_division=0) * size

	# auc = roc_auc_score(label_log, prob_log)
	# ap = average_precision_score(label_log, prob_log)

	return accuracy/data_size, f1/data_size, precision/data_size, recall/data_size
