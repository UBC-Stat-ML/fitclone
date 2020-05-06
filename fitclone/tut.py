import os
import revive_conditional.py
docs = yaml.load_all(open('/Users/sohrabsalehi/projects/fitness/batch_runs/sa501_config_202005-05-194506.751013/yaml/param_chunk_0.yaml', 'r'))
for doc in docs:
	res = run_model_comparison_ne(doc, '/Users/sohrabsalehi/projects/fitness/batch_runs/sa501_config_202005-05-194506.751013')
	
