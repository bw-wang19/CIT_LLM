#
# @file /workspace/acl/code/utils/algs/LM_Cocktail/__init__.py
#
# recurrence from $LM-Cocktail: Resilient Tuning of Language Models via Model Merging$(ACL 2024 findings)
#
# original paper on https://arxiv.org/abs/2311.13534, github: https://github.com/FlagOpen/FlagEmbedding
	
from .cocktail import mix_models, mix_models_with_data, mix_models_by_layers