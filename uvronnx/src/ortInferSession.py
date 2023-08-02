# -*- coding:utf-8 -*-
# @FileName  :ortInferSession.py
# @Time      :2023/8/3 00:20
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com

from pathlib import Path
from typing import List, Union
from uvronnx.src.utils.logger import logger
import numpy as np
from onnxruntime import (GraphOptimizationLevel, InferenceSession,
                         SessionOptions, get_available_providers, get_device)


class UVROrtInferSession:
    def __init__(self, config):
        sess_opt = SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = False
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        cuda_ep = "CUDAExecutionProvider"
        cpu_ep = "CPUExecutionProvider"
        cpu_provider_options = {
            "arena_extend_strategy": "kSameAsRequested",
        }

        EP_list = []
        if (
            config["use_cuda"]
            and get_device() == "GPU"
            and cuda_ep in get_available_providers()
        ):
            EP_list = [(cuda_ep, config[cuda_ep])]
        EP_list.append((cpu_ep, cpu_provider_options))

        self._verify_model(config["model_path"])
        logger.info(f"Loading onnx model at {str(config['model_path'])}")
        self.session = InferenceSession(
            str(config["model_path"]), sess_options=sess_opt, providers=EP_list
        )

        if config["use_cuda"] and cuda_ep not in self.session.get_providers():
            logger.warning(
                f"{cuda_ep} is not available for current env, "
                f"the inference part is automatically shifted to be "
                f"executed under {cpu_ep}.\n "
                "Please ensure the installed onnxruntime-gpu version"
                " matches your cuda and cudnn version, "
                "you can check their relations from the offical web site: "
                "https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html",
                RuntimeWarning,
            )

    def __call__(
        self, input_chunk: np.ndarray
    ) -> np.ndarray:

        input_dict = {
            "input": input_chunk,
        }

        return self.session.run(None, input_dict)[0]

    def get_input_names(
        self,
    ):
        return [v.name for v in self.session.get_inputs()]

    def get_output_names(
        self,
    ):
        return [v.name for v in self.session.get_outputs()]

    def get_character_list(self, key: str = "character"):
        return self.meta_dict[key].splitlines()

    def have_key(self, key: str = "character") -> bool:
        self.meta_dict = self.session.get_modelmeta().custom_metadata_map
        if key in self.meta_dict.keys():
            return True
        return False

    @staticmethod
    def _verify_model(model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exists.")
        if not model_path.is_file():
            raise FileExistsError(f"{model_path} is not a file.")