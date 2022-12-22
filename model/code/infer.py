# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json

import argparse
import math
from pprint import pprint

import paddle
from uie_predictor import UIEPredictor


def model_fn(model_dir):
    args = parse_args()
    args.model_path_prefix = os.path.join(model_dir, 'inference')
    args.device = 'cpu'
#     args.device = 'gpu'
    args.schema = ['地址', '银行卡号', '电子邮箱', '姓名', '手机号码', '电话号码', '身份证号', '护照号']
    predictor = UIEPredictor(args)
    return predictor


def input_fn(request_body, request_content_type):
#     print('[DEBUG] request_body:', type(request_body))
#     print('[DEBUG] request_content_type:', request_content_type)
    
    """An input_fn that loads a pickled tensor"""
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.  
        return request_body
    
def predict_fn(input_data, model):
    outputs = model.predict(input_data)
    return outputs


# def output_fn(prediction, content_type):
#     pass


def parse_args():
    parser = argparse.ArgumentParser()
#     # Required parameters
#     parser.add_argument(
#         "--model_path_prefix",
#         type=str,
#         required=True,
#         help="The path prefix of inference model to be used.", )
    parser.add_argument(
        "--position_prob",
        default=0.5,
        type=float,
        help="Probability threshold for start/end index probabiliry.", )
#     parser.add_argument(
#         "--use_fp16",
#         action='store_true',
#         help="Whether to use fp16 inference, only takes effect when deploying on gpu.",
#     )
    parser.add_argument(
        "--max_seq_len",
        default=512,
        type=int,
        help="The maximum input sequence length. Sequences longer than this will be split automatically.",
    )
    
    parsed, unknown = parser.parse_known_args() # this is an 'internal' method
    # which returns 'parsed', the same as what parse_args() would return
    # and 'unknown', the remainder of that
    # the difference to parse_args() is that it does not exit when it finds redundant arguments

    for arg in unknown:
        if arg.startswith(("-", "--")):
            # you can pass any arguments to add_argument
            parser.add_argument(arg, type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    texts = [
        '我叫王五，男，30岁，生活在中国北京，我的身份证号是110111199201010000，手机号是13912345678，电话号码是010-88888888，您可以给我发邮件wangwu@qq.com，我的银行卡号是6288888888888888.'
    ]
    model = model_fn('../')
    result = predict_fn(texts, model)
    print(result)
