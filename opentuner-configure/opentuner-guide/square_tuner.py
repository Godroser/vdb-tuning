#!/usr/bin/env python
import opentuner
from opentuner import ConfigurationManipulator
from opentuner import IntegerParameter
from opentuner import MeasurementInterface
from opentuner import Result

class SquareTuner(MeasurementInterface):

  def manipulator(self):
    """
    定义搜索空间。
    """
    manipulator = ConfigurationManipulator()
    manipulator.add_parameter(
      IntegerParameter('x', 0, 10))
    return manipulator

  def run(self, desired_result, input, limit):
    """
    运行一个给定的配置并返回性能。
    """
    cfg = desired_result.configuration.data
    x_value = cfg['x']

    # 一个简单的目标函数，OpenTuner会尝试最小化它
    objective_value = (x_value - 5)**2

    return Result(time=objective_value)

  def save_final_config(self, configuration):
    """在调优结束时调用"""
    print("Found optimal configuration:", configuration.data)

if __name__ == '__main__':
  argparser = opentuner.default_argparser()
  SquareTuner.main(argparser.parse_args())