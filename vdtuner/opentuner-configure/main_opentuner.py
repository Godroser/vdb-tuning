import argparse
import logging
import sys
import time
import subprocess as sp
import json
import pty  # [新增] 用于伪终端控制
import os   # [新增] 用于底层IO读取

import opentuner
from opentuner import ConfigurationManipulator
from opentuner import IntegerParameter
from opentuner import EnumParameter
from opentuner import MeasurementInterface
from opentuner import Result

sys.path.append("/home/dyx/VDTuner/auto-configure")
from configure import filter_index_rule, configure_index, filter_system_rule, configure_system

log = logging.getLogger(__name__)

class VDTunerInterface(MeasurementInterface):
    def __init__(self, args):
        super(VDTunerInterface, self).__init__(args)
        self.run_engine_path = '/home/dyx/VDTuner/vector-db-benchmark-master/run_engine_test.sh'

    def manipulator(self):
        """
        Define the search space
        """
        manipulator = ConfigurationManipulator()
        
        # Index Parameters
        manipulator.add_parameter(EnumParameter('index_type', [
            'FLAT', 'IVF_FLAT', 'IVF_SQ8', 'IVF_PQ', 'HNSW', 'SCANN', 'AUTOINDEX'
        ]))
        manipulator.add_parameter(IntegerParameter('nlist', 1, 10000))
        manipulator.add_parameter(IntegerParameter('nprobe', 1, 100))
        manipulator.add_parameter(EnumParameter('m', [1, 2, 4, 5, 10, 20, 25, 50, 100]))
        manipulator.add_parameter(IntegerParameter('nbits', 1, 10))
        manipulator.add_parameter(IntegerParameter('M', 4, 64))
        manipulator.add_parameter(IntegerParameter('efConstruction', 8, 512))
        manipulator.add_parameter(IntegerParameter('ef', 100, 1000))
        manipulator.add_parameter(IntegerParameter('reorder_k', 100, 1000))

        # System Parameters
        manipulator.add_parameter(IntegerParameter('dataCoord*segment*maxSize', 100, 10000))
        manipulator.add_parameter(IntegerParameter('dataCoord*segment*sealProportion', 1, 99))
        manipulator.add_parameter(EnumParameter('queryCoord*autoHandoff', [True, False]))
        manipulator.add_parameter(EnumParameter('queryCoord*autoBalance', [True, False]))
        manipulator.add_parameter(IntegerParameter('common*gracefulTime', 100, 100000))
        manipulator.add_parameter(IntegerParameter('dataNode*segment*insertBufSize', 1000, 10000000000))
        manipulator.add_parameter(IntegerParameter('rootCoord*minSegmentSizeToEnableIndex', 100, 10000))

        return manipulator

    def run(self, desired_result, input, limit):
        """
        Run the benchmark with the given configuration
        """
        cfg = desired_result.configuration.data
        
        # Separate index and system parameters
        index_keys = ['index_type', 'nlist', 'nprobe', 'm', 'nbits', 'M', 'efConstruction', 'ef', 'reorder_k']
        
        index_conf = {k: cfg[k] for k in index_keys if k in cfg}
        system_conf = {k: cfg[k] for k in cfg if k not in index_keys}

        # Apply configurations
        print(f"[OpenTuner] Applying config: {index_conf} {system_conf}")
        configure_index(*filter_index_rule(index_conf))
        configure_system(filter_system_rule(system_conf))

        # Run the benchmark
        run_cmd = f'timeout 900 {self.run_engine_path} "" "" glove-100-angular'
        print(f"[OpenTuner] Running benchmark command: {run_cmd}")
        
        t1 = time.time()
        try:
            # === [修改开始] 使用 pty 伪终端来保留进度条动画 ===
            master, slave = pty.openpty()
            
            # 使用 slave 作为子进程的 stdout/stderr，这样子进程会认为它是 TTY
            # close_fds=True 确保除了 std 句柄外不传递其他句柄
            process = sp.Popen(run_cmd, shell=True, stdout=slave, stderr=slave, close_fds=True)
            
            # 父进程已经拿到了 master 句柄，不需要 slave 句柄了，关闭它以免死锁
            os.close(slave) 

            stdout_lines = []
            
            while True:
                try:
                    # 从 master 读取数据，每次读 1024 字节
                    # os.read 是阻塞的，直到有输出或者 EOF
                    output = os.read(master, 1024)
                    if not output:
                        break
                except OSError:
                    # 子进程结束时可能会抛出 I/O 错误（Input/output error），跳出循环
                    break
                
                # 解码并实时打印
                # end='' 防止 print 自动加换行符
                # flush=True 强制刷新缓冲区，确保进度条立刻显示
                text = output.decode('utf-8', errors='replace')
                print(text, end='', flush=True)
                stdout_lines.append(text)
            
            # 等待进程完全结束
            process.wait()
            # === [修改结束] ===
            
            stdout_str = "".join(stdout_lines)
            
            # 解析逻辑保持不变
            result_lines = stdout_str.split()
            metrics = []
            for val in reversed(result_lines):
                try:
                    metrics.append(float(val.strip(',')))
                    if len(metrics) == 3:
                        break
                except ValueError:
                    continue
            
            if len(metrics) == 3:
                # metrics are in reverse order: p95_time, rps, mean_precisions
                p95_time = metrics[0]
                rps = metrics[1]
                precision = metrics[2]
                
                print(f"\n[OpenTuner] Round Result: RPS={rps}, Precision={precision}, P95_Time={p95_time}\n")
                
                if precision < 0.9:
                     return Result(time=float('inf'), accuracy=precision) # Treat as invalid/poor

                # Return -RPS as time (since OpenTuner minimizes time/value)
                return Result(time=-rps, accuracy=precision)
            else:
                return Result(time=float('inf'), state='ERROR', msg="Output parse failed")

        except Exception as e:
             return Result(time=float('inf'), state='ERROR', msg=str(e))

if __name__ == '__main__':
    argparser = opentuner.default_argparser()
    args = argparser.parse_args()
    
    VDTunerInterface.main(args)