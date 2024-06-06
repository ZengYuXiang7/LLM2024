# coding : utf-8
# Author : yuxiang Zeng

# coding : utf-8
# Author : yuxiang Zeng

import torch
import torch.nn as nn
import torch.profiler
import time
import gc


def benchmark(model, runs=1, interval=0.1, avg_between=10, device='cpu', profile_layers=False, input_shape=(1, 3, 224, 224)):
    # 指定 PyTorch 设备（CPU 或 GPU）
    device = torch.device(device)
    # 将模型加载到指定设备
    model = model.to(device)
    # 将模型设为评估模式
    model.eval()

    # 初始化随机输入
    def generate_random_input():
        return torch.randn(input_shape).to(device)

    # 函数用于运行模型
    def run_model(random_input):
        with torch.no_grad():
            model(random_input)

    # 配置 Profiling（性能分析）
    profiler = None
    if profile_layers:
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],  # 分析 CPU 和 CUDA 活动
            record_shapes=True,         # 记录张量的形状
            with_stack=True,            # 包含调用堆栈信息
            profile_memory=True         # 分析内存使用
        )

    # 运行 Benchmark
    times = []
    for r in range(runs):
        # 生成随机输入数据
        random_input = generate_random_input()
        # 记录当前垃圾收集器状态
        gcold = gc.isenabled()
        # 禁用垃圾收集器以确保准确计时
        gc.disable()
        try:
            subtimes = []
            for _ in range(avg_between):
                if profile_layers:
                    # 在性能分析器上下文中运行模型
                    with profiler:
                        s = time.time()
                        run_model(random_input)
                        subtimes.append(time.time() - s)
                    profiler.step()  # 记录本次运行的数据
                else:
                    # 直接测量推理时间
                    s = time.time()
                    run_model(random_input)
                    subtimes.append(time.time() - s)
        finally:
            # 恢复垃圾收集器状态
            if gcold:
                gc.enable()

        # 保存本次运行的推理时间数据
        times.append(subtimes)
        # 设置两次运行之间的时间间隔
        time.sleep(interval)

    # 打印 Profiling 结果
    if profile_layers and profiler is not None:
        # 导出 Chrome Trace 格式的 Profiling 数据
        profiler.export_chrome_trace("trace.json")
        # 按自 CPU 时间总和排序并打印前 n 行数据
        if device == 'cpu':
            print(profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))
        else:
            print(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))

    return times


class ExampleModel(nn.Module):
    def __init__(self):
        super(ExampleModel, self).__init__()
        self.layer1 = nn.Conv2d(3, 64, kernel_size=3)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(64 * 111 * 111, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # 使用例
    model = ExampleModel()
    # 运行 benchmark 并进行性能分析
    benchmark_times = benchmark(model, runs=10, interval=0.01, avg_between=5, device='cpu', profile_layers=True, input_shape=(1, 3, 224, 224))
    print(benchmark_times)
