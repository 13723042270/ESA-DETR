import os
import warnings, os

warnings.filterwarnings('ignore')
from ultralytics import RTDETR



if __name__ == '__main__':
    # model = RTDETR('/root/autodl-tmp/ESA-DETR/ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml')

    # model = RTDETR('/root/autodl-tmp/ESA-DETR/ultralytics/cfg/models/rt-detr/rtdetr-EFB.yaml')
    # model = RTDETR('/root/autodl-tmp/ESA-DETR/ultralytics/cfg/models/rt-detr/rtdetr-EFB-SEFF.yaml')
    # model = RTDETR('/root/autodl-tmp/ESA-DETR/ultralytics/cfg/models/rt-detr/rtdetr-EFB-ASFI.yaml')
    model = RTDETR('/root/autodl-tmp/ESA-DETR/ultralytics/cfg/models/rt-detr/rtdetr-EFB-ASFI-SEFF.yaml')
    # model = RTDETR('/root/autodl-tmp/ESA-DETR/ultralytics/cfg/models/rt-detr/rtdetr-SEFF.yaml')
    # model.load('') # loading pretrain weights
    # model.train(data=r'/root/autodl-tmp/datasets/TTK110/TTK110.yaml',
    model.train(data=r'/root/autodl-tmp/datasets/CCTSDB2021/CCTSDB.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=8,
                workers=8, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                # device='0,1', # 指定显卡和多卡训练参考<使用教程.md>下方常见错误和解决方案
                # resume='', # last.pt path
                project='runs/train',
                name='rtdetr-ESA',
                )

# 模拟实验对象（根据你的实际代码调整）
class Experiment:
    def flush(self):
        # 假设第一次返回 True，第二次返回 False
        if not hasattr(self, "_count"):
            self._count = 0
        self._count += 1
        return self._count < 2  # 示例：假设需要刷新两次
experiment = Experiment()
# 定义配置对象
class Config:
    shutdown = True  # 控制是否关机
cfg = Config()
if cfg.shutdown:
    # 持续刷新直到数据写入完成
    while experiment.flush():
        print("Flushing data...")
    print("All data saved. Shutting down...")
    os.system("shutdown")  # 实际执行关机

