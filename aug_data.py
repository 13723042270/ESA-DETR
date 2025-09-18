import os
import cv2
import random
import shutil
import albumentations as A
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


# ==================== 配置参数 ====================
class Config:
    # 路径配置
    ORIGINAL_IMAGES_DIR = "/root/autodl-tmp/datasets/TTK110/images/train"
    ORIGINAL_LABELS_DIR = "/root/autodl-tmp/datasets/TTK110/labels/train"
    AUG_IMAGES_DIR = "/root/autodl-tmp/datasets/TTK110/images/train_aug"
    AUG_LABELS_DIR = "/root/autodl-tmp/datasets/TTK110/labels/train_aug"
    VISUALIZATION_DIR = "/root/autodl-tmp/results"  # 新增可视化目录
    # 增强参数
    AUG_RATIO = 0.2  # 增强比例
    WEATHER_DISTRIBUTION = {
        "rain": 0.33,
        "fog": 0.34,  # 增加雾的比例
        "snow": 0.33
    }

    # 雾效专用参数
    FOG_PARAMS = {
        "fog_range": (0.3, 0.6),  # 雾浓度范围
        "alpha": 0.07,  # 透明度
        "brightness": (-0.25, -0.15),  # 亮度调整
        "rgb_shift": ((-20, 0), (-15, 0), (10, 20)),  # 颜色偏移
        "blur_limit": (1, 3),  # 模糊程度
        "noise_coef": (0.003, 0.004)  # 颗粒密度
    }

    # 验证参数
    VALIDATE_SAMPLES = 5
    SHOW_ANALYSIS = True
    SHOW_COMPARISON = True  # 添加这个属性
    VISUALIZE_AUG = True    # 新增可视化开关

# ==================== 天气增强器 ====================
class WeatherAugmentor:
    def __init__(self, config):
        self.cfg = config
        self._setup_dirs()
        self.transforms = self._build_transforms()
        self.classes = self._load_classes()  # 假设标签类别存储在配置或文件中
    def _setup_dirs(self):
        """创建输出目录"""
        os.makedirs(self.cfg.AUG_IMAGES_DIR, exist_ok=True)
        os.makedirs(self.cfg.AUG_LABELS_DIR, exist_ok=True)
        os.makedirs(self.cfg.VISUALIZATION_DIR, exist_ok=True)  # 新增可视化目录

    def _load_classes(self):
        """从配置或文件中加载类别列表（需根据实际情况修改）"""
        return ['pl80', 'p6', 'p5', 'pm55', 'pl60', 'ip', 'p11', 'i2r', 'p23', 'pg',
                'il80', 'ph4', 'i4', 'pl70', 'pne', 'ph4.5', 'p12', 'p3', 'pl5',
                'w13', 'i4l', 'pl30', 'p10', 'pn', 'w55', 'p26', 'p13', 'pr40',
                'pl20', 'pm30', 'pl40', 'i2', 'pl120', 'w32', 'ph5', 'il60',
                'w57', 'pl100', 'w59', 'il100', 'p19', 'pm20', 'i5', 'p27', 'pl50']

    def _build_transforms(self):
        """构建增强管道"""
        return {
            "rain": A.Compose([
                A.RandomRain(
                    brightness_coefficient=0.7,
                    drop_width=1,
                    blur_value=2,
                    p=1
                )
            ]),  # 雨效保持原样

            "fog": A.Compose([
                # 基础雾效层
                A.RandomFog(
                    fog_coef_lower=self.cfg.FOG_PARAMS["fog_range"][0],
                    fog_coef_upper=self.cfg.FOG_PARAMS["fog_range"][1],
                    alpha_coef=self.cfg.FOG_PARAMS["alpha"],
                    p=1
                ),
                # 光线调整层
                A.RandomBrightnessContrast(
                    brightness_limit=self.cfg.FOG_PARAMS["brightness"],
                    contrast_limit=(-0.2, 0.1),
                    p=0.9
                ),
                # 颜色偏移层
                A.RGBShift(
                    r_shift_limit=self.cfg.FOG_PARAMS["rgb_shift"][0],
                    g_shift_limit=self.cfg.FOG_PARAMS["rgb_shift"][1],
                    b_shift_limit=self.cfg.FOG_PARAMS["rgb_shift"][2],
                    p=0.8
                ),
                # 颗粒感层
                A.AdvancedBlur(
                    blur_limit=self.cfg.FOG_PARAMS["blur_limit"],
                    p=0.7
                ),
                # 添加噪声
                # A.GaussNoise(
                #     var_limit=(self.cfg.FOG_PARAMS["noise_coef"][0] * 100,
                #                self.cfg.FOG_PARAMS["noise_coef"][1] * 100),
                #     p=0.5
                # ),
                # 细节保留层
                A.Sharpen(
                    alpha=(0.1, 0.2),
                    lightness=(0.9, 1.1),
                    p=0.6
                )
            ], p=1),

            "snow": A.Compose([
                A.RandomSnow(
                    snow_point_lower=0.1,
                    snow_point_upper=0.3,
                    brightness_coeff=2.0,
                    p=1
                )
            ])  # 雪效保持原样
        }

    def run(self):
        """执行增强流程"""
        image_files = self._get_image_list()
        selected_files = self._select_files(image_files)

        weather_counts = {k: 0 for k in self.cfg.WEATHER_DISTRIBUTION}
        error_count = 0

        for idx, fname in enumerate(tqdm(selected_files, desc="Processing")):
            try:
                weather = self._select_weather()
                img_path = os.path.join(self.cfg.ORIGINAL_IMAGES_DIR, fname)
                label_path = os.path.join(self.cfg.ORIGINAL_LABELS_DIR,
                                          fname.rsplit('.', 1)[0] + '.txt')

                # 读取并增强
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                aug = self.transforms[weather](image=img)

                # 保存结果
                new_name = f"{fname.rsplit('.', 1)[0]}_{weather}_{idx}.jpg"
                self._save_data(aug["image"], label_path, new_name)

                # 记录雾效分析
                if weather == "fog" and self.cfg.SHOW_ANALYSIS:
                    self._analyze_fog_effect(img, aug["image"])

                weather_counts[weather] += 1

                # 可视化增强结果（整合show_labels功能）
                if self.cfg.VISUALIZE_AUG:
                    self._visualize_single(aug["image"], label_path, new_name)

            except Exception as e:
                error_count += 1
                print(f"Error processing {fname}: {str(e)}")

        # 打印报告
        print("\n=== 增强结果 ===")
        print(f"成功率:  ({len(selected_files) - error_count} / {len(selected_files)})")
        for k, v in weather_counts.items():
            print(f"{k}: {v} 张")

        # 执行验证
        Validator(self.cfg).run()

    def _visualize_single(self, image, src_label, new_name):
        """单样本可视化（整合原show_labels功能）"""
        labels = []
        if os.path.exists(src_label):
            with open(src_label) as f:
                labels = [line.strip().split() for line in f]

        h, w = image.shape[:2]
        for label in labels:
            try:
                class_id = int(float(label[0]))  # 转换为整数ID
                xc, yc, bw, bh = map(float, label[1:])
            except (ValueError, IndexError):
                print(f"无效标签格式: {label}")
                continue
            if class_id < 0 or class_id >= len(self.classes):
                class_name = "Unknown"
            else:
                class_name = self.classes[class_id]

            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            # 绘制边框和具体类别名称
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        # 保存可视化结果
        vis_path = os.path.join(self.cfg.VISUALIZATION_DIR, new_name)
        cv2.imwrite(vis_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"可视化结果保存至: {vis_path}")

    def _get_image_list(self):
        """获取可用的图像文件列表"""
        all_files = os.listdir(self.cfg.ORIGINAL_IMAGES_DIR)
        valid_files = [f for f in all_files
                       if os.path.isfile(os.path.join(self.cfg.ORIGINAL_IMAGES_DIR, f))
                       and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        return valid_files

    def _select_files(self, files):
        """随机选择要增强的文件"""
        random.shuffle(files)
        return files[:int(len(files) * self.cfg.AUG_RATIO)]

    def _select_weather(self):
        """随机选择天气类型"""
        return random.choices(
            population=list(self.cfg.WEATHER_DISTRIBUTION.keys()),
            weights=list(self.cfg.WEATHER_DISTRIBUTION.values()),
            k=1
        )[0]

    def _save_data(self, image, src_label, new_name):
        """保存增强结果"""
        # 直接保存增强后的图像（不带标注框）
        cv2.imwrite(
            os.path.join(self.cfg.AUG_IMAGES_DIR, new_name),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )
        # 复制标签文件
        shutil.copy(
            src_label,
            os.path.join(self.cfg.AUG_LABELS_DIR,
                         os.path.splitext(new_name)[0] + ".txt")
        )

    def _analyze_fog_effect(self, orig_img, aug_img):
        """雾效量化分析"""
        orig_lab = cv2.cvtColor(orig_img, cv2.COLOR_RGB2LAB)
        aug_lab = cv2.cvtColor(aug_img, cv2.COLOR_RGB2LAB)

        # 计算关键指标
        metrics = {
            "亮度降幅": orig_lab[..., 0].mean() - aug_lab[..., 0].mean(),
            "对比度比": aug_lab[..., 0].std() / orig_lab[..., 0].std(),
            "色度偏移A": aug_lab[..., 1].mean() - orig_lab[..., 1].mean(),
            "色度偏移B": aug_lab[..., 2].mean() - orig_lab[..., 2].mean(),
            "边缘保留率": cv2.Laplacian(aug_lab[..., 0], cv2.CV_64F).var() /
                          cv2.Laplacian(orig_lab[..., 0], cv2.CV_64F).var()
        }

        # 可视化对比
        plt.figure(figsize=(15, 6))
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR))
        plt.title("Original")

        plt.subplot(122)
        plt.imshow(cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
        plt.title("Fog Augmented")
        plt.show()
        # 关闭图形窗口
        plt.close()


# ==================== 验证模块 ====================
class Validator:
    def __init__(self, config):
        self.cfg = config

    def run(self):
        """执行完整验证流程"""
        print("\n=== 数据验证开始 ===")
        self._check_label_matching()
        self._validate_coordinates()
        self._visualize_samples()
        if self.cfg.SHOW_COMPARISON:
            self._show_comparison()

    def _check_label_matching(self):
        """检查标签匹配"""
        missing = []
        for img in os.listdir(self.cfg.AUG_IMAGES_DIR):
            if os.path.isfile(os.path.join(self.cfg.AUG_IMAGES_DIR, img)):  # 先判断是否为文件
                label = os.path.splitext(img)[0] + ".txt"
                if not os.path.exists(os.path.join(self.cfg.AUG_LABELS_DIR, label)):
                    missing.append(img)

        if missing:
            print(f"\033[31m警告\033[0m: 发现{len(missing)}个缺失标签")
            print("前5个缺失样本:")
            for m in missing[:5]:
                print(f" - {m}")
        else:
            print("✓ 所有图片都有对应的标签文件")

    def _validate_coordinates(self):
        """验证坐标有效性"""
        invalid = []
        for label_file in os.listdir(self.cfg.AUG_LABELS_DIR):
            path = os.path.join(self.cfg.AUG_LABELS_DIR, label_file)
            with open(path) as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        values = list(map(float, line.strip().split()))
                        if len(values) != 5 or any(not (0 <= v <= 1) for v in values[1:]):
                            invalid.append((label_file, line_num))
                    except:
                        invalid.append((label_file, line_num))

        if invalid:
            print(f"\033[31m警告\033[0m: 发现{len(invalid)}处无效坐标")
            print("前5个问题坐标:")
            for item in invalid[:5]:
                print(f"文件 {item[0]} 第 {item[1]} 行")
        else:
            print("✓ 所有坐标值有效")

    def _visualize_samples(self):
        """可视化随机样本"""
        samples = random.sample(os.listdir(self.cfg.AUG_IMAGES_DIR),
                                self.cfg.VALIDATE_SAMPLES)

        for img_file in samples:
            img_path = os.path.join(self.cfg.AUG_IMAGES_DIR, img_file)
            label_path = os.path.join(self.cfg.AUG_LABELS_DIR,
                                      os.path.splitext(img_file)[0] + ".txt")

            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            labels = []

            if os.path.exists(label_path):
                with open(label_path) as f:
                    labels = [line.strip().split() for line in f]

            plt.figure(figsize=(12, 8))
            plt.imshow(image)
            plt.title(f"验证样本: {img_file}")

            # 绘制标注框
            h, w = image.shape[:2]
            for label in labels:
                class_id, xc, yc, bw, bh = map(float, label)
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                rect = plt.Rectangle((x1, y1), int(bw * w), int(bh * h),
                                     linewidth=2, edgecolor='lime', facecolor='none')
                plt.gca().add_patch(rect)
                plt.text(x1, y1 - 10, f'Class {int(class_id)}',
                         color='lime', fontsize=12, backgroundcolor='black')

            plt.axis('off')
            plt.show()
            # 关闭图形窗口
            plt.close()

    def _show_comparison(self):
        """显示增强前后对比"""
        aug_images = [f for f in os.listdir(self.cfg.AUG_IMAGES_DIR)
                      if "_rain_" in f or "_fog_" in f or "_snow_" in f]
        samples = random.sample(aug_images, min(3, len(aug_images)))

        for sample in samples:
            # 解析原始文件名
            orig_name = "_".join(sample.split("_")[:-2]) + ".jpg"
            orig_path = os.path.join(self.cfg.ORIGINAL_IMAGES_DIR, orig_name)

            if not os.path.exists(orig_path):
                continue

            fig, ax = plt.subplots(1, 2, figsize=(20, 10))

            # 显示原始图像
            orig_img = cv2.cvtColor(cv2.imread(orig_path), cv2.COLOR_BGR2RGB)
            ax[0].imshow(orig_img)
            ax[0].set_title(f"原始\n{orig_name}")

            # 显示增强图像
            aug_img = cv2.cvtColor(cv2.imread(os.path.join(
                self.cfg.AUG_IMAGES_DIR, sample)), cv2.COLOR_BGR2RGB)
            ax[1].imshow(aug_img)
            ax[1].set_title(f"增强\n{sample}")

            for a in ax:
                a.axis('off')

            plt.tight_layout()
            plt.show()
            # 关闭图形窗口
            plt.close()

    def validate_fog_quality(self, img):
        """雾效质量验证"""
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        # 理想指标范围
        return {
            "亮度(L)": (80 < lab[:, :, 0].mean() < 120),
            "色度A": (125 < lab[:, :, 1].mean() < 135),
            "色度B": (130 < lab[:, :, 2].mean() < 145),
            "边缘锐度": cv2.Laplacian(lab[:, :, 0], cv2.CV_64F).var() > 200
        }


# ==================== 主程序 ====================
if __name__ == "__main__":
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    config = Config()
    augmentor = WeatherAugmentor(config)
    augmentor.run()
