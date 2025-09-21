# Enhanced Small Traffic Sign Detection under Adverse Weather: The ESA-DETR Algorithm.
Official PyTorch implementation of ESA-DETR.
[Enhanced Small Traffic Sign Detection under Adverse Weather: The ESA-DETR Algorithm] Qiang Li, Jinxia Yu, Jie Wu, Mingfu Zhu and Yang Chen

<details>
  <summary>Abstract</summary>
  Under adverse weather conditions, detecting small traffic signs accurately is crucial for autonomous driving safety. This paper introduces ESA-DETR, an optimized algorithm designed to enhance detection accuracy and reliability. Leveraging an edge fusion block for boundary structure perception, a structure-enhanced feature fusion module for efficient multi-level feature integration, and an adaptive sparse feature interaction module for practical information focus, ESA-DETR achieves significant improvements. Experimental results on the TT100K and CCTSDB datasets demonstrate a mAP\(_{50}\)  increase of 2.4\% and 1.7\%, respectively, compared to baseline algorithms, with a 27\% reduction in model parameters and a 15\% decrease in computational load. These findings underscore the potential of ESA-DETR in real-world autonomous driving applications.
</details>

# DataSets
TT100K(http://cg.cs.tsinghua.edu.cn/traffic-sign)

# Installation

```python
conda create -n ESADETR python=3.10   
conda activate ESADETR 
pip install -r requirements.txt
```

# Performance


Model  | Test Size  | #Params | mAP50(%)
 ---- | ----- | ------  
 Faster R-CNN  | 640 | 165  | 78.4
 YOLOv5m       | 640 | 25.1 | 83.5
 YOLOv8m       | 640 | 25.8 | 83.9
 YOLOv10m      | 640 | 16.5 | 83.3
 RT-DETR-r18   | 640 | 19.9 | 83.8
 ESA-DETR (ours)| 640 | **15.1** | **86.2**


# Training
`python train.py`
# Validation
`python val.py`
# Testing
`python test.py`
# Acknowledgement
The code base is built with [ultralytics].
Thanks for the great implementations!

