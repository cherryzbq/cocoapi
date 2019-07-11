import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval_zbq import COCOeval_zbq
import numpy as np
import skimage.io as io
import pylab
import json
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
annType = ['segm','bbox','keypoints']
annType = annType[2]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
print('Running demo for *%s* results.'%(annType))
#initialize COCO ground truth api
dataDir='/home/zhubq/datasets/coco'
dataType='val2017' #!!!!!!!!!!!!!!!!!!!!!!!!!
annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
cocoGt=COCO(annFile)
#initialize COCO detections api
resFile='/home/zhubq/%s_%s_results.json'
resFile = resFile%(prefix, dataType)
cocoDt=cocoGt.loadRes(resFile)
# Next line in the demo ipyntbk

imgIds=sorted(cocoGt.getImgIds())
imgIds=imgIds[0:100]
imgId = imgIds[np.random.randint(100)]
# running evaluation
cocoEval = COCOeval_zbq(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
OKS_rough = cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
with open('oks_rough_coco_0612.json','w') as f:
	json.dump(OKS_rough)