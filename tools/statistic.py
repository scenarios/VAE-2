from os import path as osPath
import glob

import numpy as np

import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of vp")
parser.add_argument('--root', type=str)
parser.add_argument('--name', default=None, type=str)
parser.add_argument('--candidate', default=None, type=str)


def stat_meanvar_cityscapes(root, candidate):
    meanvarResult = {'0_msssimloss': [], '0_psnrloss': [], '0_ssimloss': [], '0_reconloss': [],
                     '1_msssimloss': [], '1_psnrloss': [], '1_ssimloss': [], '1_reconloss': [],
                     '2_msssimloss': [], '2_psnrloss': [], '2_ssimloss': [], '2_reconloss': []}
    videoList = glob.glob(osPath.join(root,'*'))
    for v in videoList:
        for c in candidate:
            curPath = osPath.join(root, v, c+'predict')
            for k in meanvarResult.keys():
                with open(osPath.join(curPath, '_'.join([c, k])+'.txt')) as f:
                    val = np.loadtxt(f, usecols=(0,), unpack=True)
                    mean = np.mean(val)
                    std = np.std(val)
                    meanvarResult[k].append((mean, std))
        print("{}: mean {}, std {}".format(v, mean, std))
    for k in meanvarResult.keys():
        mean = sum(val[0] for val in meanvarResult[k]) / len(meanvarResult[k])
        std = sum(val[1] for val in meanvarResult[k]) / len(meanvarResult[k])
        meanvarResult[k]=(mean, std)

    return meanvarResult


def stat_bestsample(root, candidate, points):
    bestsampleResult = {'0_msssimloss': {}, '0_psnrloss': {}, '0_ssimloss': {}, '0_reconloss':{},
                     '1_msssimloss': {}, '1_psnrloss': {}, '1_ssimloss': {}, '1_reconloss': {},
                     '2_msssimloss': {}, '2_psnrloss': {}, '2_ssimloss': {}, '2_reconloss': {}}
    for k in bestsampleResult.keys():
        for p in points:
            bestsampleResult[k][str(p)] = []

    videoList = glob.glob(osPath.join(root, '*'))
    for v in videoList:
        for c in candidate:
            curPath = osPath.join(root, v, c + 'predict')
            for k in bestsampleResult.keys():
                with open(osPath.join(curPath, '_'.join([c, k]) + '.txt')) as f:
                    val = np.loadtxt(f, usecols=(0,), unpack=True)
                    for p in points:
                        bestSample = min(val[0:p]) if 'reconloss' in k else max(val[0:p])
                        bestsampleResult[k][str(p)].append(bestSample)

    for k in bestsampleResult.keys():
        for p in points:
            bestsampleResult[k][str(p)] = sum(bestsampleResult[k][str(p)]) / len(bestsampleResult[k][str(p)])

    return bestsampleResult


def stat_meanvar_numpredict(root, candidate):
    meanvarResult = {'0_msssimloss': [], '0_psnrloss': [], '0_ssimloss': [], '0_reconloss': [],
                     '1_msssimloss': [], '1_psnrloss': [], '1_ssimloss': [], '1_reconloss': [],
                     '2_msssimloss': [], '2_psnrloss': [], '2_ssimloss': [], '2_reconloss': []}
    numList = glob.glob(osPath.join(root,'*'))
    ms = []
    stds = []
    numList.sort()
    numList = numList[0:40]
    print(numList)
    for v in numList:
        for c in candidate:
            curPath = osPath.join(root, v, c+'_axis.txt')
            curGtPath = osPath.join(root, v, 'gt_axis.txt')
            with open(curPath) as f:
                val = np.loadtxt(f, unpack=True)
            with open(curGtPath) as f:
                gtval = np.expand_dims(np.loadtxt(f, unpack=True), axis=1)
            l1_val = np.abs(val - gtval)
            m = np.mean(l1_val)
            std = np.mean(np.std(l1_val, axis=1))
            ms.append(m)
            stds.append(std)
    ms = sum(ms) / len(ms)
    stds = sum(stds) / len(stds)

    return {'l1_mean': ms, 'l1_standardDeviation': stds}


def stat_flow_std(root):
    import numpy as np
    from PIL import Image
    itemList = glob.glob(osPath.join(root, '*'))
    flowstds = []
    flow_max = 0
    for item in itemList:
        flowList = glob.glob(osPath.join(root, item, '*'))
        flowImages = []
        for s in flowList:
            flowImage = np.asarray(Image.open(s).convert('RGB').resize((256, 128)), dtype=np.float32)
            flowImages.append(np.expand_dims(flowImage, axis=0))
        flowImages = np.concatenate(flowImages, axis=0)
        std = np.mean(np.std(flowImages, axis=0), axis = 2)
        std_val = np.mean(std)
        flow_max = max(flow_max, np.max(std))
        flowstds.append((osPath.join(root, item, 'flowstd.jpg'), std))
        with open(osPath.join(root, item, 'std.txt'), 'w') as f:
            print(std_val, file=f)
    for pth, std in flowstds:
        std = std / flow_max * 255
        stdImg = Image.fromarray(std.astype(np.uint8), mode='L')
        stdImg.save(pth)

def main():
    args = parser.parse_args()

    #results = stat_meanvar_cityscapes(osPath.join(args.root, 'epoch0'), [args.candidate])
    #results = stat_meanvar_numpredict(osPath.join(args.root, 'epoch0'), [args.candidate])
    #results = stat_bestsample(osPath.join(args.root, 'epoch0'), [args.candidate], points=[1,3,5,20,50,100])
    #with open(osPath.join(args.root, args.name+'_bestsample.txt'), 'w') as f:
    #    print(results, file=f)
    stat_flow_std(args.root)

if __name__ == '__main__':
    main()