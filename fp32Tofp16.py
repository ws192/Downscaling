import numpy

fp32Arr = numpy.load(r'D:\lla.npy')
fp16Arr = fp32Arr.astype(numpy.float16)
numpy.save(r'D:\llafp16.npy', fp16Arr)
