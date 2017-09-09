import caffe
import numpy as np 

root = 'D:/caffe/caffe-master/examples/my_classify/'
deploy = root + 'deploy.prototxt'
caffe_model = root + 'train_iter_500.caffemodel'
img = root + '4.jpg'
label_file = root + 'label.txt'
mean_file = root + 'mean.npy'

net = caffe.Net(deploy,caffe_model,caffe.TEST)

transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))
net.blobs['data'].reshape(1,3,256,256)

im = caffe.io.load_image(img)
net.blobs['data'].data[...] = transformer.preprocess('data', im)

for layer_name,blob in net.blobs.iteritems():
	print layer_name + '\t' + str(blob.data.shape)
out = net.forward()
labels = np.loadtxt(label_file,str,delimiter = '\t')
prob = net.blobs['prob'].data[0].flatten()

print prob
order = prob.argsort()[-1]
print 'the class is:',labels[order]
