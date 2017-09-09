D:\caffe\caffe-master\Build\x64\Debug\convert_imageset.exe --resize_height=256 --resize_width=256 --shuffle --backend="lmdb" train/ train/list.txt trainlmdb
D:\caffe\caffe-master\Build\x64\Debug\convert_imageset.exe --resize_height=256 --resize_width=256 --shuffle --backend="lmdb" test/ test/list.txt testlmdb
D:\caffe\caffe-master\Build\x64\Debug\compute_image_mean.exe trainlmdb image_mean.binaryproto
pause 