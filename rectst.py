import mxnet as mx

record = mx.recordio.MXRecordIO('/tmp/mult.rec', 'r')
for i in range(10):
    header, img = mx.recordio.unpack_img(record.read())
    print(header)


data = mx.io.ImageRecordIter(
    path_imgrec="/tmp/mult.rec",
    # path_imglst="/tmp/mult.lst",
    label_width=2,
    data_name='data',
    label_name='softmax_label',
    batch_size=8,
    data_shape=(3, 224, 224))

batch = data.next()
print(batch.index)
print(batch.label)
# print(data.next().index)

data = mx.io.ImageRecordIter(
    path_imgrec="imagenet/train_256_q90.rec",
    label_width=1,
    data_name='data',
    label_name='softmax_label',
    batch_size=8,
    data_shape=(3, 224, 224))

batch = data.next()
print(batch.index)
print(batch.label)
