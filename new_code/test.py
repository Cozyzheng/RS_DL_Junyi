from RSjunyi import rs

img = rs.img_read('/fastdata/deeplearning/原始数据/train2.png', info=False)
label = rs.img_read('/fastdata/deeplearning/原始数据/train2_labels_8bits.png', info=False)
rs.img_clip_ramdon(img, label, 3000, img_w = 256, img_h = 256, agument=True)