
def predict(args):
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model(args["model"])
    stride = args['stride']
    for n in range(len(TEST_SET)):
        path = TEST_SET[n]
        # load the image
        image = cv2.imread('./test/' + path)
        # pre-process the image for classification
        # image = image.astype("float") / 255.0
        # image = img_to_array(image)
        h, w, _ = image.shape
        padding_h = (h // stride + 1) * stride
        padding_w = (w // stride + 1) * stride
        padding_img = np.zeros((padding_h, padding_w, 3), dtype=np.uint8)
        padding_img[0:h, 0:w, :] = image[:, :, :]
        padding_img = padding_img.astype("float") / 255.0
        padding_img = img_to_array(padding_img)
        print
        'src:', padding_img.shape
        mask_whole = np.zeros((padding_h, padding_w), dtype=np.uint8)
        for i in range(padding_h // stride):
            for j in range(padding_w // stride):
                crop = padding_img[:3, i * stride:i * stride + image_size, j * stride:j * stride + image_size]
                _, ch, cw = crop.shape
                if ch != 256 or cw != 256:
                    print
                    'invalid size!'
                    continue

                crop = np.expand_dims(crop, axis=0)
                # print 'crop:',crop.shape
                pred = model.predict_classes(crop, verbose=2)
                pred = labelencoder.inverse_transform(pred[0])
                # print (np.unique(pred))
                pred = pred.reshape((256, 256)).astype(np.uint8)
                # print 'pred:',pred.shape
                mask_whole[i * stride:i * stride + image_size, j * stride:j * stride + image_size] = pred[:, :]

        cv2.imwrite('./predict/pre' + str(n + 1) + '.png', mask_whole[0:h, 0:w])