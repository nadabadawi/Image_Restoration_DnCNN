# def val_datagen(epoch_iter=10,epoch_num=5,batch_size=3,data_dir=args.val_data):

    while(True):
        n_count = 0
        xs = []
        if n_count == 0:
            #print(n_count)
            file_list = glob.glob(data_dir+'/*.png')  # get name list of all .png files
            for i in range(len(file_list)):
                img = dg.cv2.imread(file_list[i], 1)  # RGB scale
                xs.append(img)
            # xs = xs.reshape((xs.shape[0]*xs.shape[1],xs.shape[2],xs.shape[3], xs.shape[4]))
            # xs = dg.datagenerator(data_dir)
            print("I am here!")
            # Get Training Data
            # train_y, temporary_y = train_test_split(total, train_size=(1655/2207), random_state=0)

            # # Get Validation & Testing Data
            # # xs, test_y = train_test_split(temporary_y, train_size=0.5, random_state=0)
            print("LENGTH: XS", len(xs))
            assert len(xs)%args.batch_size ==0, \
            log('make sure the last iteration has a full batchsize, this is important if you use batch normalization!')
            xs = xs.astype('float32')/255.0
            # # print("XS.shape: ", xs.shape[0])
            indices = list(range(xs.shape[0]))
            # n_count = 1
        for _ in range(epoch_num):
            np.random.shuffle(indices)    # shuffle
            for i in range(0, len(indices), batch_size):
                batch_x = xs[indices[i:i+batch_size]]
                noise =  np.random.normal(0, args.sigma/255.0, batch_x.shape)    # noise
                #noise =  K.random_normal(ge_batch_y.shape, mean=0, stddev=args.sigma/255.0)
                batch_y = batch_x + noise 
                # for l in batch_y:
                    # print("FLAG: ", l)
                    # dg.show(batch_y[l])
                yield (batch_y, batch_x)
                # print ("BATCH Y    ", batch_y, "BATCH X    ", batch_x)


# def check_test(image_dir, image_name):
# #   #captions on the validation set
# #   rid = np.random.randint(0, len(test_image_names))
# #   image_name = test_image_names[rid]

# #   real_caption = image_dict[image_name]

#   image_path = image_dir + image_name + '.jpg'
#   #result = evaluate(image_path, max_caption_words, encoder, decoder)

#   # Displaying the image
#   original_image = load_image(image_path)
#   plt.axis('off')
#   plt.imshow(original_image)
# #   print('Real Captions:')
# #   for caption in real_caption:
# #     print(' ' + caption)
# #   print('\nPrediction Caption:\n', ' '.join(result))
#   return result

# def plotgraph():
#     # Get training and test loss histories
#     training_loss = hist.history['accuracy']
#     val_loss = hist.history['val_accuracy']

#     # Create count of the number of epochs
#     epoch_count = range(1, len(training_loss) + 1)

#     # Visualize loss history
#     plt.figure()
#     plt.plot(epoch_count, training_loss, 'r--')
#     plt.plot(epoch_count, val_loss, 'b-')
#     plt.legend(['Training Accuracy', 'Validation Accuracy'])
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.show()

# def display_image(y_true, y_pred):
#     original_image = load_image(image_path)
#     plt.axis('off')
#     plt.imshow(original_image)

# # The main directory containing the dataset
# dataset_dir = './datasets/'
# # The directory to access the images
# images_dir =  dataset_dir + 'Flicker8k_Dataset/'

#def gen_patches(file_name):

    # read image
    img = cv2.imread(file_name, 1)  # RGB scale
    h, w, l = img.shape
    # show(img)
    # print("Height: ", h, " Width: ", w, ", Length: ", l)
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h*s),int(w*s)
        img_scaled = cv2.resize(img, (h_scaled,w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled-patch_size+1, stride):
            for j in range(0, w_scaled-patch_size+1, stride):
                x = img_scaled[i:i+patch_size, j:j+patch_size, :]
                
                # temp = x
                # if (i == 0 and j == 0):
                #     print("Here: ", x)
                #patches.append(x)        
                # data aug
                # print("X", x)
                # if np.mean(x) == 255:
                #     continue
                    # print('***************All white************************')
                # else:
                #     print('Not all white')
                # for a in x:
                #     print("a: ", a)
                    # for b in a:
                        # print("B: ", b)
                    # if np.mean(i) != 255.0:
                #         print("Before X: ", i)
                #         print("np.mean:   ----   ", np.mean(i))
                #         print("After: ", i)
                # if np.mean(temp) != 255:
                for k in range(0, aug_times):
                    # if np.mean(temp) == 255:
                    #     continue
                        # x = img_scaled[i:i+patch_size, j:j+patch_size]
                    x_aug = data_aug(x, mode=np.random.randint(0,8))
                    patches.append(x_aug)
    # print("Patches: ", len(patches))
    return patches


# def datagenerator1(data_dir='data/newimgs',verbose=False):

    
    file_list = glob.glob(data_dir+'/*.png')  # get name list of all .png files
    # initialize
    data = []

    for i in range(len(file_list)):
        # print("Loop: ", i)
        patch = gen_patches(file_list[i])
        data.append(patch)
        if verbose:
            print(str(i+1)+'/'+ str(len(file_list)) + ' is done ^_^')
    data = np.array(data, dtype='uint8')
    # print("Data", data)
    print("Data.shape ", data.shape) #(No images, patches, dimns)
    data = data.reshape((data.shape[0]*data.shape[1],data.shape[2],data.shape[3], data.shape[4])) ## of parameters must be same as prev but change values to 3d
    #data = data.reshape((data.shape[0],data.shape[1],data.shape[3], 1)) ## of parameters must be same as prev but change values to 3d

    print("Data.shape  NEW ", data.shape)
    discard_n = len(data)-len(data)//batch_size*batch_size;
    data = np.delete(data,range(discard_n),axis = 0)
    print('^_^-training data finished-^_^')
    return data
    
