from utilities import *

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, 
    ystart, 
    ystop, 
    xstart,
    xstop,
    scale, 
    svc, 
    X_scaler, 
    orient, 
    pix_per_cell, 
    cell_per_block, 
    spatial_size, 
    hist_bins,
    window_size, FE,
    use_color,
    use_hog,
    hog_channel,
    DEBUG_BOX):
    """
    """

    img = image_scale_to_01(img)
    img_tosearch = crop_image(img, (ystart, ystop), (xstart, xstop))

    #ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YUV')
    ctrans_tosearch = resize_image_by_scale(ctrans_tosearch, scale)

    # Define blocks and steps as above
    nxblocks = (ctrans_tosearch.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ctrans_tosearch.shape[0] // pix_per_cell) - cell_per_block + 1

    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = window_size[0]
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog_rtn= FE._get_hog_unravel(ctrans_tosearch, {'hog_channel': hog_channel}, vis=False, feature_vec=False)

    bbox_list = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            if use_hog:
            # Extract HOG for this patch
                hog_features = FE._get_sub_hog_features(hog_rtn, 
                    {'hog_channel': hog_channel}, 
                    (ypos, ypos+nblocks_per_window),
                    (xpos, xpos+nblocks_per_window))
                test_features = X_scaler.transform(np.hstack(( hog_features)).reshape(1, -1))
            else:
                hog_features = np.asarray([])

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
                # Extract the image patch
            #subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], window_size)
            subimg = ctrans_tosearch[ytop:ytop+window, xleft:xleft+window]

            # Get color features
            if use_color:

                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)
            else:
                spatial_features = np.asarray([])
                hist_features = np.asarray([])

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            #test_features = X_scaler.transform(np.hstack((hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)


            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                box = ((xbox_left+xstart, ytop_draw+ystart),\
                    (xbox_left+win_draw+xstart,ytop_draw+win_draw+ystart))
                bbox_list.append(box)
            if DEBUG_BOX == 'GRID':
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                box = ((xbox_left+xstart, ytop_draw+ystart),\
                    (xbox_left+win_draw+xstart,ytop_draw+win_draw+ystart))
                bbox_list.append(box)
    return bbox_list


def draw_boxes(image, box):
    """
    Single image
    """
    draw_img = np.copy(image)
    cv2.rectangle(draw_img,box[0], box[1],(0,0,255),6)
    return draw_img
