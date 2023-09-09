import matplotlib.pyplot as plt

from extraction import extraction_data_transforms
from poses import Poses, merge_two_images
from numpy import ndarray
import numpy as np
import cv2
import matplotlib

# matplotlib.use('TkAgg')

if __name__ == '__main__':
    path_transforms = 'data\\transforms.json'
    camera_intrinsics, cameras_extrinsic = extraction_data_transforms(path_transforms)
    # cameras_extrinsic = np.concatenate((cameras_extrinsic[-2:], cameras_extrinsic[:2]))
    sample_rate = 1
    cameras_extrinsic = cameras_extrinsic[::sample_rate]

    poses = Poses(cameras_extrinsic, 15 / sample_rate)
    # poses.show_poses()
    poses.complete_path()
    # poses.show_poses()
    a = poses.get_generated_poses()
    b = poses.get_cut_indices()
    print(b)
    poses.show_all_stages(boundaries=3)

    merge_two_images('img.png', 'img_1.png', 'output.png')

    base_image_path = 'image_{num}.png'
    num = 0
    generator = poses.generate_interactive_images(boundaries=5)
    try:
        while True:
            next(generator)
            answer = generator.send(base_image_path.format(num=num))
            num += 1
            print(answer)
    except StopIteration:
        print('Done generating images.')
        # Using cv2.imshow() method
        # Displaying the image
        # cv2.imshow('interactive', np.asarray(plot))
        #
        # # waits for user to press any key
        # # (this is necessary to avoid Python kernel form crashing)
        # cv2.waitKey(1000)
        #
        # # closing all open windows
        # cv2.destroyAllWindows()
        plt.imshow(plot)
        plt.show()
    i = 0
