import mediapipe as mp
import numpy as np
import cv2 as ocv
import os
import timeit
start = timeit.default_timer()
# mp_holistic Parameters
static_image_mode = False
model_complexity = 2  # set to 1 if on weaker hardware
smooth_landmarks = True
enable_segmentation = False
smooth_segmentation = False
holistic_min_detection_confidence = 0.5
holistic_min_tracking_confidence = 0.5


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


def mediapipe_opencv_transform(image, mp_model):
    # Color space transform from ocv to mediapipe
    image = ocv.cvtColor(image, ocv.COLOR_BGR2RGB)
    image.flags.writeable = False  # Set Image Array to read only(immutable)
    results = mp_model.process(image)  # Run model on the image array
    # Set Image Array to be writable again(mutable)
    image.flags.writeable = True
    # Color space transform from mediapipe to ocv
    image = ocv.cvtColor(image, ocv.COLOR_RGB2BGR)
    return image, results


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten(
    ) if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten(
    ) if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
    ) if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
    ) if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh, pose, face])


def landmarks(iamge, results):
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    pass


vidcap = ocv.VideoCapture(r'G:\code\Y2S1\CS3244\Data\s1.mp4')  # the files here

with mp_holistic.Holistic(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        smooth_landmarks=smooth_landmarks,
        smooth_segmentation=smooth_segmentation,
        min_detection_confidence=holistic_min_detection_confidence,
        min_tracking_confidence=holistic_min_tracking_confidence)\
        as holistic:
    while vidcap.isOpened():
        # Camera input
        # success is the boolean and image is the video frame output
        success, image = vidcap.read()
        if not success:
            vidcap.release()
            break
        # if success:
        #     ocv.imshow('yourmom', image)
        # Run Model on Input and draw landmarks
        image, results = mediapipe_opencv_transform(image, holistic)
        keypoints = extract_keypoints(results)
        # npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
        # np.save(npy_path, keypoints)


# exit_ reset to False is here because if you dont rerun the notebook and rather rerun the cell exit would be set to true
exit_ = False
ocv.destroyAllWindows()
stop = timeit.default_timer()
print('Time: ', stop - start)
