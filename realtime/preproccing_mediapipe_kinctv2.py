from collections import namedtuple

Keypoint = namedtuple('Keypoint', ['x', 'y', 'z', 'confidence'])

# mp2vkv2: mediapipe to kinect v2
def mediapipe_to_kinect_v2(mediapipe_keypoints):
    """
    Convert keypoints from MediaPipe format to Kinect v2 format.

    Parameters:
    - mediapipe_keypoints: List of lists containing x, y, z, and visibility values for each keypoint.

    Returns:
    - kinect_v2_keypoints: List of tuples (x, y, z, confidence) for each keypoint.
    """
    # Mapping from MediaPipe keypoints to Kinect v2 keypoints mp2vkv2
    mediapipe_to_kinect_v2_mapping = {
        0: [24, 23],  # Spine Base
        1: [11, 12, 23, 24],  # Spine Mid
        2: [9, 10, 11, 12],   # Neck
        3: [0],   # Head
        4: [11],  # Left Shoulder
        5: [13],  # Left Elbow
        6: [15],  # Left Wrist
        7: [19, 17, 15],  # Left Hand
        8: [12],  # Right Shoulder
        9: [14],  # Right Elbow
        10: [16], # Right Wrist
        11: [16, 18, 20], # Right Hand
        12: [23], # Left Hip
        13: [25], # Left Knee
        14: [27], # Left Ankle
        15: [31], # Left Foot
        16: [24], # Right Hip
        17: [26], # Right Knee
        18: [28], # Right Ankle
        19: [32], # Right Foot
        20: [11, 12], # Spine Shoulder
        21: [17, 19], # Left Hand Tip
        22: [21], # Left Thumb
        23: [18, 20], # Right Hand Tip
        24: [22], # Right Thumb
    }

    kinect_v2_keypoints = []

    for kv2_indexes in mediapipe_to_kinect_v2_mapping.values():
        x_sum, y_sum, z_sum, max_confidence = 0, 0, 0, 0

        for mp_index in kv2_indexes:
            mp_keypoint = mediapipe_keypoints[mp_index]
            x_sum += mp_keypoint[0]
            y_sum += mp_keypoint[1]
            z_sum += mp_keypoint[2] if len(mp_keypoint) > 2 else 0
            max_confidence = max(max_confidence, mp_keypoint[3] if len(mp_keypoint) > 3 else 0)

        x_mean = x_sum / len(kv2_indexes)
        y_mean = y_sum / len(kv2_indexes)
        z_mean = z_sum / len(kv2_indexes)
        kinect_v2_keypoints.append((x_mean, y_mean, z_mean, max_confidence))

    return kinect_v2_keypoints