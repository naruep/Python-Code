import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose

# Setup the Pose function for images.
pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)

# Setup the Pose function for videos.
pose_video = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)

# Initialize mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils

def detectPose(image, pose, draw=False, display=False):
    '''
    This function performs the pose detection on the most prominent person in an image.
    Args:
        image:   The input image with a prominent person whose pose landmarks needs to be detected.
        pose:    The pose function required to perform the pose detection.
        draw:    A boolean value that is if set to true the function draw pose landmarks on the output image.
        display: A boolean value that is if set to true the function displays the original input image, and the
                 resultant image and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn if it was specified.
        results:      The output of the pose landmarks detection on the input image.
    '''

    # Create a copy of the input image.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Pose Detection.
    results = pose.process(imageRGB)

    # Check if any landmarks are detected and are specified to be drawn.
    if results.pose_landmarks and draw:
        # Draw Pose Landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255),
                                                                               thickness=3, circle_radius=3),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0),
                                                                                 thickness=2, circle_radius=2))

    # Check if the original input image and the resultant image are specified to be displayed.
    if display:

        # Display the original input image and the resultant image.
        plt.figure(figsize=[22, 22])
        plt.subplot(121);
        plt.imshow(image[:, :, ::-1]);
        plt.title("Original Image");
        plt.axis('off');
        plt.subplot(122);
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output Image");
        plt.axis('off');

    # Otherwise
    else:

        # Return the output image and the results of pose landmarks detection.
        return output_image, results

def checkJumpCrouch(image, results, MID_Y=250, draw=False, display=False):
    '''
    This function checks the posture (Jumping, Crouching or Standing) of the person in an image.
    Args:
        image:   The input image with a prominent person whose the posture needs to be checked.
        results: The output of the pose landmarks detection on the input image.
        MID_Y:   The intial center y-coordinate of both shoulders landmarks of the person recorded during starting
                 the game. This will give the idea of the person's height when he is standing straight.
        draw:    A boolean value that is if set to true the function writes the posture on the output image.
        display: A boolean value that is if set to true the function displays the resultant image and returns nothing.
    Returns:
        output_image: The input image with the person's posture written, if it was specified.
        posture:      The posture (Jumping, Crouching or Standing) of the person in an image.
    '''

    # Get the height and width of the image.
    height, width, _ = image.shape

    # Create a copy of the input image to write the posture label on.
    output_image = image.copy()

    # Retreive the y-coordinate of the left shoulder landmark.
    left_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)

    # Retreive the y-coordinate of the right shoulder landmark.
    right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height)

    # Calculate the y-coordinate of the mid-point of both shoulders.
    actual_mid_y = abs(right_y + left_y) // 2

    # Calculate the upper and lower bounds of the threshold.
    lower_bound = MID_Y - 15
    upper_bound = MID_Y + 100

    # Check if the person has jumped that is when the y-coordinate of the mid-point
    # of both shoulders is less than the lower bound.
    if (actual_mid_y < lower_bound):

        # Set the posture to jumping.
        posture = 'Jumping'

    # Check if the person has crouched that is when the y-coordinate of the mid-point
    # of both shoulders is greater than the upper bound.
    elif (actual_mid_y > upper_bound):

        # Set the posture to crouching.
        posture = 'Crouching'

    # Otherwise the person is standing and the y-coordinate of the mid-point
    # of both shoulders is between the upper and lower bounds.
    else:

        # Set the posture to Standing straight.
        posture = 'Standing'

    # Check if the posture and a horizontal line at the threshold is specified to be drawn.
    if draw:
        # Write the posture of the person on the image.
        cv2.putText(output_image, posture, (5, height - 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

        # Draw a line at the intial center y-coordinate of the person (threshold).
        cv2.line(output_image, (0, MID_Y), (width, MID_Y), (255, 255, 255), 2)

    # Check if the output image is specified to be displayed.
    if display:

        # Display the output image.
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output Image");
        plt.axis('off');

    # Otherwise
    else:

        # Return the output image and posture indicating whether the person is standing straight or has jumped, or crouched.
        return output_image, posture

# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture(0)
camera_video.set(3, 1280)
camera_video.set(4, 960)

# Create named window for resizing purposes.
cv2.namedWindow('Verticial Movements', cv2.WINDOW_NORMAL)

# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():

    # Read a frame.
    ok, frame = camera_video.read()

    # Check if frame is not read properly then continue to the next iteration to read the next frame.
    if not ok:
        continue

    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)

    # Get the height and width of the frame of the webcam video.
    frame_height, frame_width, _ = frame.shape

    # Perform the pose detection on the frame.
    frame, results = detectPose(frame, pose_video, draw=True)

    # Check if the pose landmarks in the frame are detected.
    if results.pose_landmarks:
        # Check the posture (jumping, crouching or standing) of the person in the frame.
        frame, _ = checkJumpCrouch(frame, results, draw=True)

    # Display the frame.
    cv2.imshow('Verticial Movements', frame)

    # Wait for 1ms. If a a key is pressed, retreive the ASCII code of the key.
    k = cv2.waitKey(1) & 0xFF

    # Check if 'ESC' is pressed and break the loop.
    if (k == 27):
        break

# Release the VideoCapture Object and close the windows.
cv2.moveWindow('Verticial Movements', 0, 0)
camera_video.release()
cv2.destroyAllWindows()
