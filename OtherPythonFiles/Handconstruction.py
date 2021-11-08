vidcap = ocv.VideoCapture(1400)

with mp_holistic.Holistic(
        min_detection_confidence=holistic_min_detection_confidence,
        min_tracking_confidence=holistic_min_tracking_confidence)\
        as holistic:
    while vidcap.isOpened():

        # success is the boolean and image is the video frame output
        success, image = vidcap.read()

        # Selfie mode controlqqqq
        if ocv.waitKey(5) & 0xFF == ord('f'):
            flip_image = not flip_image
            # uncomment to test flip state
            # print(flip_image)

        if flip_image:
            image = ocv.flip(image, 1)

        # Camera Video Feed is just an arbitrary window name
        ocv.imshow('Camera Video Feed', image)

        # Exit Feed (using q key)
        # reason for 0xff is waitKey() returns 32 bit integer but key input(Ascii) is 8 bit so u want rest of 32 to be 0 as 0xFF = 11111111 and & is bitwise operator
        if ocv.waitKey(5) & 0xFF == ord('q'):
            exit_ = not exit_
        if exit_:
            break
vidcap.release()
ocv.destroyAllWindows()
