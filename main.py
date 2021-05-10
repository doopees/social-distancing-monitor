# Import from local library
from monitor import *

# Define some constants
nframe = 0
w, h = 371, 528

# Define the targets and corners used to define the
# transformation matrix
offset = np.float32([127, 0])
targets = offset + np.float32([[0, 0], [117, 0], [117, 528], [0, 528]])
corners = np.float32([[745, 0], [890, 10], [379, 540], [18, 455]])

# Initialize the utility classes
frames = FramesContainer()
persons = PersonsContainer()
calculator = Calculator(corners, targets, (w, h))
painter = Painter()

# Get the iterator with the data of each frame
frames_data = get_frames_data()

# Initialize the video stream
stream = FileVideoStream('TownCentreXVID.avi')

# Start the stream
stream.start()

# Get the initial time and save it
timestamp = time()
start = timestamp

# Loop over frames from the video stream
while stream.running():
    # If there is no more information left, exit the loop
    try:
        frame_info = next(frames_data)
    except:
        break

    # Grab the current frame
    picture = stream.read()

    # Make the current frame object
    frame = Frame(nframe)

    # For every person in the frame
    for person_info in frame_info:
        # Get the person's information
        person_id, _, left, top, right, bot = person_info
        # Create the person object if appearing for first time
        if person_id not in persons:
            person = Person(person_id)
            # Add the new person object to the container
            persons.append(person)
        else:
            # Else, fetch the person from the persons container
            person = persons[person_id]
        # Calculate the center, position and bounding box of the person
        center, position, rectangle = calculator.process(left, top, right, bot)
        # Add the calculated information to the person object
        person.set_info(center, position, rectangle)
        # Add the current frame to the person object, and vice versa
        person.add_frame(nframe)
        frame.add_person(person)

    # Calculate the bird view for every person in the frame
    # and update their bird_view attribute
    bird_views = calculator.get_bird_views(frame.positions)
    frame.update_persons_birdviews(bird_views)

    # Create the canvas over which the bird view points are to
    # be painted
    canvas = np.zeros((h, w, 3), dtype='uint8')

    # For every person in the frame, paint their bounding box
    # and bird view. Initially, it is asumed that the person in safe.
    for person in frame.persons:
        painter.paint_bbox(picture, person.rectangle, 'safe')
        painter.paint_birdview(canvas, person.bird_view, 'safe')

    # Get the pairs of persons that are close to each other
    at_risk_pairs = calculator.get_at_risk_pairs(frame)

    # For every pair found, paint thei bounding box, bird view
    # connect their centers in the camera view and their positions
    # in the bird eye view. It is no longer asumed that the persons are safe.
    for p1, p2 in at_risk_pairs:
        painter.paint_bbox(picture, p1.rectangle, 'at_risk')
        painter.paint_bbox(picture, p2.rectangle, 'at_risk')
        painter.paint_birdview(canvas, p1.bird_view, 'at_risk')
        painter.paint_birdview(canvas, p2.bird_view, 'at_risk')
        painter.paint_connect(picture, p1.center, p2.center, 'at_risk')
        painter.paint_connect(canvas, p1.bird_view, p2.bird_view, 'at_risk')
        # Update the pair info
        calculator.update_pair_info(frame, picture, p1, p2)

    # Append the current frame to the frames container, and check
    # if it is the worst frame seen so far, as in the frame with
    # the largest number of persons exposed.
    frames.append(frame)
    frames.check_if_worst(frame, picture, canvas)

    # Resize the canvas so that it has the same height as picture
    canvas = cv2.resize(canvas, (379, 540))

    # Make sure that the video is displayed at 25 fps
    while time() - timestamp < 1/fps:
        pass

    # Show the camera view and bird eye view concatenated
    picture = np.concatenate([picture, canvas], axis=1)
    cv2.imshow('Social distancing detector', picture)

    # Save the current time
    timestamp = time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    nframe += 1

# When the stream has finished, destroy all windows
cv2.destroyAllWindows()

# Show the duration of the displayed video
print(f'Seconds played: {timestamp-start:.2f}')
