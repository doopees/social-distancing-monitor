# Import from standard library
from queue import Queue
from random import choice
from time import time, sleep
from threading import Thread
from operator import attrgetter
from itertools import combinations

# Import third party libraries
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Set seaborn style
sns.set_style('ticks')


# Define some constants
fps = 25
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)


class FileVideoStream:
    """Allows fast video streaming with OpenCV by using multithreading."""

    def __init__(self, path, queue_size=128):
        """
        Initialize the file video stream and set the boolean used to
        indicate if the thread should be stopped or not.

        Input
            path: the path to the video file
            queueSize: the maximum number of frames to store in the queue
        """
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.queue = Queue(maxsize=queue_size)

    def start(self):
        """Start a thread to read frames from the file video stream."""
        thread = Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()

    def update(self):
        """
        Read and decode frames from the video file and maintain the
        queue data structure.
        """
        while True:
            sleep(0.001)
            if self.stopped:
                break
            if not self.queue.full():
                # Read the next frame from the file and resize it
                grabbed, frame = self.stream.read()
                frame = cv2.resize(frame, (960, 540))
                # If the end of the video has been reached
                if not grabbed:
                    self.stop()
                    break
                # Add the frame to the queue
                self.queue.put(frame)

    def read(self):
        """Return the next frame in the queue."""
        return self.queue.get()

    def stop(self):
        """Indicate that the thread should be stopped."""
        self.stopped = True

    def running(self):
        """Indicate that the video stream has not finished yet."""
        return not self.stopped


class Person:
    """A person that appears in the video stream"""

    def __init__(self, id_):
        """
        Create a Person object and set his frames and encountered
        dictionaries to be initially empty.

        self.frames has the frames in which the person appears as keys,
        and the number of persons within 2 meters of him as the
        corresponding values.

        self.encountered has the persons encountered (distance inferior
        to 2 meters) as keys, and the number of frames spent close to the
        other person as values.

        Input
            id_: the identfier of the person
        """
        self.id = id_
        self.encountered = {}
        self.frames = {}

    def __repr__(self):
        """Return the string representation of the person."""
        return f'Person {self.id}'

    def set_info(self, center, position, rectangle):
        """
        Set some additional attributes to the person.

        Input
            center: a tuple(x, y) corresponding to the center of
                    the bounding box of the person
            position: a tuple (x, y) corresponding to the coordinates
                      of the person's feet, as this is used to represent
                      location
            rectangle: the list [(x_left, y_top), (x_right, y_bot)]
                       representing two corners of the person's bounding box
        """
        self.center = center
        self.position = position
        self.rectangle = rectangle

    def dataframe(self):
        """
        Return a dataframe with columns time and persons_within_2m.

        time contains the time instants in which the person appears in
        the video.

        persons_within_2m indicates how many persons are close at the
        corresponding time.
        """
        time = list(self.frames.keys())
        time = [t/fps for t in time]
        persons = list(self.frames.values())
        df = pd.DataFrame(data=[time, persons]).T
        df.columns = ('time', 'persons_within_2m')
        return df

    def scatterplot(self):
        """
        Show the scatter plot for the dataframe returned by
        self.dataframe().
        """
        df = self.dataframe()
        sns.scatterplot(x='time', y='persons_within_2m', data=df)
        sns.despine(trim=True)
        plt.show()

    def pieplot(self):
        """
        Show a pie chart with two slices corresponding to the total time
        in which the person is safe (no other person within 2 meters) and the
        total time in which the person is at risk.
        """
        frames_safe = self.num_frames - self.num_frames_exposed
        frames_risk = self.num_frames_exposed
        plt.pie([frames_safe, frames_risk], labels=['safe', 'at_risk'],
                shadow=True, autopct='%1.2f%%',
                wedgeprops=dict(linewidth=1, ec='k'))
        plt.title(f'person {self.id} status on {self.screen_time} s')
        plt.show()

    def add_frame(self, nframe):
        """
        Add a new frame in which the person appears to the frames
        dictionary and set the number of surrounding persons to zero.

        Input
            nframe: the frame number to add
        """
        self.frames[nframe] = 0

    def show_worst_frame(self, painter, by='others', show=True):
        """
        Display the worst frame for the person as defined by by.
        A blue bounding box is drawn around the person to identify him.

        Input
            by: a string indicating the desired worst frame to show
                'others': the worst frame is defined as the frame with
                          the largest number of persons within 2 meters.
                'time': actually used the same frame as 'others', but displays
                        diferent labels. Not meant to use when calling this
                        method by itself. Instead, just serves as an utility
                        parameter for other methods that call this method.
                Defaults to 'others'.
            show: a boolean indicating if it is desired to actually display the
                  worst picture. Not meant to use then calling show_worst_frame
                  by itself. Defaults to True.
        """
        copy = self.worst_picture.copy()
        (nframe, nwithin), time = self.worst_frame, self.time_exposed
        painter.paint_bbox(copy, self.worst_rectangle, 'showing')
        plt.imshow(copy[:,:,::-1])
        plt.title(f'frame {nframe} (time {nframe/fps} s)')
        if by == 'others':
            string = f'Person {self.id} - has {nwithin} persons around'
        if by == 'time':
            string = f'Person {self.id} - exposed for {time} s'
        plt.xlabel(string)
        plt.xticks([])
        plt.yticks([])
        if show:
            plt.show()

    @property
    def num_encounters(self):
        """Return the number of persons encountered."""
        return len(self.encountered)

    @property
    def worst_frame(self):
        """
        Return the tuple (frame_number, num_persons_around)
        corresponding to the frame with the largest number
        of persons arround.
        """
        return max(self.frames.items(), key=lambda x: x[1])

    @property
    def frames_exposed(self):
        """
        Filter the frames dictionary and return a dictionary
        containing just the frames in which the person is exposed.
        """
        return {k:v for k, v in self.frames.items() if v > 0}

    @property
    def num_frames_exposed(self):
        """Return the number of frames in which the person is exposed."""
        return len(self.frames_exposed)

    @property
    def time_exposed(self):
        """Return the total time in which the person is exposed."""
        return self.num_frames_exposed / fps

    @property
    def num_frames(self):
        """Return the number of frames in which the person appears."""
        return len(self.frames)

    @property
    def screen_time(self):
        """Return the total screen time of the person."""
        return self.num_frames / fps


class PersonsContainer:
    """A list-like container for the persons in the video"""

    def __init__(self):
        """
        Initialize the persons dictionary as empty.
        Each entry in the dictionary has a person id as the
        key and the corresponding Person object as the value.
        """
        self.persons = {}

    def __len__(self):
        """Return the current number of persons in the container."""
        return len(self.persons)

    def __contains__(self, id_):
        """
        Check if the person with the given id is contained.

        Input
            id_: the desired person id to check
        """
        return id_ in self.persons

    def __getitem__(self, id_):
        """
        Return the person object with the given id

        Input
            id_: the id of the desired person
        """
        return self.persons[id_]

    def __iter__(self):
        """Return an iterator with all the persons in the container."""
        return iter(self.persons.values())

    def append(self, person):
        """
        Append a new person to the container.

        Input
            person: the Person object to append
        """
        id_ = person.id
        self.persons[id_] = person

    def random_person(self):
        """Return a random person from the container."""
        return choice(list(self.persons.values()))

    def most_exposed(self, by='others', n=1):
        """
        Return the n most exposed persons as defined
        by by

        Input
            by: the criterion by which the most exposed
                persons are defined
                'others': more exposition means more persons
                          encountered.
                'time': more exposition means more time around
                        other persons.
                Defaults to 'others'.
            n: the desired number of most exposed persons to return.
               Defaults to 1.
        """
        if by == 'others':
            getter = attrgetter('num_encounters')
        if by == 'time':
            getter = attrgetter('time_exposed')
        sorted_ = sorted(self, key=getter, reverse=True)
        if n == 1:
            return sorted_[0]
        else:
            return sorted_[:n]

    def show_exposed(self, painter, by='others'):
        """
        Display pictures of the four most exposed persons
        as defined by by.

        Input
            by: the criterion by which the most exposed
                persons are defined
                'others': exposition means persons encountered
                'time': exposition means time around other persons
                Defaults to 'others'
        """
        top = self.most_exposed(by=by, n=4)
        for i, person in enumerate(top):
            plt.subplot(2, 2, i+1)
            person.show_worst_frame(painter, by=by, show=False)
        plt.tight_layout()
        plt.show()

    def dataframe(self):
        """
        Return a dataframe where each entry is a person that
        has appeared on the video and the next columns

            safe: boolean indicating whether the person has
                  been exposed or not
            encounters: number of persons encountered
            time_on_screen: total time on screen
            time_exposed: total time of exposure by being close to others
            worst_time: time instant when the largest number of people
                        are around the person
            max_simult: number of people around on the worst time
        """
        data = []
        cols = ('safe', 'encounters', 'time_on_screen',
                'time_exposed', 'worst_time', 'max_simult')
        for person in self:
            data.append([
                person.num_encounters == 0,
                person.num_encounters,
                person.screen_time,
                person.time_exposed,
                person.worst_frame[0]/fps,
                person.worst_frame[1]
            ])
        df = pd.DataFrame(data, columns=cols)
        df.index.name = 'person'
        return df

    def datapoints(self, by='others', ignore_safe=False):
        """
        Return a pandas series with each entry corresponding
        to a person that has appeared on the video and the
        value associated with each entry being an attribute
        of the person, as defined by by.

        Input
            by: a string defining the attribute of interest
                'others': the attribute is the number of encounters
                'time': time attribute is the time exposed
                Defaults to 'others'.
            ignore_safe: boolean indicating whether to include or not
                         entries for the persons that had not been exposed
                         Defaults to False.
        """
        if by == 'others':
            getter = attrgetter('num_encounters')
            name = 'had contact with x persons'
        if by == 'time':
            getter = attrgetter('time_exposed')
            name = 'exposed for x seconds'
        data = np.array([getter(person) for person in self])
        if ignore_safe:
            data = data[data > 0]
        data = pd.Series(data, name=name)
        data.index.name = 'person id'
        return data

    def dataplot(self, by='others', ignore_safe=False):
        """
        Shows the scatter, swarm and boxplot for the data series
        returned by self.data(by, ignore_safe)

        Input
            by: a string defining the attribute of interest for the plot
                'others': the attribute is the number of encounters
                          Defaults to 'others'
                'time': time attribute is the time exposed
            ignore_safe: boolean indicating whether to include or not
                         the persons that had not been exposed
                         Defaults to False.
        """
        gridspec_kw = {'width_ratios': (3, 1)}
        data = self.datapoints(by, ignore_safe)
        fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw=gridspec_kw)
        sns.scatterplot(y=data, x=data.index, ax=ax1)
        sns.boxplot(y=data, palette='vlag', ax=ax2)
        sns.swarmplot(y=data, color='0.3', ax=ax2)
        ax2.set(ylabel='', yticks=[], xticks=[])
        sns.despine(ax=ax1)
        sns.despine(ax=ax2, left=True)
        fig.subplots_adjust(wspace=0, hspace=0)
        plt.show()

    def hist(self, by='others', ignore_safe=False):
        """
        Shows the histogram for the data series returned by
        self.data(by, ignore_safe).

        Input
            by: a string defining the attribute of interest for the plot
                'others': the attribute is the number of encounters
                          Defaults to 'others'
                'time': time attribute is the time exposed
            ignore_safe: boolean indicating whether to include or not
                         the persons that had not been exposed
                         Defaults to False.
        """
        data = self.datapoints(by, ignore_safe)
        avg_ = data.mean()
        max_ = data.max()
        if by == 'time':
            sns.distplot(data, kde=False, rug=True)
        if by == 'others':
            sns.distplot(data, kde=False, bins=1+max_,
                         hist_kws={'range': (-0.5, 0.5+max_)})
        plt.axvline(avg_, color='k', linestyle='--', linewidth=1)
        min_ylim, max_ylim = plt.ylim()
        if by == 'others':
            string = f'Mean: {avg_:.2f} persons ({len(data)} persons evaluated)'
            xlabel = '# of persons encountered'
        if by == 'time':
            string = f'Mean: {avg_:.2f} s ({len(data)} persons evaluated)'
            xlabel = 'exposure time (s)'
        plt.text(avg_*1.1, max_ylim*0.9, string)
        plt.xlabel(xlabel)
        plt.ylabel('frequency')
        sns.despine(trim=True)
        plt.show()


class Frame:
    """A frame that appears in the video stream"""

    def __init__(self, number):
        """
        Initialize the frame object with its corresponding
        number and an empty dict for the persons in the frame.
        The persons dictionary has persons as keys and whether
        the person is safe or at risk in the frame as values.

        Input
            number: the number of the frame in the video
        """
        self.number = number
        self.persons = {}  # Person: 'safe' or 'at_risk'

    def __setitem__(self, person, state):
        """
        Set the state 'safe' or 'at_risk' for the desired person.
        Input
            person: the person which state is desired to change
            state: the desired state for the person. Possible values
                   are 'safe' and 'at_risk'.
        """
        self.persons[person] = state

    def __repr__(self):
        """Return the string representation of the frame."""
        return f'Frame {self.number}'

    @property
    def safe_persons(self):
        """Return the Person objects in the frame with state 'safe'."""
        return [person
                for person, state in self.persons.items()
                if state == 'safe']

    @property
    def at_risk_persons(self):
        """Return the Person objects in the frame with state 'at_risk'."""
        return [person
                for person, state in self.persons.items()
                if state == 'at_risk']

    @property
    def num_persons(self):
        """Return the total number of persons in the frame."""
        return len(self.persons)

    @property
    def num_safe(self):
        """Return the number of safe persons in the frame."""
        return len(self.safe_persons)

    @property
    def num_at_risk(self):
        """Return the number of persons at risk in the frame."""
        return len(self.at_risk_persons)

    @property
    def positions(self):
        """
        Return a list containing the positions for all
        the persons in the frame.
        """
        return [person.position
                for person in self.persons]

    def add_person(self, person):
        """
        Add a new person to the frame and set his status to 'safe'

        Input
            person: the person to add to the frame
        """
        self[person] = 'safe'

    def update_persons_birdviews(self, bird_views):
        """
        Update the (x, y) coordinates of the bird view of every
        person in the frame.

        Input
            bird_views: a list containing the new bird view coordinates
                        for every person in the frame
        """
        for person, bird_view in zip(self.persons, bird_views):
            person.bird_view = bird_view

    def pieplot(self):
        """
        Show a pie chart with the percentage of safe and at risk persons
        in the frame.
        """
        data = [self.num_safe, self.num_at_risk]
        plt.pie(data, labels=['safe', 'at_risk'], shadow=True,
                autopct='%1.2f%%', wedgeprops=dict(linewidth=1, ec='k'))
        plt.title('persons status')
        plt.show()


class FramesContainer:
    """A list-like container for the frames in the video"""

    def __init__(self):
        """
        Initialize the frames list as empty. Each element in the
        list is a Frame object.
        """
        self.frames = []

    def __len__(self):
        """Return the number of frames seen so far."""
        return len(self.frames)

    def __getitem__(self, number):
        """
        Return the frame identified by the given number
        Input
            number: the number of the frame to return
        """
        return self.frames[number]

    @property
    def _safe_at_risk_persons(self):
        """
        Return a list of tuples with each tuple corresponding
        to a frame and with values equal to the number of safe
        and at risk persons in each frame.
        """
        return [(frame_.num_safe, frame_.num_at_risk)
                for frame_ in self.frames]

    def append(self, frame):
        """
        Append a new frame to the frames container.

        Input
            frame: the desired frame to append
        """
        self.frames.append(frame)

    def status_df(self):
        """
        Return a dataframe containing information about the
        number of safe and at risk persons in each frame.
        """
        df = pd.DataFrame(data=self._safe_at_risk_persons,
                          columns=('safe', 'at_risk'))
        df.index = df.index / fps
        df.index.name = 'time'
        return df

    @property
    def _worst(self):
        """Return the frame with the most number of persons at risk."""
        return max(self.frames, key=lambda x: x.num_at_risk)

    def check_if_worst(self, frame, picture, canvas):
        """
        Check if the given frame is the frame with the most
        number of persons at risk. If so, save the picture
        of the frame.

        Input
            frame: the frame to check
            picture: the picture corresponding to the given frame
            canvas: the canvas with the bird views of the persons
                    in the frame
        """
        if frame.num_at_risk == self._worst.num_at_risk:
            self._worst_picture = picture
            self._worst_birdview = canvas

    def stackplot(self):
        """Show the stack plot of the data returned by self.status_df()."""
        df = self.status_df()
        df.plot.area(alpha=0.7)
        sns.despine(trim=True)
        plt.title('persons vs. time')
        plt.xlabel('time (s)')
        plt.ylabel('persons')
        plt.margins(0, 0)
        plt.show()

    def lineplot(self):
        """Show the line plot of the data returned by self.status_df()."""
        df = self.status_df()
        sns.lineplot(data=df)
        sns.despine(trim=True)
        plt.title('persons vs. time')
        plt.xlabel('time (s)')
        plt.ylabel('persons')
        plt.show()

    def pieplot(self):
        """
        Show a pie chart with the percentage of safe and at risk persons
        for all the frames seen.
        """
        df = self.status_df()
        plt.pie(df.sum(), labels=df.columns, shadow=True,
                autopct='%1.2f%%', wedgeprops=dict(linewidth=1, ec='k'))
        plt.title('persons status')
        plt.show()

    def worst_frame(self):
        """Return the frame with the largest number of exposed persons"""
        return self._worst

    def worst_birdview(self):
        return self._worst_birdview

    def show_worst_frame(self):
        """Show the frame with the largest number of exposed persons."""
        frame = self.worst_frame()
        plt.imshow(self._worst_picture[:,:,::-1])
        plt.axis('off')
        plt.grid(False)
        plt.title(f'Frame #{frame.number} ' +
                  f'(Time: {frame.number/fps:.2f} s) - ' +
                  f'{frame.num_at_risk} persons exposed')
        plt.show()

    def show_worst_birdview(self):
        """Show the bird view with the largest number of exposed persons."""
        frame = self.worst_frame()
        canvas = self.worst_birdview()
        plt.imshow(canvas[:,:,::-1])
        plt.axis('off')
        plt.grid(False)
        plt.title(f'Frame #{frame.number} ' +
                  f'(Time: {frame.number/fps:.2f} s) - ' +
                  f'{frame.num_at_risk} persons exposed')
        plt.show()

    def random_frame(self):
        """Return a random person from the container."""
        return choice(list(self.frames))


class Calculator:
    """A calculator for performing operations on the video data."""

    def __init__(self, corners, targets, canvas_shape):
        """
        Initialize the calculator. self.matrix stores the transformation
        matrix used to calculate the bird view points for each position.

        Input
            corners: the corners of the road as seen in the camera view
            targets: the desired new positions in the canvas for the
                     corners of the street
            canvas_shape: a tuple with the dimensions of the canvas (w, h)
        """
        self.matrix = cv2.getPerspectiveTransform(corners, targets)
        self.shape = canvas_shape

    def process(self, left, top, right, bottom):
        """
        Return the center of the bounding box, the position (defined by
        coordinates of feet) and the corners of the bounding box for a
        single person.

        Input
            left: x coordinate of the left side of the bounding box
            top: y coordinate of the top side of the bounding box
            right: x coordinate of the right side of the bounding box
            bottom: y coordinate of the bottom side of the bounding box
        """
        center = [(left+right) // 2, (top+bottom)//2]
        position = [(left+right)//2, bottom]
        rectangle = [(left, top), (right, bottom)]
        return center, position, rectangle

    def get_bird_views(self, positions):
        """
        Return a list containing the (x, y) coordinates of the
        bird view of every position given in positions.

        Input
            positions: an array of the positions for which the bird view
                       is desired
        """
        bird_views = np.float32([positions])
        bird_views = cv2.perspectiveTransform(bird_views, self.matrix,
                                              self.shape)
        bird_views = np.squeeze(bird_views).astype('int32')
        return bird_views

    def get_at_risk_pairs(self, frame):
        """
        Return a list with every pair of persons withing 2 meters
        one of each other for the given frame.
        No repeated pairs are returned.

        Input
            frame: the frame for which it is wanted to find all
                   exposed pairs
        """
        pairs = []
        persons = frame.persons.keys()
        bird_views = [person.bird_view for person in persons]
        zipped = zip(persons, bird_views)
        for (p1, bv1), (p2, bv2) in combinations(zipped, 2):
            if np.linalg.norm(bv1 - bv2) < 43:
                pairs.append((p1, p2))
        return pairs

    def update_pair_info(self, frame, picture, p1, p2):
        """
        Update information for the given pair. If it's the first
        time they encounter, add each person to the other person
        encountered dictionary. Else, increment the number of times
        they've encountered each other.

        Input
            frame: the Frame object in which the pair appears
            picture: the picture of the given frame
            p1: person one of the pair
            p2: person two of the pair
        """
        self.update_person_info(frame, picture, p1)
        self.update_person_info(frame, picture, p2)
        if p1 not in p2.encountered:
            p1.encountered[p2] = 1
            p2.encountered[p1] = 1
        else:
            p1.encountered[p2] += 1
            p2.encountered[p1] += 1

    def update_person_info(self, frame, picture, person):
        """
        Set the person status to 'at_risk', increment the
        number of persons by which the person is surrounded by
        and determine whether the given frame corresponds to the
        worst frame for the person. If that's the case, save the
        frame picture as the worst picture for the person.

        Input
            frame: the frame object in which the person appears
            picture: the picture of the frame
            person: the person to update
        """
        frame[person] = 'at_risk'
        person.frames[frame.number] += 1
        if person.frames[frame.number] == person.worst_frame[1]:
            person.worst_picture = picture
            person.worst_rectangle = person.rectangle


class Painter:
    """Auxiliary class to paint shapes on pictures"""

    def __init__(self):
        """Initialize the painter."""
        pass

    def paint_bbox(self, picture, rectangle, status):
        """
        Paint the bounding box of a person in the frame.

        Input
            picture: the picture on which the painting is to be
                     performed
            rectangle: the corners of the bounding box
            status: 'safe' or 'at_risk', controls the color of
                    the bounding box
        """
        color = self.get_color(status)
        topleft, botright = rectangle
        cv2.rectangle(picture, topleft, botright, color, 3)

    def paint_birdview(self, canvas, bird_view, status):
        """
        Paint the bird_view of a person in the canvas.

        Input
            canvas: the canvas on which the painting is to be
                    performed
            bird_view: the (x, y) coordinates of the bird view
            color: 'safe' or 'at_risk', controls the color of the
                   bounding box
        """
        color = self.get_color(status)
        cv2.circle(canvas, tuple(bird_view), 5, color, -1)

    def paint_connect(self, picture, center1, center2, status):
        """
        Paint a connecting segment between center1 and center2 in the
        given picture.

        Input
            picture: the picture on which the painting is to be
                     performed
            center1: the first point that defines the segment to be painted
            center2: the second point that defines the segment to be painted
            color: 'safe' or 'at_risk', controls the color of the segment
        """
        color = self.get_color(status)
        cv2.line(picture, tuple(center1), tuple(center2), color, 3)

    def get_color(self, status):
        """Return the color corresponding to the given status."""
        return {
            'safe': YELLOW,
            'at_risk': RED,
            'showing': BLUE
        }.get(status)


def get_frames_data():
    """
    Return an iterator with each iteration returning the information
    about the current frame. Each information batch is a 2D numpy
    arrays of six columns, where for each column corresponds to
    id, frame, body_left, body_top, body_right and body_bottom respectively,
    and number of rows equal to the number of persons in the current frame.
    """
    cols = ['id', 'frame', 'body_left', 'body_top',
            'body_right', 'body_bottom']

    unused = ['head_valid', 'body_valid', 'head_left',
              'head_top', 'head_right', 'head_bottom']

    df = pd.read_csv('TownCentre-groundtruth.top', header=None)

    df = df.drop([2, 3, 4, 5, 6, 7], axis=1)
    df.iloc[:, -4:] = df.iloc[:, -4:] / 2
    df = df.astype(int)
    df.columns = cols

    nframes = max(df['frame'])
    frames_data = [df[df.frame == nframe] for nframe in range(nframes)]
    frames_data = [np.array(df) for df in frames_data]
    return iter(frames_data)
