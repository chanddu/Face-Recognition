import numpy as np
from skimage import io
from dlib import get_frontal_face_detector, shape_predictor, face_recognition_model_v1

class FaceRecognition:
    def __init__(self):
        # HOG Face Detection
        self.face_detector = get_frontal_face_detector()
        self.pose_estimator = shape_predictor('./models/shape_predictor_68_face_landmarks.dat')
        self.face_encoder = face_recognition_model_v1('./models/dlib_face_recognition_resnet_model_v1.dat')

    def load_image_file(self, file):
        """
        Loads the image
            :param self: 
            :param file: Image to load
        """   
        return io.imread(file)

    def find_face_locations(self, image, upsample_factor=1):
        """
        Return the locations of faces in the given image
            :param self: 
            :param image: Image in which the face locations have to be determined
            :param upsample_factor=1: 
            :return: Returns a dlib iterable containing dlib rect objects of the face locations
        """   
        return self.face_detector(image, upsample_factor)

    def find_face_landmarks(self, img, face_locations=None):
        """
        Returns the landmarks/parts for each face in img
            :param self: 
            :param img: Image for which the Face landmarks have to be returned
        """

        # The 1 in the second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        if face_locations is None:
            face_locations = self.find_face_locations(img)
        return [self.pose_estimator(img, d) for k, d in enumerate(face_locations)]
            
    def face_embeddings(self, image, face_locations=None, num_itters=10):
        """
        Maps human faces into 128D vectors where pictures of the same person are mapped near to
        each other and pictures of different people are mapped far apart.
            :param self: 
            :param image: Image containing faces
            :param num_itters=1: Number of times to re-sample the face when calculating encoding.
            :return: Vector of 128-D Face Embeddings
        """   
        landmarks = self.find_face_landmarks(image, face_locations)
        return [np.array(self.face_encoder.compute_face_descriptor(image, land_mark, num_itters)) for land_mark in landmarks]

    def compute_eucledian_distance(self, face_descriptor1, face_descriptor2):
        """
        Computes the eucledian distance between the given face embeddings
            :param self: 
            :param face_descriptor1: Face embeddig 1, usually the descriptor for which the label is known
            :param face_descriptor2: Face embedding 2, usually the descriptor for which the label is unknown
        """   
        if len(face_descriptor1) == 0:
            return np.empty((0))
        return np.linalg.norm(face_descriptor1 - face_descriptor2, axis=1)

    def compare_faces(self, known_face_embeddings, unknown_face_embeddings, threshold = 0.6):        
        """
        Computes the Eucledian distance between the given face descriptors to see if there is a match
            :param self: 
            :param known_face_embeddings: list of descriptors of faces whose label is known
            :param unknown_face_embeddings: list of descriptors of face whose label is un-known
            :param threshold=0.6: 
            In general, if two face descriptor vectors have a eucledian deistance between them less than 0.6
            then they are from the same person, otherwise they are from different people.
        """
        return list(self.compute_eucledian_distance(known_face_embeddings, unknown_face_embeddings) <= threshold)